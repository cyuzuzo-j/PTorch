##################################################
###   Benchmark — ptorch                      ###
###   Alternating projections (PyTorch)       ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import Linear, ReLU, MultiHeadAttention, Conversion, Mean, SumReLU
from ptorch.core.ops import CrossEntropyProjection, HardMarginProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from experiments.nlp.data import SST2DataModule
import tqdm, time
import wandb
import argparse

FRAMEWORK = "ptorch_parr"

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ── Model ────────────────────────────────────────
class TextMLP(tnn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, classes, norm="l2"):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()  # bridge right after embedding

        last = embed_dim
        self.hidden_layers = tnn.ModuleList()
        
        for f in hidden_dims:
            self.hidden_layers.append(Linear(last, f, norm=norm))  # projection-based
            self.hidden_layers.append(ReLU())              # projection-based
            last = f
        self.out = Linear(last, classes, norm=norm)

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.mean(dim=1)
        x = self.conversion(x)   # bridge: gradient → projection (only for embedding)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)      # ptorch LinearBias
            x = self.hidden_layers[i + 1](x)  # ptorch ReLU
        return self.out(x)


class TinyAttention(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes, attention_type='simplex', norm="l2", num_heads=1):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, heads=num_heads, attention_type=attention_type, norm=norm)
        self.out = Linear(embed_dim, classes, norm=norm)
        self.mean = Mean(dim=1)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        embedded = self.embedding(x) # B, S, E
        embedded = self.conversion(embedded)  # bridge: gradient → projection
        
        # Self-attention
        context = self.attention(embedded) # B, S, E
        
        pooled = self.mean(context) # B, E
        
        return self.out(pooled)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device, model_name):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    num_heads = int(cfg.get("attention_heads", 1))
    mlp_dropout = float(cfg.get("mlp_dropout", 0.0))
    if num_heads <= 0:
        raise ValueError(f"attention_heads must be >= 1, got {num_heads}")

    norm = cfg.get("ptorch_norm", 2)
    norm_str = str(norm).lower()
    if norm_str in ("linf", "inf"):
        model_norm = "linf"
    elif norm_str in ("l2", "2"):
        model_norm = "l2"
    else:
        raise ValueError(f"Unsupported ptorch_norm '{norm}'. Use one of: 2, l2, inf, linf.")

    if task_cfg["name"] != "SST2":
        raise NotImplementedError(f"Task {task_cfg['name']} not supported yet.")

    ds = SST2DataModule(
        batch_size=batch_size, 
        max_seq_len=task_cfg.get("max_seq_len", 64),
        vocab_size=task_cfg.get("vocab_size", 10000),
        seed=run_seed
    )
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    vocab_size = ds.vocab.get_piece_size() if hasattr(ds.vocab, 'get_piece_size') else len(ds.vocab)

    if model_name == "mlp":
        model = TextMLP(
            vocab_size, 
            task_cfg["embed_dim"], 
            task_cfg["hidden_dim"], 
            task_cfg["classes"],
            norm=model_norm,
        ).to(device)
    else:
        model = TinyAttention(
            vocab_size,
            task_cfg["embed_dim"],
            task_cfg["classes"],
            attention_type=cfg.get("attention_type", "simplex"),
            norm=model_norm,
            num_heads=num_heads,
        ).to(device)

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})

    optimizer       = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{model_name}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": f"{FRAMEWORK}_{model_name}",
        "task": task_cfg["name"],
        "embed_dim": task_cfg.get("embed_dim"),
        "hidden": task_cfg.get("hidden_dim"),
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": run_seed,
        "base_seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
        "attention_heads": num_heads,
        "mlp_dropout": mlp_dropout,
        "ptorch_norm": norm,
        "ptorch_model_norm": model_norm,
        **ptorch_config.snapshot(),
    })

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
        projected = HardMarginProjection.apply(logits, y_oh)
        optimizer.zero_grad()
        projected.sum().backward()
        loss = F.cross_entropy(logits.detach(), y.long())
        optimizer.step()
        return loss

    def eval_fn(x, y):
        with torch.no_grad():
            return (model(x).argmax(dim=-1) == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                model.eval()
                accs = []
                for x, y in val_loader:
                    x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
                    accs.append(eval_fn(x, y))
                if accs:
                    val_acc = float(torch.stack(accs).mean())
                else: 
                    val_acc = 0.0
                    
                model.train()
                wandb.log({"val/val_acc": val_acc}, step=step)
                wandb.log({"training_time_s": time.time() - t0}, step=step)
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            x, y = next(train_iter)
            loss = step_fn(
                torch.tensor(x, dtype=torch.long, device=device),
                torch.tensor(y, dtype=torch.long, device=device))
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    
    test_accs = []
    for x, y in test_loader:
        x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
        test_accs.append(eval_fn(x, y))
        
    if test_accs:
        final_acc = float(torch.stack(test_accs).mean())
    else:
        final_acc = 0.0
        
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc, best_val_acc, best_step, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'attention'], default='mlp', help='Model choice')
    args = parser.parse_args()
    
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} ({args.model}) | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device, args.model)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
