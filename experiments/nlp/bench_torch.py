##################################################
###   Benchmark — torch (PyTorch baseline)    ###
###   Standard Adam + cross-entropy           ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
from experiments.nlp.data import SST2DataModule
import tqdm, time
import wandb
import argparse

FRAMEWORK = "torch"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ── Models ───────────────────────────────────────
class TextMLP(tnn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, classes):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        
        last = embed_dim
        layers = []
        for f in hidden_dims:
            layers += [tnn.Linear(last, f), tnn.ReLU()]
            last = f
        self.body = tnn.Sequential(*layers)
        self.out  = tnn.Linear(last, classes)

    def forward(self, x):
        # x is (batch, seq_len)
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        # Mean pooling over sequence length
        pooled = embedded.mean(dim=1) # (batch, embed_dim)
        return self.out(self.body(pooled))

class TinyAttention(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        # Using PyTorch native MultiheadAttention
        self.attention = tnn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.out = tnn.Linear(embed_dim, classes)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        embedded = self.embedding(x) # B, S, E
        
        # Self-attention: query=key=value=embedded
        context, _ = self.attention(embedded, embedded, embedded) # B, S, E
        
        # Global average pool over sequence length
        pooled = context.mean(dim=1) # B, E
        
        return self.out(pooled)

# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device, model_name):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    # Note: Only SST2 is supported in this benchmark so far.
    if task_cfg["name"] != "SST2":
        raise NotImplementedError(f"Task {task_cfg['name']} not supported yet.")
        
    ds = SST2DataModule(
        batch_size=batch_size, 
        max_seq_len=task_cfg.get("max_seq_len", 64),
        vocab_size=task_cfg.get("vocab_size", 10000),
        seed=seed
    )
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    if model_name == "mlp":
        model = TextMLP(
            vocab_size=ds.vocab.get_piece_size() if hasattr(ds.vocab, 'get_piece_size') else len(ds.vocab), 
            embed_dim=task_cfg["embed_dim"], 
            hidden_dims=task_cfg["hidden_dim"], 
            classes=task_cfg["classes"]
        ).to(device)
    else:
        model = TinyAttention(
            vocab_size=ds.vocab.get_piece_size() if hasattr(ds.vocab, 'get_piece_size') else len(ds.vocab),
            embed_dim=task_cfg["embed_dim"],
            classes=task_cfg["classes"]
        ).to(device)
        
    opt_name   = cfg["torch_optimizer"]
    opt_kwargs = cfg.get("torch_optimizer_kwargs", {})
    optimizer  = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    run = wandb.init(project="pjax", name=f"{cfg['experiment_name']}_{model_name}")
    wandb.config.update({
        "framework": f"{FRAMEWORK}_{model_name}",
        "task": task_cfg["name"],
        "embed_dim": task_cfg.get("embed_dim"),
        "hidden": task_cfg.get("hidden_dim"),
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
    })

    def step_fn(x, y):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y.long())
        loss.backward()
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
                torch.tensor(y, dtype=torch.long,  device=device))
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
