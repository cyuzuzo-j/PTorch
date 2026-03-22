##################################################
###   Benchmark — ptorch                      ###
###   Hybrid Alternating Attention            ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
#torch.set_float32_matmul_precision('high')
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import Linear, ReLU, MultiHeadAttention, Conversion, Mean
from ptorch.core.ops import CrossEntropyProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from experiments.nlp.data import SST2DataModule
import tqdm, time
import wandb
import argparse

FRAMEWORK = "ptorch_hybrid"
## Uses local config path
CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config_attention.yaml'))
OPTIM_MODULES = vars(ptorch_optim_static)

# ── Model ────────────────────────────────────────
class TinyAttention(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes, attention_type='simplex', norm="l2", num_heads=1, linear_cls=Linear):
        super().__init__()
        # Standard PyTorch Embedding
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        
        # Bridge: allows gradient mode to pass through, projection mode to convert
        self.conversion = Conversion()
        
        # Ptorch Attention
        self.attention = MultiHeadAttention(embed_dim, embed_dim, heads=num_heads, attention_type=attention_type, norm=norm)
        
        if linear_cls is Linear:
            self.out = linear_cls(embed_dim, classes, norm=norm)
        else:
            self.out = linear_cls(embed_dim, classes)
            
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
def run(cfg, task_cfg, batch_size, run_number, device, alternating_freq, loss_projection_cls=CrossEntropyProjection):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    num_heads = int(cfg.get("attention_heads", 1))

    norm = cfg.get("ptorch_norm", 2)
    norm_str = str(norm).lower()
    if norm_str in ("linf", "inf"):
        model_norm = "linf"
    else:
        model_norm = "l2"

    linear_cls = Linear

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

    model = TinyAttention(
        vocab_size,
        task_cfg["embed_dim"],
        task_cfg["classes"],
        attention_type=cfg.get("attention_type", "simplex"),
        norm=model_norm,
        num_heads=num_heads,
        linear_cls=linear_cls,
    ).to(device)

    # Standard PyTorch Adam is fine because optimizer logic handles gradients / projections targets based on global flag
    opt_name = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_attention_{task_cfg['name']}_bs{batch_size}_freq{alternating_freq}_run{run_number}"
    run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": f"{FRAMEWORK}",
        "task": task_cfg["name"],
        "embed_dim": task_cfg.get("embed_dim"),
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": run_seed,
        "base_seed": seed,
        "run_number": run_number,
        "alternating_freq": alternating_freq,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "attention_heads": num_heads,
        "ptorch_norm": model_norm,
    })

    def step_fn(x, y, use_projections):
        # Dynamically set the global projection context based on schedule
        with ptorch_config.config.projections(use_projections):
            logits = model(x)
            
            optimizer.zero_grad()
            if use_projections:
                y_oh = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
                projected_logits = loss_projection_cls.apply(logits, y_oh)
                proj_loss = projected_logits.sum()
                proj_loss.backward()
            else:
                loss = F.cross_entropy(logits, y.long())
                loss.backward()

            optimizer.step()
            
            with torch.no_grad():
                detached_loss = F.cross_entropy(logits.detach(), y.long()).item()
                
            return detached_loss

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
                with ptorch_config.config.projections(False): # Always evaluate purely forward without targets
                    model.eval()
                    accs = []
                    for x, y in val_loader:
                        x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
                        accs.append(eval_fn(x, y))
                    val_acc = float(torch.stack(accs).mean()) if accs else 0.0
                        
                    model.train()
                    
                wandb.log({"val/val_acc": val_acc}, step=step)
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
            x_device = torch.tensor(x, dtype=torch.long, device=device)
            y_device = torch.tensor(y, dtype=torch.long, device=device)
            
            # Simple alternating logic based on frequency parameter
            use_projections_flag = ((step // alternating_freq) % 2 == 1)
            
            loss = step_fn(x_device, y_device, use_projections_flag)
            
            wandb.log({"train/loss": float(loss), "train/use_projections": int(use_projections_flag)}, step=step)
            step += 1
            pbar.update(1)
            
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
        
    model.eval()
    test_accs = []
    with ptorch_config.config.projections(False):
        for x, y in test_loader:
            x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
            test_accs.append(eval_fn(x, y))
    final_acc = float(torch.stack(test_accs).mean()) if test_accs else 0.0
        
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.finish()
    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alternating-freq', type=int, default=None, help='Steps of grad followed by steps of proj')
    args = parser.parse_args()
    
    cfg    = yaml.safe_load(open(CFG_PATH))
    alternating_freq = args.alternating_freq if args.alternating_freq is not None else cfg.get('alternating_freq', 10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size} freq={alternating_freq}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device, alternating_freq)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
