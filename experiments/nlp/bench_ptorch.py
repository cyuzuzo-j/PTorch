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
from ptorch.nn.modules import Linear, LinearMain, ReLU, MultiHeadAttention, Conversion, Mean, SumReLU, RMSNorm, ReLUSquared, CausalSelfAttention, CrossEntropyLoss
from ptorch.core.ops import CrossEntropyProjection, HardMarginProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from experiments.nlp.data import SST2DataModule
import tqdm, time
import wandb
import argparse
import copy
import torch.distributed as dist


FRAMEWORK = "ptorch_parr"

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ── Model ────────────────────────────────────────
class TextMLP(tnn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, classes, norm="l2", linear_cls=Linear):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()  # bridge right after embedding

        last = embed_dim
        self.hidden_layers = tnn.ModuleList()
        
        for f in hidden_dims:
            if linear_cls is Linear:
                self.hidden_layers.append(linear_cls(last, f, norm=norm))  # projection-based
            else:
                self.hidden_layers.append(linear_cls(last, f))
            self.hidden_layers.append(SumReLU())              # projection-based
            #self.norm = RMSNorm(f)
            last = f
        if linear_cls is Linear:
            self.out = linear_cls(last, classes, norm=norm, g=float('inf'))
        else:
            self.out = linear_cls(last, classes, g=float('inf'))

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded.mean(dim=1)
        x = self.conversion(x)   # bridge: gradient → projection (only for embedding)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)      # ptorch LinearBias
            x = self.hidden_layers[i + 1](x)  # ptorch ReLU
        return self.out(x)


class TinyAttention(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes, attention_type='simplex', norm="l2", num_heads=1, linear_cls=Linear):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, heads=num_heads, attention_type=attention_type, norm=norm)
        self.norm = RMSNorm(embed_dim)
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
        context = self.norm(context)
        
        pooled = self.mean(context) # B, E
        
        return self.out(pooled)


class GPTMLP(tnn.Module):
    def __init__(self, dim: int, mlp_mult: int, norm="l2", linear_cls=Linear):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = linear_cls(dim, hidden, bias=False, norm=norm) if linear_cls is Linear else linear_cls(dim, hidden, bias=False)
        self.act = ReLU()
        self.proj = linear_cls(hidden, dim, bias=False, norm=norm) if linear_cls is Linear else linear_cls(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class GPTBlock(tnn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_mult: int,
        norm="l2",
        linear_cls=Linear
    ):
        super().__init__()
        #self.attn_norm = RMSNorm(dim)
        #self.mlp_norm = RMSNorm(dim)
        
        # Using num_kv_heads = num_heads, rope_base = 10000.0, qk_gain_init = 1.5
        self.attn = CausalSelfAttention(dim, num_heads, num_heads, 10000.0, 1.5, norm=norm)
        self.mlp = GPTMLP(dim, mlp_mult, norm=norm, linear_cls=linear_cls)
        self.attn_scale = tnn.Parameter(torch.ones(1,dim, dtype=torch.float32))
        self.mlp_scale = tnn.Parameter(torch.ones(1,dim, dtype=torch.float32))
        self.resid_mix = tnn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = torch.add(torch.mul(mix[0][None, None, :], x), torch.mul(mix[1][None, None, :], x0))
        attn_out = self.attn(x)
        x = torch.add(x, torch.mul(self.attn_scale.to(dtype=x.dtype)[None, None, :], attn_out))
        x = x.squeeze()
        x = torch.add(x, torch.mul(self.mlp_scale.to(dtype=x.dtype)[None, None, :], self.mlp(x)))
        x = x.squeeze()
        return x

class TinyGPT(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes, num_layers=4, num_heads=1, mlp_mult=2, norm="l2", linear_cls=Linear):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()
        
        self.blocks = tnn.ModuleList([
            GPTBlock(embed_dim, num_heads, mlp_mult, norm=norm, linear_cls=linear_cls)
            for _ in range(num_layers)
        ])
        
        self.mean = Mean(dim=1)
        if linear_cls is Linear:
            self.out = linear_cls(embed_dim, classes, bias=False, norm=norm)
        else:
            self.out = linear_cls(embed_dim, classes, bias=False)
            
        
    def forward(self, x):
        x = self.embedding(x) # B, S, E
        x = self.conversion(x)  # bridge: gradient → projection
        x0 = x
        
        for block in self.blocks:
            x = block(x, x0)
            
        x = x
        pooled = self.mean(x) # B, E
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

    linear_impl = str(cfg.get("ptorch_linear_impl", "projection")).lower()
    if bool(cfg.get("ptorch_use_classic_linear", False)):
        linear_impl = "classic"

    if linear_impl == "classic":
        linear_cls = LinearClassic
    elif linear_impl == "main":
        linear_cls = LinearMain
    else:
        linear_cls = Linear

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
            linear_cls=linear_cls,
        ).to(device)
    elif model_name == "gpt":
        model = TinyGPT(
            vocab_size,
            task_cfg["embed_dim"],
            task_cfg["classes"],
            num_layers=cfg.get("num_layers", 4),
            num_heads=num_heads,
            mlp_mult=cfg.get("mlp_mult", 2),
            norm=model_norm,
            linear_cls=linear_cls,
        ).to(device)
    else:
        model = TinyAttention(
            vocab_size,
            task_cfg["embed_dim"],
            task_cfg["classes"],
            attention_type=cfg.get("attention_type", "simplex"),
            norm=model_norm,
            num_heads=num_heads,
            linear_cls=linear_cls,
        ).to(device)

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})

    if opt_name == "Muon":
        optimizers_proj = [Muon(model.parameters(), lr=opt_kwargs.get("lr", 0.02), momentum=0.95, backend_steps=5, weight_decay=opt_kwargs.get("weight_decay", 0.3))]
    else:
        optimizers_proj = [OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)]

    optimizers = optimizers_proj 
    for opt in optimizers_proj :
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]

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
        "ptorch_linear_impl": linear_impl,
        "ptorch_linear_class": linear_cls.__name__,
        **ptorch_config.snapshot(),
    })

    # GPU memory logging helper (works when device is CUDA)
    def _mb(x: int) -> float:
        return float(x) / (1024.0 * 1024.0)

    def log_gpu_mem(step: int, phase: str = ""):
        if not torch.cuda.is_available():
            return
        try:
            dev = device if getattr(device, 'type', None) == 'cuda' else None
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            peak = torch.cuda.max_memory_allocated(dev)
            wandb.log({
                f"gpu/{phase}allocated_mb": _mb(allocated),
                f"gpu/{phase}reserved_mb": _mb(reserved),
                f"gpu/{phase}peak_allocated_mb": _mb(peak),
            }, step=step)
        except Exception:
            # Best-effort logging; do not crash training on unexpected CUDA queries
            pass

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
        loss = CrossEntropyLoss()
        projected = loss(logits, y_oh)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        projected.sum().backward()
        loss = F.cross_entropy(logits.detach(), y.long())
        for opt in optimizers:
            opt.step()
        return loss

    def eval_fn(x, y):
        with torch.no_grad():
            return (model(x).argmax(dim=-1) == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    warmup_steps = cfg.get("warmup_steps", 0)
    if warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for _ in range(warmup_steps):
            x_w, y_w = next(train_iter)
            step_fn(torch.tensor(x_w, dtype=torch.long, device=device), torch.tensor(y_w, dtype=torch.long, device=device))
        model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Reset peak memory stats before main loop so peak is meaningful
    if torch.cuda.is_available():
        dev = device if getattr(device, 'type', None) == 'cuda' else None
        torch.cuda.reset_peak_memory_stats(dev)
    max_steps = cfg.get("max_steps", 0)
    warmdown_iters = cfg.get("warmdown_iters", int(0.1 * max_steps) if max_steps else 0)

    def lr_mul(step: int) -> float:
        if warmdown_iters <= 0 or not max_steps:
            return 1.0
        warmdown_start = max(max_steps - warmdown_iters, 0)
        return max((max_steps - step) / max(warmdown_iters, 1), 0.0) if warmdown_start <= step < max_steps else 1.0


    with tqdm.tqdm(unit="step") as pbar:
        while True:             

            scale = lr_mul(step)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group.get("base_lr", group["lr"]) * scale
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
                wandb.log({"val/val_acc": val_acc, "val/use_projections": bool(ptorch_config.use_projections)}, step=step)
                wandb.log({"training_time_s": time.time() - t0}, step=step)
                # Log GPU memory during evaluation
                log_gpu_mem(step, phase="val/")
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"switched paradigm {step}")
                    new_val = not ptorch_config.use_projections
                    ptorch_config.update("use_projections", new_val)
                    optimizers = optimizers_proj
                    if new_val:
                        for opt in optimizers:
                            for group in opt.param_groups:
                                group["lr"] = group["lr"] /2

                    no_improve = 0
            x, y = next(train_iter)
            x_t = torch.tensor(x, dtype=torch.long, device=device)
            y_t = torch.tensor(y, dtype=torch.long, device=device)
            loss = step_fn(x_t, y_t)
            loss_val = float(loss)
            # Compute training accuracy on the same batch and log it
            train_acc = float(eval_fn(x_t, y_t))
            log_dict = {"train/loss": loss_val, "train/train_acc": train_acc}
            wandb.log(log_dict, step=step)
            
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
    parser.add_argument('--model', choices=['mlp', 'attention', 'gpt'], default='gpt', help='Model choice')
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
