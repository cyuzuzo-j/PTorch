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
from ptorch.nn.modules import Linear, LinearMain, ReLU, MultiHeadAttention, Conversion, Mean, SumReLU, RMSNorm, ReLUSquared, CausalSelfAttention, Softcap
from ptorch.core.ops import CrossEntropyProjection, HardMarginProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from experiments.nlp.data import SST2DataModule
import tqdm, time
import wandb
import argparse
import copy
import torch.distributed as dist

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if ptorch_config.use_projections:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.copy_(p.data - p.grad)

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


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
            self.hidden_layers.append(ReLU())              # projection-based
            self.norm = RMSNorm(f)
            last = f
        if linear_cls is Linear:
            self.out = linear_cls(last, classes, norm=norm)
        else:
            self.out = linear_cls(last, classes)

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
        self.act = ReLUSquared()
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
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        
        # Using num_kv_heads = num_heads, rope_base = 10000.0, qk_gain_init = 1.5
        self.attn = CausalSelfAttention(dim, num_heads, num_heads, 10000.0, 1.5, norm=norm)
        self.mlp = GPTMLP(dim, mlp_mult, norm=norm, linear_cls=linear_cls)
        self.attn_scale = tnn.Parameter(torch.ones(1, dim, dtype=torch.float32))
        self.mlp_scale = tnn.Parameter(torch.ones(1, dim, dtype=torch.float32))
        self.resid_mix = tnn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = torch.add(torch.mul(mix[0][None, None, :], x), torch.mul(mix[1][None, None, :], x0))
        attn_out = self.attn(self.attn_norm(x))
        x = torch.add(x, torch.mul(self.attn_scale.to(dtype=x.dtype), attn_out))
        x = torch.add(x, torch.mul(self.mlp_scale.to(dtype=x.dtype), self.mlp(self.mlp_norm(x))))
        return x

class TinyGPT(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes, num_layers=4, num_heads=1, mlp_mult=2, norm="l2", linear_cls=Linear):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()
        self.initial_norm = RMSNorm(embed_dim)
        
        self.blocks = tnn.ModuleList([
            GPTBlock(embed_dim, num_heads, mlp_mult, norm=norm, linear_cls=linear_cls)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(embed_dim)
        self.mean = Mean(dim=1)
        if linear_cls is Linear:
            self.out = linear_cls(embed_dim, classes, bias=False, norm=norm)
        else:
            self.out = linear_cls(embed_dim, classes, bias=False)
            
        self.cap = Softcap(logit_softcap=30.0)
        
    def forward(self, x):
        x = self.embedding(x) # B, S, E
        x = self.conversion(x)  # bridge: gradient → projection
        x = self.initial_norm(x)
        x0 = x
        
        for block in self.blocks:
            x = block(x, x0)
            
        x = self.final_norm(x)
        pooled = self.mean(x) # B, E
        return self.cap(self.out(pooled))


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
        optimizers = [Muon(model.parameters(), lr=opt_kwargs.get("lr", 0.02), momentum=0.95, backend_steps=5, weight_decay=opt_kwargs.get("weight_decay", 0.1))]
    else:
        optimizers = [OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)]

    for opt in optimizers:
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

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
        projected = CrossEntropyProjection.apply(logits, y_oh)
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

    max_steps = cfg.get("max_steps", 0)
    warmdown_iters = cfg.get("warmdown_iters", int(0.1 * max_steps) if max_steps else 0)

    def lr_mul(step: int) -> float:
        if warmdown_iters <= 0 or not max_steps:
            return 1.0
        warmdown_start = max(max_steps - warmdown_iters, 0)
        return max((max_steps - step) / max(warmdown_iters, 1), 0.0) if warmdown_start <= step < max_steps else 1.0

    alpha_start = float(cfg.get("projection_alpha_start", 0.001))
    alpha_end = float(cfg.get("projection_alpha_end", 100000.0))
    g_start = float(cfg.get("projection_g_start", 1.0))
    g_end = float(cfg.get("projection_g_end", 1.0))
    
    ema_loss = None
    ema_loss_slow = None
    current_alpha = alpha_start
    current_g = g_start

    # Added to control how aggressively alpha changes per step. 
    # Without this, alpha will instantly hit its max/min bounds.
    update_rate = 0.1  

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if ema_loss_slow is not None and ema_loss_slow > 0 and ptorch_config.use_projections:
                ratio = ema_loss / ema_loss_slow
                
                # ratio > 1.0: loss is increasing or stagnant -> positive adjustment
                # ratio < 1.0: loss is decreasing -> negative adjustment
                adjustment = ratio - 1.0
                
                # Clamp the adjustment to prevent exploding updates on massive loss spikes
                adjustment = max(-0.5, min(0.5, adjustment))
            else:
                adjustment = 0.0
            
            # Apply the update
            current_alpha += (alpha_end - alpha_start) * adjustment * update_rate
            current_g += (g_end - g_start) * adjustment * update_rate
            
            # CRITICAL: Clamp current_alpha so it doesn't drift below 0.1 or above 10
            current_alpha = max(alpha_start, min(alpha_end, current_alpha))                
            current_g = max(g_start, min(g_end, current_g))                
            ptorch_config.update("projection_g", current_g)
                
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
            loss_val = float(loss)
            if ema_loss is None:
                ema_loss = loss_val
                ema_loss_slow = loss_val
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * loss_val
                ema_loss_slow = 0.99 * ema_loss_slow + 0.01 * loss_val
                
            log_dict = {"train/loss": loss_val, "train/ema_loss": ema_loss}
            if max_steps > 0:
                log_dict["train/projection_alpha"] = current_alpha
                log_dict["train/projection_g"] = current_g
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
