##################################################
###   Benchmark — pjax (combined)              ###
###   Unified runner for `pjax` and `pjax_orr` ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import jax
import jax.numpy as jnp
import importlib
import argparse
import tqdm, time
import wandb

from experiments.nlp.data import SST2DataModule

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_impl(name):
    mod = importlib.import_module(name)
    nn = importlib.import_module(f"{name}.nn")
    optim = importlib.import_module(f"{name}.optim")
    optim_static = importlib.import_module(f"{name}.optim_static") if importlib.util.find_spec(f"{name}.optim_static") else None
    return {'mod': mod, 'nn': nn, 'optim': optim, 'optim_static': optim_static}


def get_params(model, key, init_x=None):
    """Initialize model parameters, trying with a sample input first."""
    if init_x is not None:
        try:
            return model.init(key, init_x)
        except TypeError:
            pass
    return model.init(key)


class TextMLPBase:
    def __init__(self, pjax, nn, vocab_size, embed_dim, hidden_dims, classes):
        self.nn = nn
        self.pjax = pjax
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.classes = classes

    def build(self):
        nn = self.nn
        pjax = self.pjax   
        class TextMLP(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dims, classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                last = embed_dim
                self.n_hidden = len(hidden_dims)
                for i, f in enumerate(hidden_dims):
                    setattr(self, f"linear_{i}", nn.Linear(last, f))
                    setattr(self, f"relu_{i}",   nn.ReLU(f))
                    last = f
                self.out = nn.Linear(last, classes)

            def __call__(self, x):
                embedded = self.embedding(x)           # (B, S, E)
                pooled = pjax.sum(embedded, axis=1)   # (B, E)
                h = pooled
                for i in range(self.n_hidden):
                    h = getattr(self, f"linear_{i}")(h)
                    h = getattr(self, f"relu_{i}")(h)
                return self.out(h)

        return TextMLP(self.vocab_size, self.embed_dim, self.hidden_dims, self.classes)


class TinyAttentionBase:
    def __init__(self, pjax, nn, vocab_size, embed_dim, classes):
        self.nn = nn
        self.pjax = pjax
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.classes = classes

    def build(self):
        nn = self.nn
        pjax = self.pjax   
        class TinyAttention(nn.Module):
            def __init__(self, vocab_size, embed_dim, classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.attention = nn.MultiHeadAttention(embed_dim, embed_dim, heads=1)
                self.out = nn.Linear(embed_dim, classes)

            def __call__(self, x):
                embedded = self.embedding(x)            # (B, S, E)
                context = self.attention(embedded)      # (B, S, E)
                pooled = pjax.sum(context, axis=1)     # (B, E)
                return self.out(pooled)

        return TinyAttention(self.vocab_size, self.embed_dim, self.classes)


def run(cfg, task_cfg, batch_size, run_number, key, impl_name, model_name):
    impl = load_impl(impl_name)
    pjax, nn, optim, optim_static = impl['mod'], impl['nn'], impl['optim'], impl['optim_static']

    optim_modules = {**vars(optim_static), **vars(optim)} if optim_static else {**vars(optim)}

    seed = cfg.get("random_seed", 0)

    if task_cfg["name"] != "SST2":
        raise NotImplementedError(f"Task {task_cfg['name']} not supported yet.")

    ds = SST2DataModule(
        batch_size=batch_size,
        max_seq_len=task_cfg.get("max_seq_len", 64),
        vocab_size=task_cfg.get("vocab_size", 10000),
        seed=seed,
    )
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model_key, _ = jax.random.split(key)
    vocab_size = ds.vocab.get_piece_size() if hasattr(ds.vocab, 'get_piece_size') else len(ds.vocab)

    if model_name == "mlp":
        model = TextMLPBase(pjax, nn, vocab_size, task_cfg["embed_dim"], task_cfg["hidden_dim"], task_cfg["classes"]).build()
    else:
        model = TinyAttentionBase(pjax, nn, vocab_size, task_cfg["embed_dim"], task_cfg["classes"]).build()

    init_x = jnp.array(next(iter(val_loader))[0])
    params = get_params(model, model_key, init_x)

    opt_name   = cfg.get(f"{impl_name}_optimizer") or cfg.get("pjax_optimizer") or cfg.get("pjax_orr_optimizer")
    opt_kwargs = cfg.get(f"{impl_name}_optimizer_kwargs") or cfg.get("pjax_optimizer_kwargs") or cfg.get("pjax_orr_optimizer_kwargs") or {}
    optimizer  = optim_modules[opt_name](**opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{impl_name}_{model_name}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    wandb_run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": f"{impl_name}_{model_name}",
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
        **(impl['config'].snapshot() if hasattr(impl.get('config', None), 'snapshot') else {}),
    })

    @jax.jit
    def step_fn(params, x, y):
        def loss_fn(p):
            pred = model.apply(p, x)
            y_oh = jax.nn.one_hot(y, pred.shape[-1])
            return pjax.cross_entropy(pred, y_oh)

        loss = loss_fn(params)
        res = optimizer.update(loss_fn, params)
        new_params = res[0] if isinstance(res, (tuple, list)) else res
        return new_params, jnp.mean(loss)

    @jax.jit
    def eval_fn(params, x, y):
        pred = model.apply(params, x)
        return jnp.mean(jnp.argmax(pred, axis=-1) == y)

    def evaluate(loader, p):
        accs = [float(eval_fn(p, jnp.array(x), jnp.array(y))) for x, y in loader]
        return float(jnp.mean(jnp.array(accs))) if accs else 0.0

    best_val_acc, best_params, best_step = 0.0, params, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                val_acc = evaluate(val_loader, params)
                wandb.log({"val/val_acc": val_acc, "training_time_s": time.time() - t0}, step=step)
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc, best_params, best_step = val_acc, params, step
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            x, y = next(train_iter)
            params, loss = step_fn(params, jnp.array(x), jnp.array(y))
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    final_acc = evaluate(test_loader, best_params)

    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc, "total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc, best_val_acc, best_step, total_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--impl', choices=['pjax', 'pjax_orr'], default=os.environ.get('PJAX_IMPL', 'pjax'))
    p.add_argument('--model', choices=['mlp', 'attention'], default='mlp')
    args = p.parse_args()

    cfg = yaml.safe_load(open(CFG_PATH))
    base_key = jax.random.key(cfg["random_seed"])
    all_keys = jax.random.split(base_key, len(cfg["batch_sizes"]) * len(cfg["tasks"]) * cfg["num_runs"])
    ki = 0

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{args.impl} ({args.model}) | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, all_keys[ki], args.impl, args.model)
                ki += 1
                gc.collect()
                jax.clear_caches()


if __name__ == '__main__':
    main()
