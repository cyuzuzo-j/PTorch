##################################################
###   Benchmark — ptorch (cyclic projections)  ###
###   CNN architecture with FX tracing         ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn.functional as F
import torch.fx
from ptorch.core.ops import HardMarginProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
import tqdm, time
import wandb
import torch.nn as tnn
import ptorch.nn.experimental_modules as pem
from ptorch.nn.modules import ProjectionModule, HardMarginLoss, CrossEntropy

from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

ptorch_config.use_projections = True

FRAMEWORK = "ptorch_cnn"
OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# ── FX Tracer / Interpreter ──────────────────────────────────────────────────

# ── Model ────────────────────────────────────────────────────────────────────

class LeNet5_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, alpha=1.0, g=1.0):
        super().__init__()
        # LeNet-5 architecture components
        self.conv1 = pnn.Conv2D(in_channels, 6, kernel_size=5, alpha=alpha, g=g)
        self.pool1 = pem.MaxPool2d(2)
        self.relu1 = pnn.LeakyReLU(0.1)

        self.conv2 = pnn.Conv2D(6, 16, kernel_size=5, alpha=alpha, g=g)
        self.pool2 = pem.MaxPool2d(2)
        self.relu2 = pnn.LeakyReLU(0.1)

        # Predict flattening size via dummy pass
        with torch.no_grad():
            dummy_res = 28 if in_channels == 1 else 32
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            # Trace through feature extractor
            x = self.relu1(self.pool1(self.conv1(dummy_in)))
            x = self.relu2(self.pool2(self.conv2(x)))
            flatten_size = x.reshape(1, -1).size(1)

        self.fc1   = pnn.Linear(flatten_size, 120, norm="2")
        self.relu3 = pnn.LeakyReLU(0.1)
        
        self.fc2   = pnn.Linear(120, 84, norm="2")
        self.relu4 = pnn.LeakyReLU(0.1)

        self.head = pnn.Linear(84, classes, norm="2")
        self.loss = HardMarginLoss()

    def forward(self, x, y_oh):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        logits = self.head(x) 
        return self.loss(logits, y_oh)

    def forward_eval(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        logits = self.head(x) 
        return logits



# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    dataset_cls = DATASETS[dataset_name]
    
    if dataset_name == "CIFAR10":
        ds = dataset_cls(batch_size=batch_size, data_dir="./dataset", seed=run_seed)
    else:
        ds = dataset_cls(batch_size=batch_size, seed=run_seed)
        
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = LeNet5_PTorch(classes=task_cfg["classes"], in_channels=task_cfg.get("in_channels", 3)).to(device)
    

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    run_wandb = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "K": cfg.get("K", 1),
        "seed": run_seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
        **ptorch_config.snapshot(),
    })

    def eval_acc(loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                if xv.shape[-1] in [1, 3]:  
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                logits = model.forward_eval(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()
    
    K = cfg.get("K", 1)
    step = 0

    with tqdm.tqdm(total=cfg["max_steps"], unit="step") as pbar:
        while step < cfg["max_steps"]:
            # ── Fetch a new batch ─────────────────────────────────────────────
            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x_batch.shape[-1] in [1, 3]:
                x_batch = x_batch.permute(0, 3, 1, 2)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            model.train()
            # Initial real forward pass for this batch
            output = model(x_batch, y_oh)

            # ── Backward + weight update ───────────────────────────────────
            optimizer.zero_grad()
            output.sum().backward()
            optimizer.step()
    
            # ── Eval ──────────────────────────────────────────────────────────
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                wandb.log({"val/val_acc": val_acc, "training_time_s": time.time() - t0}, step=step)
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
                
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss   = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            wandb.log({"train/loss": ce_loss, "train/train_acc": train_acc}, step=step)

            pbar.update(1)
            step += 1
            
            if no_improve >= cfg["patience"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"Test Acc: {test_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": test_acc, "total_training_time_s": total_time}, step=step)
    wandb.finish()


if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()