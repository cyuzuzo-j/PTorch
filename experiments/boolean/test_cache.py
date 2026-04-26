import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.fx
from ptorch.nn.modules import Linear, ReLU, ProjectionModule, CrossEntropy
from ptorch import config

config.use_projections = True

class ProjectionTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, ProjectionModule):
            return True
        return super().is_leaf_module(m, module_qualified_name)

class PropagateCacheTest(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node):
        cache_to_apply = None
        if n.op == 'call_module':
            submod = self.module.get_submodule(n.target)
            if hasattr(submod, 'projection_forward_cache') and isinstance(submod.projection_forward_cache, list):
                if submod.projection_forward_cache:
                    cache = submod.projection_forward_cache[0]
                    if cache is not None:
                        cache_to_apply = cache
                    else:
                        print(f"\n[PropagateCache] Intercepted node '{n.name}' but cache is empty.")
                    
        result = super().run_node(n)
        
        if cache_to_apply is not None:
            result.data.copy_(cache_to_apply.data)
            
        return result

class MiniMLP(tnn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 4)
        self.relu = ReLU()
        self.layer2 = Linear(4, 2)
        self.cross_entropy = CrossEntropy()


    def forward(self, x, y):
        x = self.layer1(x)
        x = self.relu(x)
        x =  self.layer2(x)
        return self.cross_entropy(x,y)


def xor_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def xor_ce_loss(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, y).item())

def main():
    torch.manual_seed(42)
    model = MiniMLP()
    
    tracer = ProjectionTracer()
    graph = tracer.trace(model)
    traced_model = torch.fx.GraphModule(model, graph)

    # XOR truth table
    x = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    y_oh = F.one_hot(y, num_classes=2).float()
    num_loops = 200

    print("=== Initial Forward Pass ===")
    graph = traced_model(x, y_oh)
    acc = xor_accuracy(graph, y)
    loss = xor_ce_loss(graph, y)
    print(f"Loop 000 | loss={loss:.6f} | acc={acc:.4f}")

    for loop_idx in range(1, num_loops + 1):
        traced_model.zero_grad(set_to_none=True)
        graph.sum().backward()

        interpreter = PropagateCacheTest(traced_model)
        graph = interpreter.run(x, y_oh)

        acc = xor_accuracy(graph, y)
        loss = xor_ce_loss(graph, y)
        print(f"Loop {loop_idx:03d} | loss={loss:.6f} | acc={acc:.4f}")

        if acc == 1.0:
            print(f"Reached perfect XOR accuracy at loop {loop_idx}.")
            break

if __name__ == "__main__":
    main()
