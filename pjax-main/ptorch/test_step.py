import torch
import torch.nn as nn
from ptorch.nn.modules import Step_NB

torch.set_default_dtype(torch.float32)

def main():
    print("Testing Step_NB...", flush=True)
    step = Step_NB()
    
    x = torch.tensor([-1.5, -0.1, 0.0, 0.1, 2.0], requires_grad=True)
    out = step(x)
    
    print(f"Input:  {x.data}")
    print(f"Output: {out.data}")
    
    target = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    
    import ptorch.core.ops as ops
    out.backward(target)
    
    print(f"Projected x (stored in grad): {x.grad}")

if __name__ == "__main__":
    main()
