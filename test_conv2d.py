import torch
import torch.nn.functional as F
import frameworks.ptorch as ptorch
import frameworks.ptorch.nn.modules as pnn

torch.set_default_dtype(torch.float32)
ptorch.config.use_projections = True

def main():
    print("START", flush=True)
    conv = pnn.Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    
    x = torch.randn(4, 1, 16, 16, requires_grad=True)
    print("x created", flush=True)
    
    out = conv(x)
    print("conv forward done", flush=True)
    print(out.shape)
    
    loss = out.sum()
    loss.backward()
    
    print("conv backward done", flush=True)

if __name__ == "__main__":
    main()
