import torch
import torch.nn as nn
from ptorch.nn.modules import Conv2D, ReLU

torch.set_default_dtype(torch.float32)

def main():
    print("START", flush=True)
    conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='same')
    
    x = torch.randn(4, 1, 16, 16)
    print("x created", flush=True)
    
    out = conv(x)
    print("conv forward done", flush=True)

if __name__ == "__main__":
    main()
