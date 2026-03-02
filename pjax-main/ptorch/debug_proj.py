import torch
import time
from ptorch.core.ops import matmul_proj_parr_pt

def run():
    print("Testing bare matmul_proj_parr_pt ...", flush=True)
    M, K, N = 1024, 10, 8
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = torch.randn(M, N)
    
    t0 = time.time()
    A_proj, B_proj = matmul_proj_parr_pt(A, B, Z)
    t1 = time.time()
    
    print(f"Success! Time taken: {t1-t0:.4f}s", flush=True)
    print(f"Outputs: A_proj shape {A_proj.shape}, B_proj shape {B_proj.shape}", flush=True)

if __name__ == "__main__":
    run()
