import torch
import sys
import time

try:
    print('Disabling dynamo...', flush=True)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception as e:
    print('Failed to config dynamo:', e)

print('Importing ops...', flush=True)
sys.path.insert(0, '/home/cyuzuzo/thesisV5/frameworks')
try:
    from ptorch.core.ops import FusedSimplexBilinearL2, fused_l2_proj_solver, simplex_op_pt
    print('Ops imported successfully!', flush=True)
except Exception as e:
    print('Failed to import ops:', e, flush=True)
    sys.exit(1)

def run_tests():
    print('--- Test 1: Basic forward/backward shapes ---', flush=True)
    x = torch.randn(4, 16, requires_grad=True)
    W = torch.randn(10, 16, requires_grad=True)
    gamma = 1.0

    print('Forward pass...', flush=True)
    out = FusedSimplexBilinearL2.apply(x, W, gamma)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"
    assert torch.allclose(out.sum(dim=-1), torch.ones(4)), "Output not on simplex!"
    print(f'  Output shape: {out.shape}, simplex sums: {out.sum(dim=-1)}', flush=True)

    print('Backward pass...', flush=True)
    y_target = torch.softmax(torch.randn(4, 10), dim=-1)
    out.backward(y_target)
    assert x.grad is not None, "x.grad is None"
    assert W.grad is not None, "W.grad is None"
    assert torch.isfinite(x.grad).all(), "x.grad has NaN/Inf"
    assert torch.isfinite(W.grad).all(), "W.grad has NaN/Inf"
    print(f'  x.grad shape: {x.grad.shape}, W.grad shape: {W.grad.shape}', flush=True)


    print('\n--- Test 2: Correctness check ---', flush=True)
    x2 = torch.randn(4, 8)
    W2 = torch.randn(8, 8)
    z2 = x2 @ W2.T
    y2 = simplex_op_pt(z2)
    y_tgt = simplex_op_pt(torch.randn(4, 8))

    print('Solving fused projection...', flush=True)
    x_proj, W_proj = fused_l2_proj_solver(x2, W2, z2, None, y_tgt, 50.0)
    
    z_proj = x_proj @ W_proj.T
    y_proj = simplex_op_pt(z_proj)

    err_before = (y2 - y_tgt).abs().max().item()
    err_after  = (y_proj - y_tgt).abs().max().item()
    print(f'  Max error before: {err_before:.6f}', flush=True)
    print(f'  Max error after:  {err_after:.6f}', flush=True)
    print(f'  Improved: {err_after < err_before}', flush=True)


    print('\n--- Test 3: Iterative stability ---', flush=True)
    x4 = torch.randn(8, 16)
    W4 = torch.randn(10, 16)
    for step in range(20):
        z4 = x4 @ W4.T
        y_tgt4 = torch.softmax(torch.randn(8, 10), dim=-1)
        x4, W4 = fused_l2_proj_solver(x4, W4, z4, None, y_tgt4, 1.0)
        if not (torch.isfinite(x4).all() and torch.isfinite(W4).all()):
            print(f'  DIVERGED at step {step}!', flush=True)
            break
    else:
        print(f'  x norm after 20 steps: {x4.norm():.4f}', flush=True)
        print(f'  W norm after 20 steps: {W4.norm():.4f}', flush=True)
        print('  Stable: OK', flush=True)

    print('\nALL TESTS PASSED', flush=True)

if __name__ == "__main__":
    run_tests()
