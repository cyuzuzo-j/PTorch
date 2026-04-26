import torch

M, K, N = 4, 2, 8
A = torch.randn(M, K)
B = torch.randn(K, N)
Z_target = torch.randn(M, N)
g = 1.0
lam = 1.0

# Forward
P = A @ B
Z0 = Z_target

# Our math:
S = A @ A.transpose(-1, -2)
I = torch.eye(M)
system_matrix = I + lam * S
rhs = P + lam * (S @ Z0)
Z_proj = torch.linalg.solve(system_matrix, rhs)
T = lam * (Z_proj - Z0)
B_proj = B - A.transpose(-1, -2) @ T

# Let's test if A @ B_proj == Z_proj
print("Error in constraint:", torch.norm(A @ B_proj - Z_proj).item())

# Let's optimize it directly with gradient descent to check if it's the minimum
B_opt = B.clone().requires_grad_(True)
Z_opt = (A @ B_opt)
opt = torch.optim.Adam([B_opt], lr=0.01)
for _ in range(5000):
    opt.zero_grad()
    Z_curr = A @ B_opt
    loss = 0.5 * torch.sum((B_opt - B)**2) + 0.5 * lam * torch.sum((Z_curr - Z_target)**2)
    loss.backward()
    opt.step()

print("Diff between analytical B and optimal B:", torch.norm(B_proj - B_opt).item())
print("Analytical loss:", 0.5 * torch.sum((B_proj - B)**2) + 0.5 * lam * torch.sum((Z_proj - Z_target)**2))
print("Optimal loss:", 0.5 * torch.sum((B_opt - B)**2) + 0.5 * lam * torch.sum(((A@B_opt) - Z_target)**2))

