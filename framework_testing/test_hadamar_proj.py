import jax
import jax.numpy as jnp
from jax import jit, jacrev
from jax.scipy.linalg import solve

# ----------------------------------------------------------
# System: F(z) = 0
# z = [w_r', w_i', x_r', x_i', y_r', y_i', lambda_r, lambda_i]
# ----------------------------------------------------------

def F(z, const):
    w_r, w_i, x_r, x_i, y_r, y_i = const
    w_rp, w_ip, x_rp, x_ip, y_rp, y_ip, lam_r, lam_i = z

    return jnp.array([
        2*(w_rp - w_r) + lam_i * x_ip + lam_r * x_rp,
        2*(w_ip - w_i) - lam_r * x_ip + lam_i * x_rp,
        2*(x_rp - x_r) + lam_i * w_ip + lam_r * w_rp,
        2*(x_ip - x_i) - lam_r * w_ip + lam_i * w_rp,
        2*(y_rp - y_r) - lam_r,
        2*(y_ip - y_i) - lam_i,
        w_rp * x_rp - w_ip * x_ip - y_rp,
        w_rp * x_ip + w_ip * x_rp - y_ip
    ])

JF = jacrev(F)


@jit
def newton_step(z, const):
    J = JF(z, const)
    f = F(z, const)
    delta = solve(J, -f)
    return z + delta, jnp.linalg.norm(delta)


def newton_solve(z0, const, tol=1e-12, max_iter=50):
    z = z0
    for k in range(max_iter):
        z_new, step_norm = newton_step(z, const)
        if step_norm < tol:
            return z_new, True, k+1
        z = z_new
    return z, False, max_iter


# ----------------------------------------------------------
# Verification: compute y = w * x
# ----------------------------------------------------------

def compute_y_from_wx(z):
    w_rp, w_ip, x_rp, x_ip, _, _, _, _ = z

    w = w_rp + 1j * w_ip
    x = x_rp + 1j * x_ip
    y = w * x

    return jnp.array([y.real, y.imag])


# ----------------------------------------------------------
# Example usage
# ----------------------------------------------------------

if __name__ == "__main__":
    # Known constants (unprimed)
    const = jnp.array([1.0, 2.0, -1.0, 0.5, 0.2, -0.4])

    # Initial guess for z
    z0 = jnp.zeros(8)

    # Solve system
    z_sol, converged, iters = newton_solve(z0, const)

    print("Converged:", converged)
    print("Iterations:", iters)
    print("Solution z =")
    print(z_sol)

    # Compute y = w x from the solution
    y_wx = compute_y_from_wx(z_sol)

    print("\nVerification of y = w x:")
    print("Computed y (from w*x):       ", y_wx)
    print("Solved y':                   ", z_sol[4:6])
    print("Difference:                  ", y_wx - z_sol[4:6])
