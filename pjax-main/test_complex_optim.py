
import jax
import jax.numpy as jnp
from pjax import optim
from pjax.core.computation import Array, Parameter, Operation

def test_complex():
    # Define a simple problem: Find x st x in A and x in B
    # A: x = 1 + 1j
    # B: x = 2 + 2j
    # This is infeasible, but we can test if steps work.
    
    # Mock projections
    def proj_a_fn(x, *args):
        return jnp.ones_like(x) * (1.0 + 1.0j)
    
    def proj_b_fn(x, *args):
        return jnp.ones_like(x) * (2.0 + 2.0j)

    # We can't easily mock the graph structure required by the optimizer without full pjax setup.
    # So we will rely on the _step method directly if possible, or construct a minimal graph.
    
    # Let's verify _step method logic directly using dummy inputs.
    
    # Dummy inputs
    vars_ = {"x": jnp.array([0.0 + 0.0j], dtype=jnp.complex64)}
    velocity = {"x": jnp.array([0.1 + 0.1j], dtype=jnp.complex64)}
    
    # Mock projections as callables
    def projection_a(v):
        return jax.tree.map(lambda x: jnp.ones_like(x) * (1.0 + 1.0j), v)
        
    def projection_b(v):
        return jax.tree.map(lambda x: jnp.ones_like(x) * (2.0 + 2.0j), v)
    
    print("Testing AlternatingProjectionsMomentum...")
    ap_mon = optim.AlternatingProjectionsMonumentum(steps_per_update=1, change_projection_order=False)
    # Patch beta since it's hardcoded to 0 in __init__ but we assume user might have changed it or implementation
    # effectively uses 0.9 in _step in the file we saw.
    
    try:
        new_vars_ap, new_vel_ap = ap_mon._step(vars_, projection_a, projection_b, velocity)
        print("AP Mon Step Result:", new_vars_ap, new_vel_ap)
    except Exception as e:
        print("AP Mon Failed:", e)
        
    print("\nTesting DouglasRachfordMomentum...")
    dr_mon = optim.DouglasRachfordMomentum(steps_per_update=1, change_projection_order=False, beta=0.9)
    try:
        new_vars_dr, new_vel_dr = dr_mon._step(vars_, projection_a, projection_b, velocity)
        print("DR Mon Step Result:", new_vars_dr, new_vel_dr)
    except Exception as e:
        print("DR Mon Failed:", e)

if __name__ == "__main__":
    test_complex()
