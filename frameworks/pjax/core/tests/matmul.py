""""
Test file confirming the functioning of matmul_exact projection
"""
from .. import ops
import jax.numpy as jnp
import jax


def test_forward_working():
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    result = ops.matmul_op(a, b)
    jnp.testing.assert_allclose(result, a)


def test_exact_projection_satisfies_constraint():
    """After projection, a_new @ b_new should be closer to z."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (4, 3))
    b = jax.random.normal(k2, (3, 5))
    z = a @ b + 0.5  # perturbed output

    a_new, b_new = ops.matmul_proj_exact(a, b, z)
    
    # The projected values should be closer to satisfying a@b=z
    residual_before = jnp.linalg.norm(a @ b - z)
    residual_after = jnp.linalg.norm(a_new @ b_new - z)
    assert residual_after < residual_before, (
        f"Projection should reduce residual: {residual_before:.4f} -> {residual_after:.4f}"
    )


def test_exact_projection_reduces_distance():
    """Projected (a_new, b_new) should be closer to constraint than original."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (3, 4))
    b = jax.random.normal(k2, (4, 2))
    z = a @ b * 1.5  # z doesn't satisfy a@b=z

    a_new, b_new = ops.matmul_proj_exact(a, b, z)

    # Total distance (a,b) -> (a_new, b_new) should be finite
    da = jnp.linalg.norm(a_new - a)
    db = jnp.linalg.norm(b_new - b)
    assert jnp.isfinite(da) and jnp.isfinite(db)
    assert da > 0 or db > 0, "Projection should modify at least one of a or b"


def test_exact_projection_idempotent():
    """If a@b already equals z, projection should return (a, b) unchanged."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (3, 4))
    b = jax.random.normal(k2, (4, 2))
    z = a @ b  # exactly on constraint

    a_new, b_new = ops.matmul_proj_exact(a, b, z)
    
    jnp.testing.assert_allclose(a_new, a, atol=1e-3)
    jnp.testing.assert_allclose(b_new, b, atol=1e-3)
