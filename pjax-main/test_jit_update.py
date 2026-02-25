import jax
import jax.numpy as jnp
from pjax.optim_static import AlternatingProjections
from pjax.core.computation import Parameter, Operation, Computation
from pjax.core.frozen_dict import FrozenDict
from functools import partial

# Create a dummy computation that mimics a forward pass
def dummy_fun(params):
    # params is a dict of Parameter nodes
    # We return an Operation that squares the parameter value
    
    class SquareOp(Operation):
        def __init__(self, parent):
            super().__init__(parent)
            self.value = parent.value ** 2
            
        def projection(self, x, output):
            # dummy projection: just sqrt
            return [jnp.sqrt(jnp.abs(output))]
            
    p = params['w']
    return SquareOp(p)

# We want to jit this update method
@partial(jax.jit, static_argnames=['opt', 'fun'])
def test_update(opt, fun, params):
    return opt.update(fun, params)

try:
    opt = AlternatingProjections()
    params = FrozenDict({'w': jnp.array([2.0, 3.0])})
    
    new_params, loss = test_update(opt, dummy_fun, params)
    print("New params:", new_params)
    print("Loss:", loss)
    print("JIT SUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("JIT FAILED!")
