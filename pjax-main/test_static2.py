import jax
import jax.numpy as jnp
from pjax.core.computation import Parameter, Operation, Computation
from pjax.core.frozen_dict import FrozenDict
from functools import partial
from pjax.optim_static import AlternatingProjections

def dummy_fun(params):
    class SquareOp(Operation):
        def __init__(self, parent):
            super().__init__('square', lambda x: x**2, lambda x, out: [jnp.sqrt(jnp.abs(out))], [parent], parent.value ** 2)
    p = params['w']
    return SquareOp(p)

opt = AlternatingProjections(steps_per_update=3)
params = FrozenDict({'w': jnp.array([2.0, 3.0])})

try:
    final, losses = opt.update(dummy_fun, params)
    print("SUCCESS")
    print(final)
except Exception as e:
    import traceback
    traceback.print_exc()
