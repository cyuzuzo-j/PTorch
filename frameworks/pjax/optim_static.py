from __future__ import annotations

from functools import partial

import jax
from jax import numpy as jnp

from .core.computation import Array, Computation, Operation, Parameter, ShapeTransform
from .core.frozen_dict import FrozenDict, freeze

# ────────────────────────────────────────────────────────
#  Optimizer
# ────────────────────────────────────────────────────────

def projection(computation: Operation | Parameter | Array, output: jnp.ndarray):
    """Compute the projection for a computation."""
    inputs = [parent.value for parent in computation.parents]

    return  computation.projection(*inputs, output)

def inverse(computation: Operation | Parameter | Array, output: jnp.ndarray):
    inputs = [parent.value for parent in computation.parents]
    return  computation.inverse(*inputs, output)
    

class CyclicProjections:
    """
        implementation of cyclic projections
    """

    def __init__(self, steps_per_update: int = 50, change_projection_order: bool = False):
        self.steps_per_update = steps_per_update
        self.change_projection_order = change_projection_order

    def update(self, fun: Computation, params: FrozenDict, output= None, steps_per_update=None):
        ### do a topological descent and project
        params = {name: Parameter(value, name=name) for name, value in params.items()}

        # do the actual forward pass
        root_computation = fun(params)
        # calculate the loss (diff root computation)
        nodes = [root_computation]
        outputs = [None]

        while nodes:
            node = nodes.pop(0)
            output = outputs.pop(0)

            # do a topological descent and project
            if isinstance(node, Parameter):
                params[node.name] = output

            elif isinstance(node, Operation | ShapeTransform):
                if isinstance(node, Operation):
                    new_vars = projection(node, output)
                elif isinstance(node, ShapeTransform):
                    new_vars = inverse(node, output)
            
                nodes.extend(node.parents)
                outputs.extend(new_vars)
        return params, -1

        

        
        


    