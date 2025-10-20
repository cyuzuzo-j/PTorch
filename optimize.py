import jax.numpy as jnp
from jax import jit
class projectionOptimizer:
    def __init__(self, projectionsA, projectionsB):
        self.projectionsA = [jit(proj) for proj in projectionsA]
        self.projectionsB = [jit(proj) for proj in projectionsB]
        
    def step_layer(self, x, w, y):
        """Perform one optimization step."""
        raise NotImplementedError("This method should be implemented by subclasses.")
        
class AlternatingProjection(projectionOptimizer):
    def step_layer(self, x, w, y):
        """Perform one optimization step."""
        x_res = jnp.copy(x)
        w_res = jnp.copy(w)
        for projection in self.projectionsA:
            x_res, w_res, y = projection(x_res, w_res, y)
        
        for projection in self.projectionsB:
            x_res, w_res, y = projection(x_res, w_res, y)
        
        return x_res, w_res, y
    
class DouglassRachford(projectionOptimizer):
    def step_layer(self, x, w, y):
        """Perform one optimization step."""
        x_res = jnp.copy(x)
        w_res = jnp.copy(w)
        for projection in self.projectionsA:
            proj_x, proj_w_, proj_y = projection(x_res, w_res, y)
            x_res = 2 * proj_x - x_res
            w_res = 2 * proj_w_ - w_res
            y = 2*proj_y - y
            
        
        for projection in self.projectionsB:
            proj_x, proj_w, proj_y = projection(x_res, w_res, y)
            x_res = 2*proj_x - x_res
            w_res = 2*proj_w - w_res
            y = 2*proj_y - y
        
        return x_res, w_res, y