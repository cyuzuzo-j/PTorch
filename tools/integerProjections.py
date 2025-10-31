import jax.numpy as np
from ortools.sat.python import cp_model

class ClosestBilinearSolver:
    def __init__(self, m, n, W_bounds=(-50, 50), x_bounds=(-50, 50), y_bounds=(-50, 50)):
        self.m, self.n = m, n
        self.model = cp_model.CpModel()

        # --- Decision variables ---
        self.W = [[self.model.NewIntVar(W_bounds[0], W_bounds[1], f"W[{i},{j}]") for j in range(n)] for i in range(m)]
        self.x = [self.model.NewIntVar(x_bounds[0], x_bounds[1], f"x[{j}]") for j in range(n)]
        self.y = [self.model.NewIntVar(y_bounds[0], y_bounds[1], f"y[{i}]") for i in range(m)]

        # --- Bilinear intermediates ---
        min_product = min(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        max_product = max(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        self.z = [[self.model.NewIntVar(min_product, max_product, f"z[{i},{j}]") for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                self.model.AddMultiplicationEquality(self.z[i][j], [self.W[i][j], self.x[j]])
            # Add bilinear constraint y = Wx
            self.model.Add(self.y[i] == sum(self.z[i][j] for j in range(n)))

        # --- Difference vars (placeholders) ---
        # These will be re-linked each run
        self.abs_diff_W = [[self.model.NewIntVar(0, 1000, f"dW[{i},{j}]") for j in range(n)] for i in range(m)]
        self.abs_diff_x = [self.model.NewIntVar(0, 1000, f"dx[{j}]") for j in range(n)]
        self.abs_diff_y = [self.model.NewIntVar(0, 1000, f"dy[{i}]") for i in range(m)]

        # We'll add constraints linking to constants later
        self.diff_constraints = []

        # --- Objective placeholder ---
        self.total_diff = self.model.NewIntVar(0, 1_000_000, "total_diff")
        self.model.Minimize(self.total_diff)

        # Prepare solver once
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 1.0
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.cp_model_presolve = False  # faster reuse

    def solve(self, x0, W0, y0):
        # Remove old constraints (if any)
        for c in self.diff_constraints:
            self.model.Proto().constraints.remove(c)
        self.diff_constraints.clear()

        # --- Rebuild difference constraints dynamically ---
        for i in range(self.m):
            for j in range(self.n):
                c = self.model.AddAbsEquality(self.abs_diff_W[i][j], self.W[i][j] - int(W0[i, j]))
                self.diff_constraints.append(c.Proto())

        for j in range(self.n):
            c = self.model.AddAbsEquality(self.abs_diff_x[j], self.x[j] - int(x0[j]))
            self.diff_constraints.append(c.Proto())

        for i in range(self.m):
            c = self.model.AddAbsEquality(self.abs_diff_y[i], self.y[i] - int(y0[i]))
            self.diff_constraints.append(c.Proto())

        # --- Update objective ---
        total_terms = [self.abs_diff_W[i][j] for i in range(self.m) for j in range(self.n)] + \
                      self.abs_diff_x + self.abs_diff_y
        self.model.Add(self.total_diff == sum(total_terms))

        # --- Solve ---
        status = self.solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            x_res = np.array([self.solver.Value(self.x[j]) for j in range(self.n)])
            W_res = np.array([[self.solver.Value(self.W[i][j]) for j in range(self.n)] for i in range(self.m)])
            y_res = np.array([self.solver.Value(self.y[i]) for i in range(self.m)])
            return x_res, W_res, y_res
        else:
            print("No solution found. Status =", status)
            return x0, W0, y0
        
        
import jax.numpy as np
from ortools.sat.python import cp_model

class ClosestBilinearSolverY:
    def __init__(self, m, n, W_bounds=(-50, 50), x_bounds=(-50, 50), y_bounds=(-50, 50)):
        self.m, self.n = m, n
        self.model = cp_model.CpModel()

        # --- Decision variables ---
        self.W = [[self.model.NewIntVar(W_bounds[0], W_bounds[1], f"W[{i},{j}]") for j in range(n)] for i in range(m)]
        self.x = [self.model.NewIntVar(x_bounds[0], x_bounds[1], f"x[{j}]") for j in range(n)]
        self.y = [self.model.NewIntVar(y_bounds[0], y_bounds[1], f"y[{i}]") for i in range(m)]

        # --- Bilinear intermediates ---
        min_product = min(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        max_product = max(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        self.z = [[self.model.NewIntVar(min_product, max_product, f"z[{i},{j}]") for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                self.model.AddMultiplicationEquality(self.z[i][j], [self.W[i][j], self.x[j]])
            self.model.Add(self.y[i] == sum(self.z[i][j] for j in range(n)))

        # --- Difference vars (placeholders) ---
        # These will be re-linked each run
        self.abs_diff_W = [[self.model.NewIntVar(0, 1000, f"dW[{i},{j}]") for j in range(n)] for i in range(m)]
        self.abs_diff_x = [self.model.NewIntVar(0, 1000, f"dx[{j}]") for j in range(n)]
        self.abs_diff_y = [self.model.NewIntVar(0, 1000, f"dy[{i}]") for i in range(m)]

        # We'll add constraints linking to constants later
        self.diff_constraints = []

        # --- Objective placeholder ---
        self.total_diff = self.model.NewIntVar(0, 1_000_000, "total_diff")
        self.model.Minimize(self.total_diff)

        # Prepare solver once
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 1.0
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.cp_model_presolve = False  # faster reuse

    def solve(self, x0, W0, y0):
        # Remove old difference constraints (if any)
        for c in self.diff_constraints:
            self.model.Proto().constraints.remove(c)
        self.diff_constraints.clear()

        # --- Rebuild difference constraints dynamically ---
        for i in range(self.m):
            for j in range(self.n):
                c = self.model.AddAbsEquality(self.abs_diff_W[i][j], self.W[i][j] - int(W0[i, j]))
                self.diff_constraints.append(c.Proto())

        for j in range(self.n):
            c = self.model.AddAbsEquality(self.abs_diff_x[j], self.x[j] - int(x0[j]))
            self.diff_constraints.append(c.Proto())

        for i in range(self.m):
            c = self.model.AddAbsEquality(self.abs_diff_y[i], self.y[i] - int(y0[i]))
            self.diff_constraints.append(c.Proto())

        # --- Update objective ---
        total_terms = [self.abs_diff_W[i][j] for i in range(self.m) for j in range(self.n)] + \
                      self.abs_diff_x + self.abs_diff_y
        self.model.Add(self.total_diff == sum(total_terms))

        # --- Solve ---
        status = self.solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            x_res = np.array([self.solver.Value(self.x[j]) for j in range(self.n)])
            y_res = np.array([self.solver.Value(self.y[i]) for i in range(self.m)])
            W_res = np.array([[self.solver.Value(self.W[i][j]) for j in range(self.n)] for i in range(self.m)])
            return x_res, W_res, y_res
        else:
            print("No solution found. Status =", status)
            return x0, W0, y0

class ClosestBilinearSolverFixedX:
    def __init__(self, m, n, W_bounds=(-50, 50), y_bounds=(-50, 50)):
        self.m, self.n = m, n
        self.W_bounds = W_bounds
        self.y_bounds = y_bounds

    def solve(self, x_fixed, W0, y0):
        # Build a fresh, small linear model each call (no bilinear terms).
        model = cp_model.CpModel()

        # Round x to nearest int and treat it as constant.
        x_fix_int = [int(np.rint(x_fixed[j])) for j in range(self.n)]

        # Decision vars: W, y
        W = [[model.NewIntVar(self.W_bounds[0], self.W_bounds[1], f"W[{i},{j}]")
              for j in range(self.n)] for i in range(self.m)]
        y = [model.NewIntVar(self.y_bounds[0], self.y_bounds[1], f"y[{i}]")
             for i in range(self.m)]

        # Linear constraints: y = W x (x is fixed integer)
        for i in range(self.m):
            model.Add(y[i] == sum(W[i][j] * x_fix_int[j] for j in range(self.n)))

        # L1 proximity to (W0, y0)
        abs_diff_W = [[model.NewIntVar(0, 1000, f"dW[{i},{j}]") for j in range(self.n)] for i in range(self.m)]
        abs_diff_y = [model.NewIntVar(0, 1000, f"dy[{i}]") for i in range(self.m)]

        for i in range(self.m):
            for j in range(self.n):
                model.AddAbsEquality(abs_diff_W[i][j], W[i][j] - int(np.rint(W0[i, j])))
        for i in range(self.m):
            model.AddAbsEquality(abs_diff_y[i], y[i] - int(np.rint(y0[i])))

        # Objective: minimize total L1 deviation
        model.Minimize(
            sum(abs_diff_W[i][j] for i in range(self.m) for j in range(self.n)) +
            sum(abs_diff_y)
        )

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.cp_model_presolve = True
        solver.parameters.max_time_in_seconds = 2.0

        status = solver.Solve(model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            y_res = np.array([solver.Value(y[i]) for i in range(self.m)])
            W_res = np.array([[solver.Value(W[i][j]) for j in range(self.n)] for i in range(self.m)])
            return np.array(x_fix_int), W_res, y_res
        else:
            return np.array(x_fix_int), W0, y0