import jax.numpy as np
from ortools.sat.python import cp_model

class ClosestBilinearSolver:
    def __init__(self, m, n, W_bounds=(-50, 50), x_bounds=(-50, 50)):
        self.m, self.n = m, n
        self.model = cp_model.CpModel()

        # --- Decision variables ---
        self.W = [[self.model.NewIntVar(W_bounds[0], W_bounds[1], f"W[{i},{j}]") for j in range(n)] for i in range(m)]
        self.x = [self.model.NewIntVar(x_bounds[0], x_bounds[1], f"x[{j}]") for j in range(n)]

        # --- Bilinear intermediates ---
        min_product = min(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        max_product = max(W_bounds[0]*x_bounds[0], W_bounds[0]*x_bounds[1], 
                          W_bounds[1]*x_bounds[0], W_bounds[1]*x_bounds[1])
        self.z = [[self.model.NewIntVar(min_product, max_product, f"z[{i},{j}]") for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                self.model.AddMultiplicationEquality(self.z[i][j], [self.W[i][j], self.x[j]])

        # --- Difference vars (placeholders) ---
        # These will be re-linked each run
        self.abs_diff_W = [[self.model.NewIntVar(0, 1000, f"dW[{i},{j}]") for j in range(n)] for i in range(m)]
        self.abs_diff_x = [self.model.NewIntVar(0, 1000, f"dx[{j}]") for j in range(n)]

        # We'll add constraints linking to constants later
        self.diff_constraints = []
        self.y_constraints = []

        # --- Objective placeholder ---
        self.total_diff = self.model.NewIntVar(0, 1_000_000, "total_diff")
        self.model.Minimize(self.total_diff)

        # Prepare solver once
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 1.0
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.cp_model_presolve = False  # faster reuse

    def solve(self, x0, W0, y_fixed):
        # Remove old constraints (if any)
        for c in self.diff_constraints:
            self.model.Proto().constraints.remove(c)
        for c in self.y_constraints:
            self.model.Proto().constraints.remove(c)
        self.diff_constraints.clear()
        self.y_constraints.clear()

        # --- Add Y=Wx constraints (Y is now fixed) ---
        for i in range(self.m):
            c = self.model.Add(sum(self.z[i][j] for j in range(self.n)) == int(y_fixed[i]))
            self.y_constraints.append(c.Proto())

        # --- Rebuild difference constraints dynamically ---
        for i in range(self.m):
            for j in range(self.n):
                c = self.model.AddAbsEquality(self.abs_diff_W[i][j], self.W[i][j] - int(W0[i, j]))
                self.diff_constraints.append(c.Proto())

        for j in range(self.n):
            c = self.model.AddAbsEquality(self.abs_diff_x[j], self.x[j] - int(x0[j]))
            self.diff_constraints.append(c.Proto())

        # --- Update objective ---
        total_terms = [self.abs_diff_W[i][j] for i in range(self.m) for j in range(self.n)] + \
                      self.abs_diff_x
        self.model.Add(self.total_diff == sum(total_terms))

        # --- Solve ---
        status = self.solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            x_res = np.array([self.solver.Value(self.x[j]) for j in range(self.n)])
            W_res = np.array([[self.solver.Value(self.W[i][j]) for j in range(self.n)] for i in range(self.m)])
            return x_res, W_res, y_fixed
        else:
            print("No solution found. Status =", status)
            return x0, W0, y_fixed
        
        
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
        self.model = cp_model.CpModel()

        # --- Decision variables ---
        self.W = [[self.model.NewIntVar(W_bounds[0], W_bounds[1], f"W[{i},{j}]") for j in range(n)] for i in range(m)]
        self.y = [self.model.NewIntVar(y_bounds[0], y_bounds[1], f"y[{i}]") for i in range(m)]

        # --- Difference vars (placeholders) ---
        self.abs_diff_W = [[self.model.NewIntVar(0, 1000, f"dW[{i},{j}]") for j in range(n)] for i in range(m)]
        self.abs_diff_y = [self.model.NewIntVar(0, 1000, f"dy[{i}]") for i in range(m)]

        # We'll add constraints linking to constants later
        self.diff_constraints = []
        self.bilinear_constraints = []

        # --- Objective placeholder ---
        self.total_diff = self.model.NewIntVar(0, 1_000_000, "total_diff")
        self.model.Minimize(self.total_diff)

        # Prepare solver once
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 1.0
        self.solver.parameters.num_search_workers = 8
        self.solver.parameters.cp_model_presolve = False  # faster reuse

    def solve(self, x_fixed, W0, y0):
        # Remove old constraints (if any)
        for c in self.diff_constraints:
            self.model.Proto().constraints.remove(c)
        for c in self.bilinear_constraints:
            self.model.Proto().constraints.remove(c)
        self.diff_constraints.clear()
        self.bilinear_constraints.clear()

        # --- Add Y=Wx constraints (X is now fixed) ---
        for i in range(self.m):
            linear_sum = sum(self.W[i][j] * int(x_fixed[j]) for j in range(self.n))
            c = self.model.Add(self.y[i] == linear_sum)
            self.bilinear_constraints.append(c.Proto())

        # --- Rebuild difference constraints dynamically ---
        for i in range(self.m):
            for j in range(self.n):
                c = self.model.AddAbsEquality(self.abs_diff_W[i][j], self.W[i][j] - int(W0[i, j]))
                self.diff_constraints.append(c.Proto())

        for i in range(self.m):
            c = self.model.AddAbsEquality(self.abs_diff_y[i], self.y[i] - int(y0[i]))
            self.diff_constraints.append(c.Proto())

        # --- Update objective ---
        total_terms = [self.abs_diff_W[i][j] for i in range(self.m) for j in range(self.n)] + \
                      self.abs_diff_y
        self.model.Add(self.total_diff == sum(total_terms))

        # --- Solve ---
        status = self.solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            y_res = np.array([self.solver.Value(self.y[i]) for i in range(self.m)])
            W_res = np.array([[self.solver.Value(self.W[i][j]) for j in range(self.n)] for i in range(self.m)])
            return x_fixed, W_res, y_res
        else:
            print("No solution found. Status =", status)
            return x_fixed, W0, y0