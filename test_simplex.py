import math
import random

def solve_exact(a, z):
    n = len(a)
    best_dist = float('inf')
    best_S = None
    
    for i in range(1, 1 << n):
        S = [j for j in range(n) if (i & (1 << j))]
        k = len(S)
        w_sum = sum(a[j] + z[j] for j in S)
        tau = (w_sum - 2.0) / k
        
        valid = True
        for j in S:
            if a[j] + z[j] < tau:
                valid = False
        if not valid: continue
        
        dist = 0
        for j in range(n):
            if j in S:
                a_new = (a[j] + z[j] + tau) / 2
                z_new = (a[j] + z[j] - tau) / 2
            else:
                a_new = min(a[j], tau)
                z_new = 0
            dist += (a_new - a[j])**2 + (z_new - z[j])**2
            
        if dist < best_dist:
            best_dist = dist
            best_S = S
            
    return best_dist, best_S

def solve_sort_prefix(a, z, c):
    n = len(a)
    v = [a[i] + c * z[i] for i in range(n)]
    idx = sorted(range(n), key=lambda i: v[i], reverse=True)
    
    best_dist = float('inf')
    best_S = None
    
    for k in range(1, n + 1):
        S = idx[:k]
        w_sum = sum(a[j] + z[j] for j in S)
        tau = (w_sum - 2.0) / k
        
        valid = True
        for j in S:
            if a[j] + z[j] < tau:
                valid = False
        if not valid: continue
        
        dist = 0
        for j in range(n):
            if j in S:
                a_new = (a[j] + z[j] + tau) / 2
                z_new = (a[j] + z[j] - tau) / 2
            else:
                a_new = min(a[j], tau)
                z_new = 0
            dist += (a_new - a[j])**2 + (z_new - z[j])**2
            
        if dist < best_dist:
            best_dist = dist
            best_S = S
            
    return best_dist, best_S

mismatches = 0
for _ in range(100000):
    a = [random.uniform(-10, 10) for _ in range(5)]
    z = [random.uniform(0, 10) for _ in range(5)]  # POSITIVE Z ONLY
    d_exact, _ = solve_exact(a, z)
    d_pref, _ = solve_sort_prefix(a, z, 1.0)
    if abs(d_exact - d_pref) > 1e-5:
        mismatches += 1
print(f"w = a+z -> {mismatches} mismatches for z >= 0")

mismatches = 0
c = math.sqrt(2) - 1
for _ in range(100000):
    a = [random.uniform(-10, 10) for _ in range(5)]
    z = [random.uniform(0, 10) for _ in range(5)]  # POSITIVE Z ONLY
    d_exact, _ = solve_exact(a, z)
    d_pref, _ = solve_sort_prefix(a, z, c)
    if abs(d_exact - d_pref) > 1e-5:
        mismatches += 1
print(f"v = a+cz -> {mismatches} mismatches for z >= 0")
