import re

with open('experiments/mlp/bench_memory.py', 'r') as f:
    text = f.read()

# Remove tracemalloc logic and rely on maxrss in _run_in_subprocess
replacement = """
import resource

def _traced_peak_mb(fn):
    # Run gn and return maxrss difference or something?
    # Actually just run fn
    fn()
    return 0.0 # Will be overridden in subprocess
"""

text = re.sub(r'def _traced_peak_mb\(fn\):.*?return peak_bytes / \(1024 \* 1024\)', replacement, text, flags=re.DOTALL)

subprocess_replacement = """
def _run_in_subprocess(func_name, M, D, N, K, return_dict):
    import resource
    # Warmup and then measure
    train_iter = _dummy_iter(N, K)
    
    # Run the function
    MEASURE_FNS[func_name](M, D, train_iter, batch_size=N, in_features=K)
    
    # Measure maxrss at end
    maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # On Linux ru_maxrss is in kilobytes
    import sys
    if sys.platform == 'darwin':
        # On macOS, ru_maxrss is in bytes
        peak_mb = maxrss_kb / (1024 * 1024)
    else:
        peak_mb = maxrss_kb / 1024
        
    return_dict[func_name] = peak_mb
"""

text = re.sub(r'def _run_in_subprocess.*?in_features=K\n    \)', subprocess_replacement, text, flags=re.DOTALL)

with open('experiments/mlp/bench_memory.py', 'w') as f:
    f.write(text)
