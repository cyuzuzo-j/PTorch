import os
import hashlib
import sys

def compute_dir_hash(directory):
    """Compute sha256 hash of all python and yaml files in a directory."""
    sha = hashlib.sha256()
    if not os.path.exists(directory):
        return sha.hexdigest()
    
    for root, dirs, files in sorted(os.walk(directory)):
        # Ignore things like __pycache__, wandb, .venv, .git and other dot folders
        blocked_dirs = {'__pycache__', 'wandb', '.venv', '.git', 'dataset', 'third_party', '.vscode-server'}
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in blocked_dirs]
        for f in sorted(files):
            if f.endswith('.py') or f.endswith('.yaml') or f.endswith('.yml') or f.endswith('.txt'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'rb') as file:
                        sha.update(file.read())
                except Exception:
                    pass
    return sha.hexdigest()

def get_code_hash():
    """Computes a combined hash of the current experiment's directory files and the ptorch framework."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    ptorch_dir = os.path.join(project_root, 'frameworks', 'ptorch')
    
    ptorch_hash = compute_dir_hash(ptorch_dir)
    
    if hasattr(sys.modules['__main__'], '__file__') and sys.modules['__main__'].__file__:
        exp_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    else:
        # Fallback to current working directory but don't walk too deep if it's the root
        exp_dir = os.getcwd()
        if os.path.basename(exp_dir) == 'learning_with_projections':
            # Don't hash the whole workspace
            print("Warning: Hashing root dir could be slow, only doing explicitly.")
        
    exp_hash = compute_dir_hash(exp_dir)
    
    combined_sha = hashlib.sha256()
    combined_sha.update(ptorch_hash.encode('utf-8'))
    combined_sha.update(exp_hash.encode('utf-8'))
    return combined_sha.hexdigest()
