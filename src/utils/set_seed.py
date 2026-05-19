import os
import random
import numpy as np

def seed_all(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for Python, NumPy, PyTorch, and igraph to ensure reproducibility.
    
    Args:
        seed: The random seed to use.
        deterministic: Whether to force deterministic algorithms in PyTorch (might reduce performance).
    """
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

        if deterministic:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
            # Ensure operations are deterministic
            torch.use_deterministic_algorithms(True, warn_only=False)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
    except Exception as e:
        print(f"[seed_all] PyTorch seeding skipped: {e}")
        
    # igraph
    try:
        import igraph as ig
        try:
            ig.set_random_number_generator(ig.random.RandomState(seed))
        except Exception:
            if hasattr(ig, "RandomState"):
                ig.set_random_number_generator(ig.RandomState(seed))
    except Exception as e:
        print(f"[seed_all] igraph seeding skipped: {e}")