import os
import random
import numpy as np
import torch


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cudnn flags only meaningful when cuda is present
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # request deterministic algos at the framework level when supported
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
