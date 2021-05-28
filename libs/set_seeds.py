import random

import numpy as np
import torch

from config.config import CONFIG


def set_seeds():
    print(f"Setting seeds with {CONFIG['seed']}...")
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])