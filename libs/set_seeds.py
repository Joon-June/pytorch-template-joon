import random

import numpy as np
import torch

from config.config import CONFIG


def set_seeds():
    print(f"Setting seeds with {CONFIG['SEED']}...")
    torch.manual_seed(CONFIG["SEED"])
    torch.cuda.manual_seed(CONFIG["SEED"])
    torch.cuda.manual_seed_all(CONFIG["SEED"])  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONFIG["SEED"])
    random.seed(CONFIG["SEED"])