import json
from dataclasses import dataclass


@dataclass
class Config:
    project_name: str
    run_name: str

    train_batch_size: int
    test_batch_size: int
    lr: float

    num_iters: int
    num_workers: int
    seed: int

    eval_every: int
    visualize_every: int
    save_every: int



def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)