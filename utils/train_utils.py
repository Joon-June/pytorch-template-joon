import dataclasses
import os

import numpy as np
import torch

from config import Config


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    def __init__(self, patience=5, eps=1e-3):
        self.patience = patience
        self.eps = eps
        self.best_train_loss = np.Inf
        self.best_val_loss = np.Inf
        self.count = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss, model):
        if val_loss < self.best_val_loss - self.eps:
            self.best_val_loss = val_loss
            self.count = 0
            self.save_checkpoint(model)
        else: # Model is not improving
            self.count += 1
            print(f"Early stopping count {self.count}/{self.patience}")
            if self.count >= self.patience and self.best_train_loss - train_loss < self.eps:
                print(f"Early stopping activated.")
                self.early_stop = True

        if train_loss < self.best_train_loss - self.eps:
            self.best_train_loss = train_loss

        return self.early_stop

    @staticmethod
    def save_checkpoint(model):
        raise NotImplementedError


class Writer:
    def __init__(self, config: Config):
        self.config = config

    def initialize(self):
        from dotenv import load_dotenv
        load_dotenv()
        assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not given!"
        self.wandb_run = wandb.init(project=config.project_name, entity="joon-june", name=config.run_name)
        self.wandb_run.config.update(dataclasses.asdict(self.config))

    def log(self, log_dict, step):
        self.wandb_run.log(log_dict, step=step)


class ModelEval:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train()

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x)


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def batch_visualize(batch):
    raise NotImplementedError


def save_checkpoint(batch):
    raise NotImplementedError