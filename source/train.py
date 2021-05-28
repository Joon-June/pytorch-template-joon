import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config