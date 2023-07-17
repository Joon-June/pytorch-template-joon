from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Config


class DatasetOnSomething(Dataset):
    def __init__(self, params):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def get_torch_transform():
        return transforms.Compose([transforms.ToTensor()])


def get_train_dataloader(config: Config):
    data = DatasetOnSomething(config)

    data_loader_params = {
        "batch_size": config.train_batch_size,
        "shuffle": True,
        "num_workers": config.num_workers
    }

    loader = DataLoader(data, **data_loader_params)
    return loader


def get_eval_dataloader(config: Config):
    data = DatasetOnSomething(config)

    data_loader_params = {
        "batch_size": config.test_batch_size,
        "shuffle": False,
        "num_workers": 0
    }

    loader = DataLoader(data, **data_loader_params)
    return loader


class DataFetcher:
    def __init__(self, loader, device=None):
        self.loader = loader
        self.device = device

        self.iter = iter(self.loader)

    def _fetch_inputs(self):
        try:
            batch = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            batch = next(self.iter)
        return batch

    def __next__(self):
        batch = self._fetch_inputs()
        return batch
