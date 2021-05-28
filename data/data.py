import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CONFIG

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


def get_dataloader(params):
    data = DatasetOnSomething(params)

    data_loader_params = {"batch_size": CONFIG["batch_size"],
                          "shuffle": False,
                          "num_workers": 1,
                          "pin_memory": False}

    loader = torch.utils.data.DataLoader(data, **data_loader_params)
    return loader