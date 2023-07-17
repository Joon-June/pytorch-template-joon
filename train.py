import argparse

import torch

from libs.config import load_config
from libs.data import DataFetcher, get_eval_dataloader, get_train_dataloader
from libs.model import Model
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)

    args = parser.parse_args()
    return args


def train():
    args = parse_arguments()
    config = load_config(args.config_path)

    model = Model()

    criterion = None

    optimizer = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataFetcher(get_train_dataloader(config), device=device)
    get_eval_dataloader(config)

    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        writer=None,
    )

    trainer.train()


if __name__ == "__main__":
    train()
