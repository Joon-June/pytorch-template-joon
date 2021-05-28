import argparse
from libs.trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config yaml file")

    args = parser.parse_args()
    return args

def train():
    args = parse_arguments()
    config = load_config(args.config_path)

    trainer = Trainer(model=,
                      criterion=,
                      optimizer=,
                      config=config,
                      device=,
                      train_data_loader=,
                      valid_data_loader=,
                      lr_scheduler=,
                      early_stopper=None)

    trainer.train()

if __name__ == '__main__':
    train()