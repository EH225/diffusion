"""
This script is used to train the U-Net based diffusion model.
"""
import sys, os, argparse

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

from ddpm_trainer import Trainer
from dataset_utils import get_dataloader
from utils import read_yaml


def run_training(config_name: str, datasets_dir: str, **kwargs) -> None:
    """
    This helper function runs training of the model for a given config file specified by
    config_name. This function:
        1. Reads in the config file as a dict
        2. Builds the train and val dataloaders
        3. Constructs the Trainer obj
        4. Run the training loop by calling trainer.train()

    :param config_name: The name of the config file to use for running training.
    :param datasets_dir: The directory where the datasets are stored, typically called "datasets".
    :return: None.
    """
    config = read_yaml(os.path.join(CURRENT_DIR, "config", f"{config_name}.yml"))
    dataloaders = {
        "train": get_dataloader(datasets_dir, config["dataset"], "train",
                                config["training"]["batch_size"]),
        "val": get_dataloader(datasets_dir, config["dataset"], "val",
                              config["eval"]["batch_size"])}
    trainer = Trainer(config=config, dataloaders=dataloaders)
    trainer.train(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDPM model training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="The name of the config file to be used for training.")
    args = parser.parse_args()
    datasets_dir = os.path.join(CURRENT_DIR, "datasets")
    run_training(args.config, datasets_dir)
