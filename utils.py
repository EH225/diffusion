"""
This module contains general utility functions.
"""

import yaml
import os
import logging
import torch
from typing import Tuple, List, Dict
import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt


def read_yaml(file_path: str) -> dict:
    """
    Helper function that reads in a yaml file specified and returns the associated data as a dict.

    :param file_path: A str denoting the location of a yaml file to read in.
    :return: A dictionary of data read in from the yaml file located at file_path.
    """
    return yaml.safe_load(open(file_path))


def get_device() -> str:
    """
    Auto-detects what hardware is available and returns a device name accordingly.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:  # Default to using the CPU if no GPU accelerator
        return "cpu"


def get_amp_dtype(device: str = "cuda"):
    """
    Determines the Automatic Mixed Precision data type that can be used on the current hardware.

    :param device: The device currently available as a string e.g. "cpu" or "cuda".
    :returns: A torch float type for auto mixed precision training.
    """
    assert isinstance(device, str), "device must be a str"
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float16

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()

    # Ampere (8.x), Hopper (9.x), Ada (8.9) → BF16 supported
    bf16_supported = (major >= 8)

    return torch.bfloat16 if bf16_supported else torch.float16


def generate_loss_plots(loss_dir: str, save_dir: str) -> None:
    """
    This function will read in the loss files cached to loss_dir and create plots and save them to save_dir
    so that we can automatically visualize the progression of the loss curves during training.

    :param loss_dir: The location where the loss .csv files are cached.
    :param save_dir: The location where the output loss curves plots will be saved to.
    """
    # 1). Read in values from disk and collect them into lists of dataframes
    train_loss, val_loss = [], []

    for filename in os.listdir(loss_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(loss_dir, filename), index_col=0)
            if filename.split("-")[0] == "train":
                train_loss.append(df)
            else:
                val_loss.append(df)

    # 2). Convert from lists of dataframes into 1 consolidated dataframe, sort by train step
    if len(train_loss) > 0:
        train_loss = pd.concat(train_loss).sort_values("step")
        train_loss.index = train_loss.step
        train_loss.drop("step", inplace=True, axis=1)

    if len(val_loss) > 0:
        val_loss = pd.concat(val_loss).sort_values("step")
        val_loss.index = val_loss.step
        val_loss.drop("step", inplace=True, axis=1)

    # 3). Generate and save a plot of the training loss
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))

    if len(train_loss) > 0:
        ax = axes[0]
        ax.plot(train_loss["loss"].rolling(50, min_periods=1).mean())  # Apply smoothing
        ax.set_title("Train loss")
        ax.grid(color="lightgray")

    if len(val_loss) > 0:
        ax = axes[1]
        ax.plot(val_loss["loss"])
        ax.set_title("Val loss")
        ax.grid(color="lightgray")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_plots.png"))
    plt.close(fig)


def save_images(images: torch.Tensor, titles: List[str], ncol: int = 4, save_path: str = None):
    """
    Helper function to save an input tensor of images to disk with titles for each image.

    :param images: A tensor of images of size (B, C, H, W).
    :param titles: A list of titles to assign to each image.
    :param ncol: The number of columns to have in the image grid. The default is 4.
    :param save_path: A full file path specifying where to save the images.
    :returns: None. Images are saved to disk.
    """
    nrow, r = divmod(len(images), ncol)
    nrow += (r > 0) * 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(15 / 8 * ncol, nrow * 2.2))
    axes = axes.reshape(-1)

    for ax, img, title in zip(axes, images, titles):
        img = img.permute(1, 2, 0).float().cpu()  # (C, H, W) -> (H, W, C)
        img = ((img + 1) / 2).clamp(0, 1)  # Rescale from [-1, +1] back to [0, 1]

        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images):]:  # Hide axes that do not have images displayed
        ax.axis("off")

    plt.tight_layout()
    save_path = "samples.png" if save_path is None else save_path
    plt.savefig(save_path)
    plt.close(fig)
