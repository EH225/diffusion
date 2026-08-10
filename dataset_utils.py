# -*- coding: utf-8 -*-
"""
This module contains functions for pre-processing images to be a given size e.g. [128, 128, 3] and also
code for dataloaders for various datasets.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
from torchvision import transforms, datasets
from utils import get_device
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


###                      ###
### Image Pre-Processing ###
###                      ###
def resize_and_crop(input_dir: str, output_dir: str, size: int) -> None:
    """
    Resizes and center-crops all images found in input_dir and saves a new copy to output_dir
    with the same name. This is used as a dataset pre-processing step to create images that are
    all the same size.

    :param input_dir: A directory of input images to read from.
    :param output_dir: The directory where processed images are to be saved.
    :param size: The size to create i.e. (size, size, 3) for each image.
    :return: None. Files are read and saved to disk.
    """
    os.makedirs(output_dir, exist_ok=True)  # Make sure the output_dir exists, create if needed
    # Define image transforms to apply to each image
    resize = Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    center_crop = CenterCrop(size)

    valid_exts = {"jpg", "jpeg", "png"}

    for img_name in tqdm(os.listdir(input_dir), ncols=75):  # Loop over all images from input_dir
        if img_name.split(".")[-1].lower() not in valid_exts:  # Check that this is an image file
            continue

        with Image.open(os.path.join(input_dir, img_name)) as img:  # Open the image
            img = img.convert("RGB")
            img = resize(img)  # Resize it to be a min(H, W) = size
            img = center_crop(img)  # Center crop so that we have [size, size, 3] images
            # Save the image to the output directory when finished processing
            img.save(os.path.join(output_dir, img_name))


###          ###
### Datasets ###
###          ###

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset, which returns [32 x 32 x 3] images and has 10 classes labeled [0-9].
    """

    def __init__(self, datasets_dir: str, split: str = "train", transform=None):
        """
        Initializes the dataset, downloads data if needed.

        :param datasets_dir: A directory where the datasets exist or should be maintained.
        :param split: The dataset split to use in this dataset i.e. "train" or "val".
        :param transform: A set of transforms to apply to each image before it is batched.
        """
        self.dataset = datasets.CIFAR10(root=datasets_dir, train=(split == "train"),
                                        download=True, transform=transform)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing keys "image" and "class_id" for a particular index in the dataset.

        :param idx: An internal image index number from [0, len(self.img_ids) - 1].
        :returns: A dict with the following format:
            image: An image of size (C=3, H=32, W=32) as a torch.Tensor
            class_id: The class_id associated with the image, a unique group identifier
        """
        image, label = self.dataset[idx]

        return {"image": image, "class_id": label}


class CelebADataset(Dataset):
    """
    celebA dataset, which returns [64 x 64 x 3] images and has 2 classes labeled [0-1].
    """

    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        """
        Initializes the celebA dataset.

        :param dataset_dir: A directory containing an images/ folder and meta_data.csv file.
        :param split: The dataset split to use in this dataset i.e. "train" or "val".
        :param transform: A set of transforms to apply to each image before it is batched.
        """
        self.img_dir = os.path.join(dataset_dir, "images")
        self.meta_df = pd.read_csv(os.path.join(dataset_dir, "meta_data.csv"))
        self.meta_df = self.meta_df.loc[self.meta_df["split"] == split, :]  # Subset for the specified split
        self.transform = transform
        self.image_names = self.meta_df["image_name"].tolist()
        # Create a dictionary of class_id value for every image name for quick retrival
        self.class_id = {row["image_name"]: int(row["class_id"]) for _, row in self.meta_df.iterrows()}

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "image" and "class_id" for a particular index in the dataset.

        :param idx: An internal image index number from [0, len(self.img_ids) - 1].
        :returns: A dict with the following format:
            image: An image of size (C=3, H=64, W=64) as a torch.Tensor
            class_id: The class_id associated with the image, a unique group identifier
        """
        # Load in the image and labels from disk
        image_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:  # Apply transforms if specified
            image = self.transform(image)

        class_id = torch.tensor(self.class_id[image_name], dtype=torch.long)
        return {"image": image, "class_id": class_id}


class AFHQDataset(Dataset):
    """
    Animal Faces High-Quality (AFHQ) dataset, which returns [128 x 128 x 3] images and has 3 classes
    labels [0-2].
    """

    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        """
        Initializes an Oxford Pets dataset.

        :param dataset_dir: A directory containing an images/ folder and meta_data.csv file.
        :param split: The dataset split to use in this dataset i.e. "train" or "val".
        :param transform: A set of transforms to apply to each image before it is batched.
        """
        self.img_dir = os.path.join(dataset_dir, "images")
        self.meta_df = pd.read_csv(os.path.join(dataset_dir, "meta_data.csv"))
        self.meta_df = self.meta_df.loc[self.meta_df["split"] == split, :]  # Subset for the specified split
        self.transform = transform
        self.image_names = self.meta_df["image_name"].tolist()
        # Create a dictionary of class_id value for every image name for quick retrival
        self.class_id = {row["image_name"]: row["class_id"] for _, row in self.meta_df.iterrows()}

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "image" and "class_id" for a particular index in the dataset.

        :param idx: An internal image index number from [0, len(self.img_ids) - 1].
        :returns: A dict with the following format:
            image: An image of size (C=3, H=128, W=128) as a torch.Tensor
            class_id: The class_id associated with the image, a unique group identifier
        """
        # Load in the image and labels from disk
        image_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, f"{image_name}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:  # Apply transforms if specified
            image = self.transform(image)

        class_id = torch.tensor(self.class_id[image_name], dtype=torch.long)
        return {"image": image, "class_id": class_id}


###             ###
### Dataloaders ###
###             ###

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Add random horizontal flip data augmentations
    transforms.ToTensor(),  # Convert to a tensor, [0, 255] -> [0.0, 1.0] pixel values
    transforms.Normalize(
        # Normalize from [0, 1] to [-1, 1] pixel values to match the generator
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to a tensor, [0, 255] -> [0.0, 1.0] pixel values
    transforms.Normalize(
        # Normalize from [0, 1] to [-1, 1] pixel values to match the generator
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


def get_dataloader(datasets_dir: str, dataset: str, split: str, batch_size: int) -> DataLoader:
    """
    Returns a Dataloader for the split specified i.e. "train" or "val".

    :param datasets_dir: The directory where the datasets are stored, typically called "datasets".
    :param dataset: The name of the dataset i.e. "cifar10", "celebA", "afhq" etc.
    :param split: The dataset split to construct a dataloader for i.e. train or val.
    :param batch_size: The number of images in each batch.
    :return: A dataloader for the given split specified in the input args.
    """
    device = get_device()  # Auto-detect what hardware is available
    if device == "cuda":  # Change kwargs depending on the device in use
        num_workers, pin_memory, persistent_workers, prefetch_factor = 8, True, True, 4
    else:
        num_workers, pin_memory, persistent_workers, prefetch_factor = 0, False, False, None

    if split == "train":
        transform = train_transform
        shuffle = True
    elif split == "val":
        transform = val_transform
        shuffle = False
    else:
        raise KeyError(f"split={split} not recognized")

    if dataset == "cifar10":
        dataset = CIFAR10Dataset(datasets_dir, split, transform)
    elif dataset == "celebA":
        dataset = CelebADataset(os.path.join(datasets_dir, "celebA"), split, transform)
    elif dataset == "afhq":
        dataset = AFHQDataset(os.path.join(datasets_dir, "afhq"), split, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor)


def get_class_labels(dataset: str) -> dict:
    """
    Returns a dictionary mapping of class_id (int) to class_name (str) for a given dataset.
    """
    if dataset == "cifar10":
        return {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
    elif dataset == "celebA":
        return {
            0: "male",
            1: "female"
        }
    elif dataset == "afhq":
        return {
            0: "dog",
            1: "cat",
            2: "wild",
        }


if __name__ == "__main__":
    # Pre-process all the images on disk for the AFHQ dataset to be [128, 128, 3]
    print("Pre-processing AFHQ dataset images to be [128, 128, 3]")
    input_dir = os.path.join(CURRENT_DIR, "datasets", "afhq", "original")
    output_dir = os.path.join(CURRENT_DIR, "datasets", "afhq", "images")
    resize_and_crop(input_dir, output_dir, 128)

    # print("Pre-processing celebA dataset images to be [64, 64, 3]")
    # input_dir = os.path.join(CURRENT_DIR, "datasets", "celebA", "img_align_celeba")
    # output_dir = os.path.join(CURRENT_DIR, "datasets", "celebA", "images")
    # resize_and_crop(input_dir, output_dir, 64)
