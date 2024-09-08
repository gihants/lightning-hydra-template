import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import transforms

from data.components.bscan_classification_data_augmentation import (
    BScanClassificationImageTransform,
)


class BscanClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        img_size: int = 112,
        batch_size: int = 8,
        num_workers: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=BScanClassificationImageTransform(
                is_train=True, img_size=self.img_size
            ),
        )
        self.val_dataset = ImageFolder(
            root=os.path.join(data_dir, "validation"),
            transform=BScanClassificationImageTransform(
                is_train=False, img_size=self.img_size
            ),
        )
        self.test_dataset = ImageFolder(
            root=os.path.join(data_dir, "test"),
            transform=BScanClassificationImageTransform(
                is_train=False, img_size=self.img_size
            ),
        )
        self.classes = self.train_dataset.classes
        self.class_to_idx = self.train_dataset.class_to_idx

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader


if __name__ == "__main__":
    _ = BscanClassificationDataModule()
