import os

import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

import monai
from monai.data import list_data_collate
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImaged

from src.metrics.confusion_matrix import ConfusionMatrix
from src.transforms.transforms import (
    DeleteChannelsd, 
    PointcloudRandomSubsampled, 
    PointcloudRandomAffined, 
    PointcloudRandomJitterd, 
    ExtractSegmentationLabeld, 
    ToFloatTensord
)


class SegmentatorModule(pl.LightningModule):
    """
    Training module in pytorch lightning.
    Can run any given architecture as long as it matches the scheme.
    """

    def __init__(self, config):
        super().__init__()

        set_determinism(seed=config.seed)
        self.save_hyperparameters()

        self.size = config.size
        self.input_dir = config.input_dir
        self.optimizer = config.optimizer
        self.epochs = config.epochs
        self.num_workers = config.num_workers

        self.model = config.model(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.confmat = ConfusionMatrix(num_classes=2)

    def prepare_data(self):
        
        # Extract filenames and combain into list of dicts
        train_names = os.listdir(os.path.join(self.input_dir, "train"))
        val_names = os.listdir(os.path.join(self.input_dir, "valid"))

        train_dict = [{"input": os.path.join(self.input_dir, train_name)} for train_name in train_names]
        val_dict = [{"input": os.path.join(self.input_dir, val_name)} for val_name in val_names]

        # Transforms
        train_transforms = Compose([
            LoadImaged(keys=["input"], reader="NumpyReader"),
            DeleteChannelsd(keys=["input"], channels=[3]),
            # Augmentation
            PointcloudRandomSubsampled(keys=["input"], sub_size=self.size),
            PointcloudRandomAffined(
                keys=["input"],
                prob=1.0,
                rotate_range=[np.pi / 6, np.pi / 6, np.pi / 6],
                translate_range=[0.01, 0.01, 0.01],
            ),
            PointcloudRandomJitterd(keys=["input"], std=0.0002),
            # Prepare for training
            ExtractSegmentationLabeld(pcd_key="input"),
            ToFloatTensord(keys=["input", "label"])
        ])

        valid_transforms = Compose([
            LoadImaged(keys=["input"], reader="NumpyReader"),
            DeleteChannelsd(keys=["input"], channels=[3]),
            PointcloudRandomSubsampled(keys=["input"], sub_size=self.size),
            ExtractSegmentationLabeld(pcd_key="input"),
            ToFloatTensord(keys=["input", "label"]),
        ])

        # Datasets
        self.train_ds = monai.data.CacheDataset(
            data=train_dict,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=self.num_workers,
        )

        self.val_ds = monai.data.CacheDataset(
            data=val_dict,
            transform=valid_transforms,
            cache_rate=1.0,
            num_workers=self.num_workers,
        )
    
    def train_dataloader(self):
        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = monai.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
        )

        return val_loader

    def forward(self, x):
        output = self.model(x)
        return output

    def calc_metrics(self, one_hot_logits, y):
        logits = torch.argmax(one_hot_logits, dim=1)

        # Compute arteries against artifacts
        y = (y > 0).long()
        logits = (logits > 0).long()

        self.confmat(logits, y)

        return {
            "accuracy": self.confmat.accuracy().item(),
            "precision": self.confmat.precision().item(),
            "recall": self.confmat.recall().item(),
            "f1_score": self.confmat.f1_score().item(),
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["input"], train_batch["label"]
        logits = self.forward(x)

        loss = self.loss_fn(logits, y.long())

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["input"], val_batch["label"]
        logits = self.forward(x)

        loss = self.loss_fn(logits, y.long())

        metrics = self.calc_metrics(logits, y)
        metrics = {"val_" + name: value for name, value in metrics.items()}

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
