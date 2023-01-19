import os
import datetime

import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_module import SegmentatorModule
from config import TrainSegmentatorConfig

def train():
    """Train surface model"""
    
    # Load config
    config = TrainSegmentatorConfig()
    np.random.seed(config.seed)

    # Init out directories
    ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    logs_path = os.path.join(config.output_dir, ct, "logs")
    checkpoint_path = os.path.join(config.output_dir, ct, "checkpoint")

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("\nLogging to:", logs_path)

    # Init model
    net = SegmentatorModule(config)

    # Set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        max_epochs=200,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    # Fit model
    trainer.fit(net)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train()
