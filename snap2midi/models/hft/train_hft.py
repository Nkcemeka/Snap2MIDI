# Imports
from pathlib import Path
from snap2midi.models.hft.hft import *
from .hft_dataset import HFTDivDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from snap2midi.utils.train_utils import pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint

class HFTDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        """
            Instantiate data module.

            Args
            -----
                config (dict): Configuration params.
        """
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        self.train_dataset = HFTDivDataset(self.config, split="train", shuffle=True)
        self.val_dataset = HFTDivDataset(self.config, split="val", shuffle=False)

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset,\
            batch_size=self.config["batch_size"], \
            num_workers=self.config["num_workers"])
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, \
            batch_size=self.config["batch_size"], \
            num_workers=self.config["num_workers"])

def main(config):
    # Create datasets and set seed
    pl.seed_everything(config["seed"], workers=True)
    dm = HFTDataModule(config)

    # Load/initialize the model
    model = HFT(config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/feature/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_total_loss',
            filename='hft-{epoch:02d}-{valid_total_loss:.4f}',
            dirpath=config["save_dir"],
            save_top_k=5,
            mode="min"
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
        )

    # create trainer
    trainer = pl.Trainer(max_epochs=config["epochs"], \
        deterministic=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        num_nodes=config["num_nodes"],
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])
