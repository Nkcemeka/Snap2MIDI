# Imports
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import h5py
from .kong_dataset import KongDataset, collate_fn
from .kong import KongPedal
import pytorch_lightning as pl
from snap2midi.utils.train_utils import pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint

class KongPedalDataModule(pl.LightningDataModule):
    def __init__(self, config: dict, extraction_config: dict):
        """
            Instantiate data module.

            Args
            -----
                config (dict): Configuration params.
        """
        super().__init__()
        self.config = config
        self.extraction_config = extraction_config
    
    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        self.train_dataset = KongDataset(f"{self.config["base_path"].rstrip('/')}/train/",\
            extend_pedal=self.extraction_config["extend_pedal"])
        self.val_dataset = KongDataset(f"{self.config["base_path"].rstrip('/')}/val/",\
            extend_pedal=self.extraction_config["extend_pedal"])

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                pin_memory=True, collate_fn=collate_fn, num_workers=self.config["num_workers"], shuffle=True)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], \
            pin_memory=True, collate_fn=collate_fn, num_workers=self.config["num_workers"], shuffle=False)
    
def main(config):
    extraction_config_path = f"{config["base_path"]}/extraction_config.h5"
    with h5py.File(extraction_config_path, "r") as hf:
        extraction_config = dict(hf.attrs)

        # convert byte strings to normal strings and np.ints to int etc
        for key in extraction_config.keys():
            if isinstance(extraction_config[key], bytes):
                extraction_config[key] = extraction_config[key].decode("utf-8")
            elif isinstance(extraction_config[key], np.generic):
                extraction_config[key] = extraction_config[key].item()

    # Create datasets 
    dm = KongPedalDataModule(config, extraction_config)
    
    # Load/initialize the model
    model = KongPedal(config, extraction_config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_total_pedal_loss',
            filename='kongpedal-step={step}',
            dirpath=config["save_dir"],
            save_top_k=-1,
            every_n_train_steps=20000,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
        )

    # create trainer
    trainer = pl.Trainer(
        max_steps=config["iterations"],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        num_nodes=config["num_nodes"],
        val_check_interval=config["val_steps"],
        log_every_n_steps=config["val_steps"],
        logger=pl_logger(config["logger_name"], 
        project_name=config["experiment_name"])
    )
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])
