# Imports
from pathlib import Path
from snap2midi.models.hft.hft import *
from .hft_dataset import HFTDataset
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
        self.train_dataset = HFTDataset(self.config, split="train")
        self.val_dataset = HFTDataset(self.config, split="val")
        self.test_dataset = HFTDataset(self.config, split="test")

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                        num_workers=self.config["num_workers"], shuffle=True)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], \
                          num_workers=self.config["num_workers"], shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return DataLoader(self.test_dataset, batch_size=self.config["batch_size"], \
                          num_workers=self.config["num_workers"], shuffle=False)
    

def main(config):
    # Create datasets 
    dm = HFTDataModule(config)

    # Load/initialize the model
    encoder = HFTEncoder(
        n_margin=config['margin_b'],
        n_frame=config['num_frame'],
        n_bin=config['n_bins'],
        cnn_channel=config["cnn_channel"],
        cnn_kernel=config["cnn_kernel"],
        d=config["d"],
        n_layers=config["enc_layer"],
        num_heads=config["enc_head"],
        pff_dim=config["pff_dim"],
        dropout=config["dropout"]
    )

    decoder = HFTDecoder(n_frame=config['num_frame'],
                         n_bin=config['n_bins'],
                         n_note=config['num_note'],
                         n_velocity=config['num_velocity'],
                         d=config["d"],
                         n_layers=config["dec_layer"],
                         num_heads=config["dec_head"],
                         pff_dim=config["pff_dim"],
                         dropout=config["dropout"])
    
    model = HFT(encoder, decoder, config)

    # create checkpoint callback
    val_flag = Path(f"{config["base_path"].rstrip('/')}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_total_loss',
            filename='hft-{epoch:02d}-{valid_total_loss:.2f}',
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
    
    test_path = config.get("test_path", None)
    if test_path is not None and Path(test_path).exists():
        trainer.test(
            model=model,
            datamodule=dm,
            ckpt_path="best"
        )


