# Imports
import pytorch_lightning as pl
from .hpp import HPPNet
from .dataset_hpp import HPPDataset
from torch.utils.data.dataloader import DataLoader
from snap2midi.utils.train_utils import pl_logger
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

class HPPDataModule(pl.LightningDataModule):
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
        train_path = f"{self.config["base_path"].rstrip('/')}/train/"
        val_path = f"{self.config["base_path"].rstrip('/')}/val/"

        assert Path(train_path).exists(), f"[TRAIN PATH]: {train_path} does not exist!"
        self.train_dataset = HPPDataset(self.config, [f"{train_path}"])

        # We don't necessarily need a validation dataset
        if Path(val_path).exists():
            self.val_dataset = HPPDataset(self.config, [f"{val_path}"])
        else:
            self.val_dataset = None

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                        num_workers=self.config["num_workers"], shuffle=True)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], \
                          num_workers=self.config["num_workers"], shuffle=False)

def main(config):
    # Create datasets 
    dm = HPPDataModule(config)
    model = HPPNet(config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',
            filename='hpp-{epoch:02d}-{val_total_loss:.2f}',
            dirpath=config["save_dir"],
            save_top_k=5,
            mode="min"
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
        )

    # create trainer
    trainer = pl.Trainer(max_steps=config["iterations"], \
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config["clip_gradient_norm"],
        num_nodes=config["num_nodes"],
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])
