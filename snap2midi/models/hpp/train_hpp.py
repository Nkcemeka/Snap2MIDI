# Imports
import pytorch_lightning as pl
from .hpp import HPPNet
from .dataset_hpp import HPPDataset
from torch.utils.data.dataloader import DataLoader
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class EpochUpdateCallback(pl.Callback):
    """Tell the dataset which epoch it is, so augmentation is redrawn.

    The augmentator seeds each excerpt from (track_id, excerpt_start, epoch).
    Leave `epoch` at 0 and every pass applies the identical draw to the
    identical excerpt, which is a fixed pre-corrupted dataset rather than
    augmentation -- and it fails silently.

    Setting the attribute here reaches the workers only because they re-fork
    each epoch, which holds while `persistent_workers` is False. Turning it on
    for speed would freeze the epoch with no error, so the coupling is checked
    against the live dataloader rather than left as a comment.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        dataset = trainer.datamodule.train_dataset
        if dataset.augmentator is not None:
            loader = trainer.train_dataloader
            if getattr(loader, "persistent_workers", False):
                raise RuntimeError(
                    "hpp augmentation needs persistent_workers=False: workers "
                    "must re-fork each epoch to pick up the new epoch number. "
                    "With persistent workers the augmentation silently freezes "
                    "at the first epoch's draw.")
        dataset.epoch = trainer.current_epoch


def build_augmentator(config: dict):
    """Construct the training augmentator, or None for a clean run."""
    if not config.get("augment"):
        return None
    if not config.get("augment_asset_root"):
        raise ValueError(
            "augment=True requires augment_asset_root, the directory "
            "holding room_ir/ and bg_noise/.")
    return Augmentator(
        sample_rate=config["sample_rate"],
        asset_root=config["augment_asset_root"],
        split="train",
        manifest_dir=config.get("augment_manifest_dir"),
        seed=config.get("seed", 1234),
        reverb_level=config.get("reverb_level", "peak"),
    )


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

        # Train only. Augmenting validation would make the loss curve partly a
        # measure of the augmentation draw rather than of the model, and the
        # checkpoint selection monitors the validation loss.
        self.train_dataset = HPPDataset(self.config, [f"{train_path}"],
                                        augmentator=build_augmentator(self.config))

        # We don't necessarily need a validation dataset
        if Path(val_path).exists():
            self.val_dataset = HPPDataset(self.config, [f"{val_path}"])
        else:
            self.val_dataset = None

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        # persistent_workers stays False on purpose -- see EpochUpdateCallback.
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                        num_workers=self.config["num_workers"], shuffle=True, \
                        persistent_workers=False)
    
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
    callbacks = [EpochUpdateCallback()]

    if val_flag:
        # Rank on val_loss/all, the sum of the four head losses. val_total_loss
        # would order runs identically -- for every model_type the subnet sums
        # re-partition the same heads, so it is exactly 3x val_loss/all -- but
        # this is the one whose value means something on its own.
        #
        # The metric stays out of `filename`: ModelCheckpoint interpolates the
        # template, and the '/' in the key would be read as a directory
        # separator. Putting the loss in the name needs a slash-free alias
        # logged alongside it, the way transkun carries val_f1 next to val/f1.
        callbacks.append(ModelCheckpoint(
            dirpath=config["save_dir"],
            filename="hpp-{step}",
            monitor="val_loss/all",
            mode="min",
            save_top_k=5,
            save_last=True
        ))
        # HPPNet trains for 200k-500k steps and stops early; max_steps alone
        # only supplies the ceiling. Patience counts validation checks rather
        # than steps, so at check_val_every_n_epoch=2 these 25 checks span far
        # more than learning_rate_decay_steps -- a run is not killed just
        # before a decay that might still have pulled the loss down.
        callbacks.append(EarlyStopping(
            monitor="val_loss/all",
            mode="min",
            patience=25
        ))
    else:
        # No validation split, so there is nothing to rank against: fall back
        # to dumping periodically and pick the checkpoint by hand.
        callbacks.append(ModelCheckpoint(
            dirpath=config["save_dir"],
            filename="hpp-{step}",
            every_n_train_steps=2000,
            save_top_k=-1,
            save_last=True
        ))

    # create trainer
    trainer = pl.Trainer(max_steps=config["iterations"], \
        callbacks=callbacks,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        num_nodes=config["num_nodes"],
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])
