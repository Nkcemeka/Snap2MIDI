# Imports
from pathlib import Path
from snap2midi.models.hft.hft import *
from .hft_dataset import HFTDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pytorch_lightning.callbacks import ModelCheckpoint


class EpochUpdateCallback(pl.Callback):
    """Tell the dataset which epoch it is, so augmentation is redrawn.

    The augmentator seeds each excerpt from (track_id, excerpt_start, epoch).
    Leave `epoch` at 0 and every epoch applies the *identical* damage to the
    identical excerpt -- no crash, no warning, just a fixed pre-corrupted
    dataset instead of augmentation. That is the whole regularization effect,
    silently gone.

    Setting the attribute here reaches the workers because they are re-forked
    each epoch, which is true only while `persistent_workers` is False. Turn it
    on for speed and the workers keep their stale copy and the epoch freezes --
    exactly the failure above, reintroduced by an unrelated tweak. So the
    coupling is checked against the live dataloader on every epoch rather than
    left as a comment.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        dataset = trainer.datamodule.train_dataset
        if dataset.augmentator is not None:
            loader = trainer.train_dataloader
            if getattr(loader, "persistent_workers", False):
                raise RuntimeError(
                    "hFT augmentation needs persistent_workers=False: workers "
                    "must re-fork each epoch to pick up the new epoch number. "
                    "With persistent workers the augmentation silently freezes "
                    "at the first epoch's draw.")
        dataset.epoch = trainer.current_epoch


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

        # Train only. Augmenting validation would make the loss curve partly a
        # measure of the augmentation draw rather than of the model, and the
        # checkpoint selection monitors valid_total_loss.
        if self.config.get("augment"):
            if self.config.get("feature_source", "audio") != "audio":
                raise ValueError(
                    "augment=True needs feature_source='audio'; the legacy "
                    "store holds spectrograms, so there is no waveform to "
                    "augment.")
            if not self.config.get("augment_asset_root"):
                raise ValueError(
                    "augment=True requires augment_asset_root, the directory "
                    "holding room_ir/ and bg_noise/.")
            # sample_rate comes from the extraction meta, not the training
            # config, so the assets are resampled to whatever the audio slab
            # actually is.
            self.train_dataset.augmentator = Augmentator(
                sample_rate=self.train_dataset.meta["sr"],
                asset_root=self.config["augment_asset_root"],
                split="train",
                manifest_dir=self.config.get("augment_manifest_dir"),
                seed=self.config["seed"],
                reverb_level=self.config.get("reverb_level", "peak"),
            )

    # Below are methods for setting up the dataloaders.
    # shuffle and worker sharding are PyTorch's job now. The dataset used to do
    # both by hand and got both wrong: it split work by division (so 3 of 4
    # workers idled) and reshuffled from a generator that never advanced (so
    # every epoch saw the same order).
    def train_dataloader(self):
        # persistent_workers stays False on purpose -- see EpochUpdateCallback.
        return DataLoader(self.train_dataset,\
            batch_size=self.config["batch_size"], \
            shuffle=True, \
            persistent_workers=False, \
            num_workers=self.config["num_workers"])

    def val_dataloader(self):
        if self.val_dataset is None:
            return []

        return DataLoader(self.val_dataset, \
            batch_size=self.config["batch_size"], \
            shuffle=False, \
            num_workers=self.config["num_workers"])

def main(config):
    # Create datasets and set seed
    pl.seed_everything(config["seed"], workers=True)
    dm = HFTDataModule(config)

    # Load/initialize the model
    model = HFT(config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_dir = "feature" if config.get("feature_source") == "legacy_feature" else "audio"
    val_flag = Path(f"{base_path}/{val_dir}/val/").exists()
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
        callbacks=[checkpoint_callback, EpochUpdateCallback()],
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
