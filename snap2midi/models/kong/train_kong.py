# Imports
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import h5py
from .kong_dataset import KongDataset, collate_fn
from .kong import KongModel
import pytorch_lightning as pl
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pytorch_lightning.callbacks import ModelCheckpoint


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
                    "kong augmentation needs persistent_workers=False: workers "
                    "must re-fork each epoch to pick up the new epoch number. "
                    "With persistent workers the augmentation silently freezes "
                    "at the first epoch's draw.")
        dataset.epoch = trainer.current_epoch


class KongDataModule(pl.LightningDataModule):
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
        # Train only. Augmenting validation would make the loss curve partly a
        # measure of the augmentation draw rather than of the model, and the
        # checkpoint selection monitors valid_total_loss.
        augmentator = None
        if self.config.get("augment"):
            if not self.config.get("augment_asset_root"):
                raise ValueError(
                    "augment=True requires augment_asset_root, the directory "
                    "holding room_ir/ and bg_noise/.")
            augmentator = Augmentator(
                sample_rate=self.extraction_config["sample_rate"],
                asset_root=self.config["augment_asset_root"],
                split="train",
                manifest_dir=self.config.get("augment_manifest_dir"),
                seed=self.config.get("seed", 1234),
                reverb_level=self.config.get("reverb_level", "peak"),
            )

        self.train_dataset = KongDataset(f"{self.config["base_path"].rstrip('/')}/train/",\
            extend_pedal=self.extraction_config["extend_pedal"],
            augmentator=augmentator)
        self.val_dataset = KongDataset(f"{self.config["base_path"].rstrip('/')}/val/",\
            extend_pedal=self.extraction_config["extend_pedal"])

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        # persistent_workers stays False on purpose -- see EpochUpdateCallback.
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                pin_memory=True, collate_fn=collate_fn, persistent_workers=False, \
                num_workers=self.config["num_workers"], shuffle=True)
    
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
    dm = KongDataModule(config, extraction_config)
    
    # Load/initialize the model
    model = KongModel(config, extraction_config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_total_loss',
            filename='kong-step={step}-loss={valid_total_loss:.4f}',
            dirpath=config["save_dir"],
            save_top_k=5,
            mode="min",
            # No every_n_train_steps: it saves from on_train_batch_end, which
            # runs before that step's validation loop, so each checkpoint was
            # ranked and named with the *previous* validation's loss -- the
            # top-5 were the weights one interval after the ones that earned
            # the score. Saving at validation end instead pairs each checkpoint
            # with its own metric. Validation cadence is val_check_interval on
            # the Trainer, so the save cadence is unchanged.
            #
            # save_last is additive to save_top_k: `last.ckpt` is whatever step
            # training stopped at, which the top-5 need not contain. Resuming
            # from a best-loss checkpoint instead would rewind the optimizer
            # moments, the LR schedule and the step counter to wherever that was.
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
        )

    # create trainer
    trainer = pl.Trainer(max_steps=config["iterations"], \
        callbacks=[checkpoint_callback, EpochUpdateCallback()],
        num_sanity_val_steps=0,
        num_nodes=config["num_nodes"],
        val_check_interval=config["val_steps"],
        # Metric cadence is independent of validation cadence: tying them means
        # the first training-loss point lands one full validation interval in,
        # which on a long run is hours before anyone can tell it is learning.
        log_every_n_steps=config.get("log_steps") or config["val_steps"],
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])

    # Save the endpoint unconditionally. ModelCheckpoint writes only at a
    # validation, and then only if that loss lands in the top five, so the final
    # model otherwise survives by coincidence. It did not for kong_paper_aug:
    # val_check_interval counts batches *within* an epoch and resets at the
    # boundary, so once an epoch began off the global grid (188,868) the
    # validations fell at 193,868 and 198,868, never on 200,000 -- and the last
    # one scored 0.7027, outside the top five, and was discarded. The run
    # trained 6,132 iterations past its last saved state and exited with them
    # unrecorded. Picking iterations as a multiple of val_steps, which is what
    # run_kong_experiment.py relies on, only holds for runs short enough to stay
    # inside one epoch.
    endpoint = Path(config["save_dir"]) / f"kong-endpoint-step={trainer.global_step}.ckpt"
    trainer.save_checkpoint(endpoint)
    print(f"saved endpoint checkpoint: {endpoint}")
    