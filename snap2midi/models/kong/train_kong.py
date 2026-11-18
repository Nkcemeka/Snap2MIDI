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
    
def _init_weights_from(model, path: str, extraction_config: dict) -> None:
    """Weights only, strict. See Trainer.train_kong(init_weights_from=...).

    The checkpoint's front end (sample rate, mel bins, frame rate, pitch range,
    window) must be the one the store was extracted with, or the weights would
    see features of another shape; checked before loading.
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    front = ("sample_rate", "n_mels", "mel_n_fft", "frame_rate", "hop_length",
             "min_pitch", "max_pitch", "window_size", "hop_size")
    pre = ckpt["hyper_parameters"]["extraction_config"]
    bad = {k: (pre.get(k), extraction_config.get(k)) for k in front
           if pre.get(k) != extraction_config.get(k)}
    if bad:
        raise SystemExit(f"[init_weights_from]: front-end mismatch between {path} "
                         f"and the store: {bad}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"[init_weights_from]: loaded weights from {path} "
          f"(trained {ckpt.get('global_step')} steps); optimizer, schedule "
          "and step counter start fresh")


def main(config, model_hook=None):
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
    if config.get("init_weights_from") and config["resume_path"] is None:
        _init_weights_from(model, config["init_weights_from"], extraction_config)
    if model_hook is not None:
        # Freezing / adapters: after the pretrained weights, before the
        # optimizer is built; on a resume too, so the model already carries
        # any adapter when Lightning restores the checkpoint.
        model_hook(model)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if config.get("ckpt_every_n_steps"):
        # A fixed grid of checkpoints, all kept, for selection offline
        # (experiments/select/pick.py) -- the fine-tuning runs.
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
            filename="kong-{step}",
            every_n_train_steps=config["ckpt_every_n_steps"],
            save_top_k=-1,
            save_last=True,
        )
    elif val_flag:
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

    # Save the endpoint unconditionally: ModelCheckpoint writes only at a
    # validation, and val_check_interval counts batches within an epoch, so the
    # last step need not fall on one and the final model could otherwise be
    # lost.
    endpoint = Path(config["save_dir"]) / f"kong-endpoint-step={trainer.global_step}.ckpt"
    on_grid = Path(config["save_dir"]) / f"kong-step={trainer.global_step}.ckpt"
    if config.get("ckpt_every_n_steps") and on_grid.exists():
        print(f"endpoint already on the checkpoint grid: {on_grid}")
    else:
        trainer.save_checkpoint(endpoint)
        print(f"saved endpoint checkpoint: {endpoint}")
    