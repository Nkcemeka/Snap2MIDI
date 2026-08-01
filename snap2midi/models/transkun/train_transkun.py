# Imports
from pathlib import Path
from torch.utils.data import DataLoader
from .transkun_dataset import TranskunDataset
import pytorch_lightning as pl
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
import moduleconf
from .utilities import collate_fn_batching
import torch 

class EpochUpdateCallback(pl.Callback):
    """Mirror the epoch onto the datamodule, which re-dices chunks with it.

    Unlike the other models here, transkun does not rely on workers re-forking
    to carry the epoch: `reload_dataloaders_every_n_epochs=1` rebuilds the whole
    dataloader each epoch, and build_chunks stamps the epoch onto the dataset
    before the new workers fork. That is why persistent_workers=True is safe
    here and is not elsewhere -- persistent workers outlive the iterations of
    one loader, not the loader itself.

    The guard therefore checks transkun's actual invariant. Without the reload,
    build_chunks is never called again: the chunk dithering freezes and so does
    the augmentation draw, both silently.
    """

    def on_train_epoch_start(
        self,
        trainer,
        pl_module
    ):
        if getattr(trainer.datamodule, "augmentator", None) is not None:
            if trainer.reload_dataloaders_every_n_epochs != 1:
                raise RuntimeError(
                    "transkun augmentation needs "
                    "reload_dataloaders_every_n_epochs=1: the epoch reaches the "
                    "dataset through build_chunks, which only runs when the "
                    "dataloader is rebuilt. Without it the augmentation "
                    "silently freezes at the first epoch's draw.")
        trainer.datamodule.current_epoch = (
            trainer.current_epoch
        )


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

class TranskunDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        """
            Instantiate data module.

            Args
            -----
                config (dict): Configuration params.
        """
        super().__init__()
        self.config = config
        self.current_epoch = 0
        # Built once and shared by setup(); the callback reads it to decide
        # whether the reload guard applies.
        self.augmentator = build_augmentator(config)
    
    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        base_path = f"{self.config["base_path"].rstrip('/')}"
        self.train_dataset = TranskunDataset(
            f"{base_path}/train/",
            self.config["sample_rate"],
            self.config["hopSizeInSecond"],
            self.config["chunkSizeInSecond"],
            audioNormalize=self.config["audioNormalize"],
            notesStrictlyContained=self.config["notesStrictlyContained"],
            ditheringFrames=self.config["ditheringFrames"],
            augmentator=self.augmentator
        )

        # Never augmented: the checkpoint callback selects on the validation
        # loss, so augmenting it would make selection partly a measure of the
        # augmentation draw rather than of the model.
        self.val_dataset = TranskunDataset(
            f"{base_path}/val/",
            self.config["sample_rate"],
            self.config["hopSizeInSecond"],
            self.config["chunkSizeInSecond"],
            audioNormalize=self.config["audioNormalize"],
            notesStrictlyContained=self.config["notesStrictlyContained"],
            ditheringFrames=self.config["ditheringFrames"],
            augmentator=None
        )

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        # shuffle is False, because we already shuffle on build_chunks in the
        # dataset class
        self.train_dataset.build_chunks(self.config["seed"]+(100*self.current_epoch),
                                        epoch=self.current_epoch)
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                prefetch_factor=max(4, self.config["num_workers"]), persistent_workers=True,
                num_workers=self.config["num_workers"], shuffle=True, drop_last=True, collate_fn=collate_fn_batching)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        self.val_dataset.build_chunks(self.config["seed"]+(100*self.current_epoch),
                                      epoch=self.current_epoch)
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], \
            prefetch_factor=max(4, self.config["num_workers"]), persistent_workers=True,\
            num_workers=self.config["num_workers"], shuffle=True, collate_fn=collate_fn_batching)
    

def main(config):
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    # Load/initialize the model
    # obtain the Model Module
    current_file_path = str(Path(__file__).parent)
    confManager = moduleconf.parseFromFile(f"{current_file_path}/conf.json")
    Transkun = confManager["Model"].module.Transkun
    conf = confManager["Model"].config
    conf.freq = config["freq"]
    model = Transkun(conf)

    # update config
    config["hopSizeInSecond"] = conf.segmentHopSizeInSecond
    config["chunkSizeInSecond"] = conf.segmentSizeInSecond
    config["audioNormalize"] = True
    config["notesStrictlyContained"] = False
    config["ditheringFrames"] = True

    # Create datamodule
    dm = TranskunDataModule(config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            filename='transkun-step={step}-f1={val_f1:.4f}',
            dirpath=config["save_dir"],
            save_top_k=5,
            every_n_epochs=1,
            mode="max",
            save_last=True
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["save_dir"],
        )

    # create trainer
    profiler = PyTorchProfiler(
        filename="profiler_log",
        dirpath=None,
        group_by_input_shape=True,
        emit_nvtx=True,
    )

    trainer = pl.Trainer(max_epochs=config["epochs"], \
        devices=config["nProcess"],
        strategy="ddp" if config["nProcess"] > 1 else "auto",
        callbacks=[checkpoint_callback, EpochUpdateCallback()],
        num_sanity_val_steps=0,
        num_nodes=config["num_nodes"],
        check_val_every_n_epoch=1,
        profiler=profiler,
        reload_dataloaders_every_n_epochs=1,
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))
    
    if config["resume_path"] is None:
        trainer.fit(model, dm)
    else:
        assert Path(config["resume_path"]).exists(), \
            f"[resume_path]: {config["resume_path"]} does not exist."
        trainer.fit(model, dm, ckpt_path=config["resume_path"])
    