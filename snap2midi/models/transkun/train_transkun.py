# Imports
from pathlib import Path
from torch.utils.data import DataLoader
from .transkun_dataset import TranskunDataset
import pytorch_lightning as pl
from snap2midi.utils.train_utils import pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
import moduleconf
from .utilities import collate_fn_batching

class EpochUpdateCallback(pl.Callback):

    def on_train_epoch_start(
        self,
        trainer,
        pl_module
    ):
        print(f"Previous epoch in datamodule: {trainer.datamodule.current_epoch}")
        trainer.datamodule.current_epoch = (
            trainer.current_epoch
        )
        print(f"Current epoch in datamodule: {trainer.datamodule.current_epoch}")

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
            seed=self.config["seed"]+(100*self.current_epoch),
            augmentator=self.config["augmentator"]
        )

        self.val_dataset = TranskunDataset(
            f"{base_path}/val/",
            self.config["sample_rate"],
            self.config["hopSizeInSecond"],
            self.config["chunkSizeInSecond"],
            audioNormalize=self.config["audioNormalize"],
            notesStrictlyContained=self.config["notesStrictlyContained"],
            ditheringFrames=self.config["ditheringFrames"],
            seed=self.config["seed"]+(100*self.current_epoch),
            augmentator=self.config["augmentator"]
        )

    # Below are methods for setting up the dataloaders
    def train_dataloader(self):
        # shuffle is False, because we already shuffle on build_chunks in the
        # dataset class
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], \
                prefetch_factor=4,
                num_workers=self.config["num_workers"], shuffle=True, drop_last=True, collate_fn=collate_fn_batching)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], \
            prefetch_factor=4, \
            num_workers=self.config["num_workers"], shuffle=True, collate_fn=collate_fn_batching)
    

def main(config):
    # Load/initialize the model
    # obtain the Model Module
    current_file_path = str(Path(__file__).parent)
    confManager = moduleconf.parseFromFile(f"{current_file_path}/conf.json")
    Transkun = confManager["Model"].module.Transkun
    conf = confManager["Model"].config
    model = Transkun(conf)

    # update config
    config["hopSizeInSecond"] = conf.segmentHopSizeInSecond
    config["chunkSizeInSecond"] = conf.segmentSizeInSecond
    config["audioNormalize"] = True
    config["notesStrictlyContained"] = False
    config["augmentator"] = None
    config["ditheringFrames"] = True

    # Create datamodule
    dm = TranskunDataModule(config)

    # create checkpoint callback
    base_path = config["base_path"].rstrip('/')
    val_flag = Path(f"{base_path}/val/").exists()
    if val_flag:
        checkpoint_callback = ModelCheckpoint(
            monitor='val/f1',
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
    