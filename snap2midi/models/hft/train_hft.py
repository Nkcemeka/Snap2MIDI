# Imports
import warnings
from pathlib import Path
from snap2midi.models.hft.hft import *
from .hft_dataset import HFTDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment


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


class PlateauPerValidation(pl.Callback):
    """Step ReduceLROnPlateau once per validation instead of once per epoch.

    Sony's loop validates and calls scheduler.step() together, so their MAESTRO
    run -- which shards the training set four ways -- gives the scheduler 80
    chances to fire across 20 epochs. Lightning instead steps a scheduler whose
    configure_optimizers declared interval="epoch" exactly once an epoch, on
    whatever the most recent validation value was, however often validation
    actually ran. Twenty checks against ReduceLROnPlateau's default patience of
    10 means the streak needed to trigger a decay is half the run, so the LR
    stays at its initial value throughout where the paper's decayed.

    This is a MAESTRO-only correction. Their MAPS run uses n_div_train=1 over
    50 epochs, which is one validation per epoch -- the same cadence this code
    already had, which is why the MAPS results needed no such fix.

    Pairs with val_check_interval < 1.0 and does nothing useful without it: at
    1.0 the frequency below is the epoch length, so the scheduler still steps
    once an epoch and only the bookkeeping differs.

    The frequency cannot be computed in configure_optimizers, where
    trainer.num_training_batches is still inf because the training data has not
    been set up yet. on_train_start is the first hook where it is known.
    """

    def on_train_start(self, trainer, pl_module):
        if not trainer.lr_scheduler_configs:
            raise RuntimeError(
                "plateau_per_validation is set, but the module configured no "
                "lr scheduler for it to retime.")

        # An int val_check_interval is already a batch count; a float is a
        # fraction of the epoch. Reading it wrong would silently retime the
        # scheduler to something unrelated to when validation runs.
        interval = trainer.val_check_interval
        frequency = (interval if isinstance(interval, int)
                     else int(trainer.num_training_batches * interval))

        config = trainer.lr_scheduler_configs[0]
        config.interval = "step"
        config.frequency = max(1, frequency)


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

def _resolve_resume_path(config) -> str | None:
    """Where to resume from, or None to start from scratch.

    `"last"` is the value a chained cluster run wants: the first job in the
    chain finds no checkpoint and starts fresh, every later job picks up the
    rolling one without anybody editing a filename between submissions. An
    explicit path still asserts, because asking for a specific checkpoint and
    silently getting a different one is worse than stopping.

    Returning None is not the same as "no resume" under SLURM: Lightning's
    checkpoint connector looks for its own `hpc_ckpt_*.ckpt` in
    default_root_dir when ckpt_path is None and SLURMEnvironment is detected,
    which is how an auto-requeued job comes back. last.ckpt is the fallback for
    the deaths that never get to run a signal handler -- node failure, OOM kill.
    """
    resume_path = config.get("resume_path")

    if resume_path is None:
        return None

    if resume_path == "last":
        last = Path(config["save_dir"]) / "last.ckpt"
        return str(last) if last.exists() else None

    assert Path(resume_path).exists(), \
        f"[resume_path]: {resume_path} does not exist."
    return resume_path


def main(config):
    # Create datasets and set seed
    pl.seed_everything(config["seed"], workers=True)
    dm = HFTDataModule(config)

    # Load/initialize the model
    model = HFT(config)

    save_dir = config["save_dir"]

    # CSUC's /scratch is wiped 7 days after the job that wrote it ends, and a
    # 20-epoch run is chained across weeks of jobs. Reading the *dataset* from
    # scratch is right -- it is the low-latency tier and the store is a copy --
    # but writing the only copy of the checkpoints there loses the run.
    if "/scratch/" in str(save_dir):
        warnings.warn(
            f"save_dir={save_dir} is on scratch, which is purged some days "
            f"after the job ends. Checkpoints are the one artefact of the run "
            f"that cannot be regenerated -- point save_dir at project storage "
            f"(/data/...) instead.",
            stacklevel=2)

    # Two checkpoint callbacks, because selection and restart want opposite
    # settings and one callback cannot be both.
    base_path = config["base_path"].rstrip('/')
    val_dir = "feature" if config.get("feature_source") == "legacy_feature" else "audio"
    val_flag = Path(f"{base_path}/{val_dir}/val/").exists()

    # (1) Selection. save_top_k=-1 keeps every epoch. The previous value of 5
    # deleted 15 of 20 candidates *during* the run, ranked on valid_total_loss
    # -- a sum of eight terms dominated by a 128-class velocity CE, which is not
    # the note F1 the paper selects on and not the number being reported. At
    # 64 MB a checkpoint, keeping all 20 costs 1.3 GB, so the ranking stays a
    # decision that can be made afterwards, on the right metric.
    if val_flag:
        select_ckpt = ModelCheckpoint(
            monitor='valid_total_loss',
            filename='hft-{epoch:02d}-{valid_total_loss:.4f}',
            dirpath=save_dir,
            save_top_k=-1,
            mode="min",
            # Save after validation rather than at train-epoch end, so the
            # monitored metric exists when the filename is interpolated.
            save_on_train_epoch_end=False,
        )
    else:
        select_ckpt = ModelCheckpoint(
            dirpath=save_dir,
            save_top_k=-1,
        )

    # (2) Restart. Fires on a step count, so it cannot monitor anything: there
    # is no validation metric at step 40000. monitor=None with save_top_k=1 is
    # "latest wins" -- a single rolling file rather than one per interval -- and
    # save_last mirrors it to last.ckpt, which is what _resolve_resume_path
    # looks for. Only this callback sets save_last; two callbacks writing
    # last.ckpt would each overwrite the other's.
    #
    # The interval defaults here rather than in Trainer.train_hft because it is
    # a property of surviving a scheduler, not a training hyperparameter, and
    # nothing about the run changes if it moves. 2000 steps is a rolling *file*,
    # so disk does not grow; the cost is one 64 MB write every ~10 min at a
    # plausible step time, which is nothing against losing a job's work. Pass
    # ckpt_every_n_steps in the config to size it properly -- roughly an eighth
    # of the steps a job fits, (walltime / step_time) / 8, with step_time from
    # scripts/hft_throughput_trial.py -- or None to switch it off.
    callbacks = [select_ckpt, EpochUpdateCallback()]
    if config.get("plateau_per_validation"):
        callbacks.append(PlateauPerValidation())
    every_n_steps = config.get("ckpt_every_n_steps", 2000)
    if every_n_steps:
        callbacks.insert(1, ModelCheckpoint(
            dirpath=save_dir,
            filename='hft-rolling-{step:08d}',
            every_n_train_steps=every_n_steps,
            monitor=None,
            save_top_k=1,
            save_last=True,
        ))

    # Under SLURM, catch the signal the scheduler sends before the walltime kill
    # (sbatch needs --signal=USR1@<seconds> --requeue for it to arrive), save,
    # and resubmit. Without this, a job killed mid-epoch loses everything since
    # the last rolling checkpoint; with it, it loses nothing. detect() keeps
    # this inert off the cluster.
    plugins = [SLURMEnvironment(auto_requeue=True)] if SLURMEnvironment.detect() else []

    # create trainer
    trainer = pl.Trainer(max_epochs=config["epochs"], \
        deterministic=True,
        callbacks=callbacks,
        plugins=plugins,
        # Lightning writes its auto-requeue checkpoints to default_root_dir.
        # Left at the default it would be the working directory, i.e. not
        # necessarily the storage tier save_dir was chosen to be on.
        default_root_dir=save_dir,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        # How often validation runs, and therefore how many checkpoint
        # candidates the run leaves behind. It does not change the LR
        # scheduler's cadence -- that is pinned to "epoch" in
        # HFT.configure_optimizers; see Trainer.train_hft's docstring. 1.0
        # keeps the historical once-an-epoch behaviour.
        val_check_interval=config.get("val_check_interval", 1.0),
        num_nodes=config["num_nodes"],
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"]))

    trainer.fit(model, dm, ckpt_path=_resolve_resume_path(config))
