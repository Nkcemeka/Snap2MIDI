# Imports
from pathlib import Path
from snap2midi.models.hft.hft import *
from .hft_dataset import HFTDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from snap2midi.utils.train_utils import pl_logger
from snap2midi.utils.augmentator import Augmentator
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


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

    # Dataloaders. Shuffling and worker sharding are PyTorch's job.
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


def _init_weights_from(model, path: str) -> None:
    """Copy the weights out of a Lightning checkpoint and nothing else.

    This is what a fine-tune needs and what `resume_path` cannot give it. A
    resume restores the optimizer and ReduceLROnPlateau state too, and the
    MAESTRO run ended with its LR decayed to 1e-8 (hft_ep17_last.ckpt:
    four decays, patience already spent). Resumed, the "fine-tune" would take
    ~5M steps' worth of Adam moments and a dead learning rate and move nothing,
    with a perfectly normal-looking loss curve. Loading the state_dict alone
    leaves configure_optimizers to build a fresh Adam at config["lr"] and a
    fresh scheduler at step 0.

    strict=True on purpose: a shape or key mismatch means the config does not
    describe the checkpoint, and silently training a half-initialised model is
    the failure to avoid.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"[init_weights_from]: loaded weights from {path} "
          f"(epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}); "
          "optimizer and LR schedule start fresh")


def main(config, model_hook=None):
    base_path = config["base_path"].rstrip('/')
    if config.get("feature_source") is None:
        # Stores from the earlier extractor hold spectrograms under feature/.
        config["feature_source"] = ("audio" if Path(f"{base_path}/audio").exists()
                                    else "legacy_feature")

    # Create datasets and set seed
    pl.seed_everything(config["seed"], workers=True)
    dm = HFTDataModule(config)

    # Load/initialize the model
    model = HFT(config)

    # Weights-only init for a fine-tune. Skipped when a resume checkpoint is
    # found, because that checkpoint already holds these weights plus the
    # fine-tune's own progress -- re-initialising would restart the run.
    ckpt_path = _resolve_resume_path(config)
    if config.get("init_weights_from"):
        if ckpt_path is None:
            _init_weights_from(model, config["init_weights_from"])
        else:
            print(f"[init_weights_from]: resuming from {ckpt_path} instead; "
                  "the pretrained weights are already inside it")

    # model_hook (freezing, adapters) runs after the strict pretrained load and
    # before configure_optimizers. It runs on a resume too: an adapter
    # checkpoint's keys include the adapters, so the model must already carry
    # them when Lightning restores it.
    if model_hook is not None:
        model_hook(model)

    save_dir = config["save_dir"]

    val_dir = "feature" if config["feature_source"] == "legacy_feature" else "audio"
    val_flag = Path(f"{base_path}/{val_dir}/val/").exists()

    # By default the five best by validation loss. With ckpt_every_n_steps,
    # every validation's checkpoint is kept (so the model can be chosen
    # afterwards on a decoded metric) and a separate rolling checkpoint plus
    # last.ckpt is written on that stride, for resume="last" after a job is
    # killed.
    every_n_steps = config.get("ckpt_every_n_steps")
    if val_flag:
        select_ckpt = ModelCheckpoint(
            monitor='valid_total_loss',
            filename='hft-{epoch:02d}-{valid_total_loss:.4f}',
            dirpath=save_dir,
            save_top_k=-1 if every_n_steps else 5,
            mode="min",
            # After validation, so the metric exists for the filename even
            # with several validations per epoch.
            save_on_train_epoch_end=False,
        )
    else:
        select_ckpt = ModelCheckpoint(
            dirpath=save_dir,
            save_top_k=-1 if every_n_steps else 1,
        )

    callbacks = [select_ckpt, EpochUpdateCallback()]
    if config.get("plateau_per_validation"):
        callbacks.append(PlateauPerValidation())
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    if every_n_steps:
        callbacks.insert(1, ModelCheckpoint(
            dirpath=save_dir,
            filename='hft-rolling-{step:08d}',
            every_n_train_steps=every_n_steps,
            monitor=None,
            save_top_k=1,
            save_last=True,
        ))

    # Fusing the encoder and decoder kernels. `.compile()` rather than
    # `torch.compile(module)`: the latter returns a wrapper, which renames every
    # key in the state_dict to `_orig_mod.*` and so writes checkpoints that no
    # uncompiled model -- including evaluate_hft -- can load, and refuses to
    # resume from any checkpoint written without it. Module.compile() patches
    # the call in place and leaves the state_dict untouched, so compiled and
    # uncompiled checkpoints stay interchangeable.
    #
    # It hooks __call__, not forward(), which works here only because HFT.forward
    # reaches its submodules as self.hft_encoder(...) rather than
    # self.hft_encoder.forward(...). Compiling HFT itself would be silently
    # skipped for exactly that reason.
    if config.get("compile_model"):
        model.hft_encoder.compile()
        model.hft_decoder.compile()

    # No cluster-environment plugin: Lightning's own detection puts TorchElastic
    # ahead of SLURM, so torchrun inside a SLURM allocation works, and on the
    # single-task path it still picks SLURMEnvironment with auto_requeue on.

    # create trainer
    trainer = pl.Trainer(max_epochs=config["epochs"], \
        deterministic=True,
        callbacks=callbacks,
        # Per *device*. Two ranks at batch 4 produce exactly the gradient one
        # rank at batch 8 produces: every loss term reduces with mean(), each
        # rank holds the same element count, and DDP averages across ranks --
        # so mean(mean(A), mean(B)) = mean(A + B). Effective batch and step
        # count are unchanged, which is why this costs no fidelity.
        devices=config.get("devices", "auto"),
        strategy=config.get("strategy", "auto"),
        # -1 is Lightning's "no limit". Set it for a timing trial so the run
        # stops at a known step count instead of a wall clock.
        max_steps=config.get("max_steps", -1),
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
        logger=pl_logger(config["logger_name"], project_name=config["experiment_name"],
                         version=config.get("logger_version")))

    trainer.fit(model, dm, ckpt_path=ckpt_path)
