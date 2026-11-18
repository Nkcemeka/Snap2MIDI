#!/usr/bin/env python
"""Fine-tune a pretrained Snap2MIDI model on GAPS under one adaptation regime.

The regimes are those of regimes.py: full (every weight), probe (the final
projection of each supervised output), lora (rank-8 adapters, alpha 16, on
every trunk matrix, readouts trained) and lora_temporal (the same adapters on
the temporal block only). Each model keeps its own training recipe, without
augmentation, from its pretrained checkpoint; the regimes differ only in which
weights train. Learning rates follow one rule: the full rate is 1e-5 for Kong
(Riley et al.) and one tenth of the pretraining rate for the others; probe and
lora use ten times the full rate. --full-rate trains lora at the full rate
(the supplementary 1x-rate control).

RUNS below holds each run's budget. Checkpoints are kept at fixed
intervals for selection on GAPS validation (experiments/select/pick.py).

LoRA checkpoints are merged into plain ones under <run>/merged/ when training
ends; those are the files the evaluation reads. TransKun is fine-tuned from
its own repository: experiments/transkun/README.md.

Usage:
    python experiments/adapt/adapt.py --model oafv2 --regime lora \
        --init <pretrained>/oafv2.ckpt
    python experiments/adapt/adapt.py --model kong --regime lora --full-rate \
        --init <pretrained>/kong.ckpt
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import regimes as fr  # noqa: E402

MODELS = ("oaf", "oafv2", "hpp", "kong", "hft")
FULL_LR = {"oaf": 6e-5, "oafv2": 6e-5, "hpp": 6e-5, "kong": 1e-5, "hft": 1e-5}
TEN_X_LR = {"oaf": 6e-4, "oafv2": 6e-4, "hpp": 6e-4, "kong": 1e-4, "hft": 1e-4}   # probe, lora

# Budget of every run: steps (hft: epochs). One budget per model; Kong LoRA
# carries the one extension of the rule in pick.py --extension (100,000 ->
# 150,000), and each lora_fullrate control follows its model's LoRA budget.
RUNS = {
    ("oaf", "full"): 40_000, ("oaf", "probe"): 40_000, ("oaf", "lora"): 40_000,
    ("oaf", "lora_temporal"): 40_000, ("oaf", "lora_fullrate"): 40_000,
    ("oafv2", "full"): 40_000, ("oafv2", "probe"): 40_000, ("oafv2", "lora"): 40_000,
    ("oafv2", "lora_temporal"): 40_000, ("oafv2", "lora_fullrate"): 40_000,
    ("hpp", "full"): 60_000, ("hpp", "probe"): 60_000, ("hpp", "lora"): 60_000,
    ("hpp", "lora_temporal"): 60_000, ("hpp", "lora_fullrate"): 60_000,
    ("kong", "full"): 100_000, ("kong", "probe"): 100_000, ("kong", "lora"): 150_000,
    ("kong", "lora_temporal"): 100_000, ("kong", "lora_fullrate"): 150_000,
    ("hft", "full"): 8, ("hft", "probe"): 8, ("hft", "lora"): 8,
    ("hft", "lora_temporal"): 8, ("hft", "lora_fullrate"): 8,
}

LORA = dict(rank=8, alpha=16.0)
WORKERS = 8


def prepare(model, name, regime):
    """Freeze or inject adapters, then hold frozen BatchNorm statistics."""
    report = fr.apply(model, regime, fr.ROLES[name], **LORA)
    fr.harden_frozen_norms(model)
    return report


def settings(name, regime, full_rate, cfg, init, resume):
    """Everything the model's Trainer method is called with."""
    lr = FULL_LR[name] if regime == "full" or full_rate else TEN_X_LR[name]
    budget = RUNS[(name, "lora_fullrate" if full_rate else regime)]
    common = dict(base_path=str(cfg["stores"] / f"{name}_gaps_ft"),
                  save_dir=str(run_dir(name, regime, full_rate, cfg)),
                  init_weights_from=str(init), lr=lr, augment=False,
                  num_workers=WORKERS, logger_name="csv")
    if name == "oaf":
        return dict(common, resume_path=resume, iterations=budget, batch_size=8,
                    feature="mel", sample_rate=16000, max_frame_secs=20.0, n_mels=229,
                    mel_n_fft=2048, hop_length=512, frame_rate=31.25, in_features=229,
                    out_features=88, temporal_sizes=[3, 3, 3], freq_sizes=[3, 3, 3],
                    out_channels=[32, 32, 64], pool_sizes=[1, 2, 2],
                    dropout_probs=[0, 0.25, 0.25], dropout_fc=0.5, fc_size=512,
                    onset_lstm_units=128, combined_lstm_units=128,
                    learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000,
                    clip_gradient_norm=3, seed=42, ckpt_every_n_steps=1000,
                    val_every_n_steps=500)
    if name == "oafv2":
        return dict(common, resume_path=resume, iterations=budget, batch_size=8,
                    sequence_length=327680, sample_rate=16000, n_fft=2048, n_mels=229,
                    htk=True, fmin=30, fmax=8000, hop_length=512, model_complexity=48,
                    learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000,
                    clip_gradient_norm=3, seed=42, ckpt_every_n_steps=1000,
                    val_every_n_steps=500)
    if name == "hpp":
        return dict(common, resume_path=resume, model_type="sp", iterations=budget,
                    batch_size=4, sequence_length=327680, sample_rate=16000,
                    hop_length=320, bins_per_semitone=4, learning_rate_decay_rate=0.98,
                    learning_rate_decay_steps=10000, clip_gradient_norm=3, seed=42,
                    checkpoint_every_n_steps=2000, early_stopping=False)
    if name == "kong":
        return dict(common, resume_path=resume, iterations=budget, batch_size=4, lr=lr,
                    learning_rate_decay_rate=0.9, learning_rate_decay_steps=10000,
                    factors=[16, 32, 32], frame_rate=100, cmp=48, momentum=0.01,
                    onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.3,
                    pedal_offset_threshold=0.3, clip_gradient_norm=3.0, num_workers=WORKERS,
                    val_steps=2000, log_steps=100, num_nodes=1, logger_name="csv",
                    seed=1234, experiment_name="KongRegimeGAPS", ckpt_every_n_steps=5000)
    if name == "hft":
        return dict(common, resume_path="last", epochs=budget, batch_size=8, devices=1,
                    train_velocity=True, val_check_interval=0.25,
                    plateau_per_validation=True, plateau_patience=2,
                    fast_attention=True, compile_model=not regime.startswith("lora"),
                    ckpt_every_n_steps=2000, max_steps=-1)
    raise ValueError(name)


def run_dir(name, regime, full_rate, cfg):
    return cfg["runs"] / f"{name}_gaps_{regime}{'_fullrate' if full_rate else ''}"


def train_library(name, regime, kwargs):
    """The library's training loop; the regime is applied by its model_hook,
    right after the pretrained weights are loaded (on a resume too, before
    Lightning restores the checkpoint)."""
    import pytorch_lightning as pl
    import snap2midi as s2m
    if name == "hft":
        import torch
        torch.set_float32_matmul_precision("high")
    else:
        pl.seed_everything(kwargs["seed"], workers=True)   # hFT seeds itself from its config
    getattr(s2m.trainer.Trainer(), f"train_{name}")(
        model_hook=lambda model: prepare(model, name, regime), **kwargs)


def merge(save_dir, model):
    """Every LoRA checkpoint of the run, merged into a plain one under merged/."""
    pattern = {"hft": "hft-epoch=*.ckpt"}.get(model, f"{model}-step=*.ckpt")
    out = Path(save_dir) / "merged"
    out.mkdir(exist_ok=True)
    for f in sorted(Path(save_dir).glob(pattern)):
        dst = out / f.name
        if not dst.exists():
            fr.merge_checkpoint_file(str(f), str(dst) + ".part", LORA["alpha"])
            Path(str(dst) + ".part").replace(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--regime", required=True, choices=fr.REGIMES)
    ap.add_argument("--init", required=True, type=Path, help="The pretrained checkpoint.")
    ap.add_argument("--full-rate", action="store_true",
                    help="lora only: train at the full fine-tuning rate.")
    ap.add_argument("--resume", type=Path, default=None,
                    help="This run's last.ckpt, to continue it (hft resumes by itself).")
    args = ap.parse_args()
    if args.full_rate and args.regime != "lora":
        raise SystemExit("--full-rate applies to --regime lora only")
    cfg = config.load(args.config)
    store = cfg["stores"] / f"{args.model}_gaps_ft"
    if not (store / ".complete").exists():
        raise SystemExit(f"{store} has no .complete marker: extract it first with "
                         f"experiments/data/extract.py --model {args.model} --dataset gaps_ft")
    if not args.init.is_file():
        raise SystemExit(f"no such checkpoint: {args.init}")
    resume = None if args.resume is None else str(args.resume)

    kwargs = settings(args.model, args.regime, args.full_rate, cfg, args.init, resume)
    train_library(args.model, args.regime, kwargs)
    save_dir = kwargs["save_dir"]
    if args.regime.startswith("lora"):
        merge(save_dir, args.model)


if __name__ == "__main__":
    main()
