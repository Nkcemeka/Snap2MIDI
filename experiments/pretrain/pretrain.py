#!/usr/bin/env python
"""Pretrain one of the five Snap2MIDI models on MAESTRO v3, with augmentation.

Each model's settings below are those of the run that produced its released
checkpoint (checkpoints.md); the training settings stored inside each
checkpoint are equal to what this script passes. Every model is trained on
its MAESTRO feature store (experiments/data/extract.py --dataset maestro)
with the same augmentation pipeline (Edwards et al.), reverb at RMS level.

  oaf    the 2018 sizes; Magenta's 0.98 per 10k LR decay; 670,000 steps
         (the paper's 50,000, extended to OAFv2's budget); every 2,000 steps kept
  oafv2  Hawthorne et al. 2019; 670,000 steps; a checkpoint every 5,000
  hpp    HPPNet-sp; 400,000 global steps (two optimizer steps per batch, so
         200,000 batches); a checkpoint every 8,000
  kong   Kong et al.; 200,000 steps; the endpoint is released
  hft    hFT-Transformer at Sony's settings: up to 20 epochs, validation four
         times per epoch with the plateau scheduler stepped on each; effective
         batch 8 as 2 GPUs x 4 with fast attention, torch.compile and TF32
         (--devices 1: 1 GPU x 8). The released model is the run's last
         checkpoint, at epoch 17.

The released checkpoints are the last step of each run (hft: see above).
TransKun is pretrained from its own repository: experiments/transkun/README.md.

Usage:
    python experiments/pretrain/pretrain.py --model oafv2
    python experiments/pretrain/pretrain.py --model oafv2 --resume <runs>/oafv2_pretrain/last.ckpt
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402

AUGMENTATION = dict(augment=True, reverb_level="rms")

SETTINGS = {
    "oaf": dict(
        batch_size=8, iterations=670_000, lr=0.0006, clip_gradient_norm=3,
        feature="mel", sample_rate=16000, max_frame_secs=20.0, n_mels=229,
        mel_n_fft=2048, hop_length=512, frame_rate=31.25,
        in_features=229, out_features=88, temporal_sizes=[3, 3, 3], freq_sizes=[3, 3, 3],
        out_channels=[32, 32, 64], pool_sizes=[1, 2, 2], dropout_probs=[0, 0.25, 0.25],
        dropout_fc=0.5, fc_size=512, onset_lstm_units=128, combined_lstm_units=128,
        learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000,
        seed=1234, ckpt_every_n_steps=2000, val_every_n_steps=2000),
    "oafv2": dict(
        batch_size=8, iterations=670_000, lr=0.0006, clip_gradient_norm=3,
        sequence_length=327680, sample_rate=16000, n_fft=2048, n_mels=229, htk=True,
        fmin=30, fmax=8000, hop_length=512, pad_mode="reflect", center=True, window="hann",
        in_features=229, out_features=88, model_complexity=48,
        learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000,
        seed=42, ckpt_every_n_steps=5000, val_every_n_steps=500),
    "hpp": dict(
        model_type="sp", batch_size=4, lr=0.0006, sequence_length=327680,
        sample_rate=16000, hop_length=320, bins_per_semitone=4,
        learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000,
        clip_gradient_norm=3, seed=42, iterations=400_000,
        checkpoint_every_n_steps=8_000, early_stopping=False),
    "kong": dict(
        batch_size=12, iterations=200_000, lr=5e-4,
        learning_rate_decay_rate=0.9, learning_rate_decay_steps=10_000,
        onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.3,
        val_steps=5_000, log_steps=200, seed=1234),
    "hft": dict(
        epochs=20, val_check_interval=0.25, plateau_per_validation=True,
        batch_size=4, devices=2, strategy="ddp", fast_attention=True, compile_model=True,
        ckpt_every_n_steps=5000),
}

WORKERS = {"oaf": 10, "oafv2": 10, "hpp": 4, "kong": 8, "hft": 16}


def kwargs(model, cfg, devices=None, resume=None):
    """Everything the model's Trainer method is called with."""
    settings = dict(SETTINGS[model])
    if model == "hft" and devices == 1:
        settings.update(batch_size=8)
        del settings["devices"], settings["strategy"]
    return dict(
        base_path=str(cfg["stores"] / f"{model}_maestro"),
        save_dir=str(cfg["runs"] / f"{model}_pretrain"),
        augment_asset_root=str(cfg["augmentation"]),
        num_workers=WORKERS[model], logger_name="csv",
        resume_path=None if resume is None else str(resume),
        **AUGMENTATION, **settings)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--model", required=True, choices=list(SETTINGS))
    ap.add_argument("--resume", type=Path, default=None,
                    help="A checkpoint of this run to continue from.")
    ap.add_argument("--devices", type=int, choices=[1, 2], default=None,
                    help="hft only: 1 trains on one GPU at batch 8 (default: 2 x 4).")
    args = ap.parse_args()
    if args.devices is not None and args.model != "hft":
        raise SystemExit("--devices applies to hft only")
    cfg = config.load(args.config)
    store = cfg["stores"] / f"{args.model}_maestro"
    if not (store / ".complete").exists():
        raise SystemExit(f"{store} has no .complete marker: extract it first with "
                         f"experiments/data/extract.py --model {args.model} --dataset maestro")

    import snap2midi as s2m
    if args.model == "hft":
        import torch
        torch.set_float32_matmul_precision("high")   # TF32, as Sony's torch 1.10 run had
    getattr(s2m.trainer.Trainer(), f"train_{args.model}")(
        **kwargs(args.model, cfg, args.devices, args.resume))


if __name__ == "__main__":
    main()
