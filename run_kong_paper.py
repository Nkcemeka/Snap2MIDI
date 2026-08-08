"""Kong on MAESTRO with augmentation, at the configuration Kong et al. published.

Kong et al., "High-resolution Piano Transcription with Pedals by Regressing
Onset and Offset Times" (arXiv:2010.01815), §IV-C: "We use a batch size 12, and
an Adam optimizer with a learning rate of 0.0005 for training. The learning rate
is reduced by a factor of 0.9 every 10 k iterations in training. Systems are
trained for 200 k iterations." Inference thresholds are 0.3 throughout.

Every one of those is reproduced here. The repo's train_kong defaults already
matched the paper on iterations, learning rate, the 0.9-per-10k decay and the
thresholds; batch size was the sole deviation, at 4 against the paper's 12.
Measured on this GPU, batch 12 peaks at 25.5 GB of 31.4 and runs 534 ms/step, so
200k iterations is about 30 hours.

This is a different run from run_kong_experiment.py, not a replacement. That one
is an augmentation A/B at 55,000 steps and batch 8 -- a budget chosen to fit two
arms into a weekend -- and its numbers stay valid on their own terms. This is the
paper's full budget, so its absolute F1 is comparable to Kong's published 96.72
note F1 in a way the shorter run's is not.

Two things are ours rather than the paper's, because the paper has neither:

  * Augmentation. Kong et al. train clean. The pipeline is Edwards', unchanged,
    which is the point of holding it identical across every architecture here.
  * reverb_level="rms". Carried over from run_kong_experiment.py, where the
    reasoning is set out at length: audiomentations peak-normalises the reverb
    output to 0.5 regardless of input, which on real items raises the mean
    log-mel by 3.5 nats and makes "did reverb fire?" predictable from loudness
    alone at AUC 0.93. "rms" preserves the excerpt's energy, as Kaldi's
    wav-reverberate does by default. Keeping it matches the existing arms.

There is no clean baseline at this configuration yet. The augmented result is
only interpretable against one, so a matching run with augment=False is the
natural companion -- another ~30 hours on the same GPU.

Usage:
    python run_kong_paper.py
    python run_kong_paper.py --resume save_dir/kong_paper_aug/last.ckpt
    bash scripts/kong_status.sh          # progress, losses, checkpoints

--resume continues an interrupted or truncated run from a checkpoint's own
optimizer moments, LR schedule and step counter, so the schedule stays
continuous across the join. Every other setting below is unchanged, which is
the point: a resumed tail is only the same run if it is the same run.
"""

import argparse

import snap2midi as s2m

MAESTRO = "./data/kong_maestro/"
ASSETS = "/home/simone-chieppa/Desktop/mtg_ir_datasets"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to continue from. Omit to start fresh.")
    args = ap.parse_args()

    s2m.trainer.Trainer().train_kong(
        resume_path=args.resume,
        base_path=MAESTRO,
        # --- Kong et al. §IV-C, verbatim ---
        batch_size=12,
        iterations=200_000,
        lr=5e-4,
        learning_rate_decay_rate=0.9,
        learning_rate_decay_steps=10_000,
        onset_threshold=0.3,
        offset_threshold=0.3,
        frame_threshold=0.3,
        # --- ours ---
        # 8 workers supply ~500 augmented items/s against the 22 the GPU eats at
        # batch 12, so the loader is not the constraint at any point.
        num_workers=8,
        val_steps=5_000,
        log_steps=200,
        seed=1234,
        logger_name="wandb",
        experiment_name="KongPaperAugmented",
        save_dir="./save_dir/kong_paper_aug",
        augment=True,
        augment_asset_root=ASSETS,
        reverb_level="rms",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
