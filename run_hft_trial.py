"""Timing trial: the paper run, from scratch, with the speedups, stopped early.

Two questions, one job:

  1. What is the real h/epoch on 2 H100s with fast attention and torch.compile?
     The projections say ~7.6 h/epoch against the measured 22.6, i.e. 20 epochs
     in ~6 days rather than 19. That number comes from a 5090 and arithmetic;
     this replaces it with the cluster's own.

  2. Does it still train the same? The 12 h trial (job 3100103) already walked
     this exact path from step 0 with the unmodified code, so its loss curve is
     the control. This run writes to its own logger version, and the two are
     overlaid afterwards.

Everything not related to speed is copied from run_hft_paper.py verbatim, so a
difference between the curves is the speedup and nothing else. The two
deliberate differences are marked DIFFERS below.

Not comparable step-for-step, and it should not be read that way. Two ranks use
a DistributedSampler where one rank used a RandomSampler, so the excerpts
arrive in a different order and any single step's loss differs. What must match
is the *trend*: the same descent, over the same steps, to roughly the same
place. A curve that sits visibly above the control, or that spikes, is the
signal to stop.

Sizing MAX_STEPS. An epoch is 2,245,273 excerpts at effective batch 8 =
280,659 steps, and val_check_interval=0.25 puts validation at 70,164 / 140,328
/ 210,492. Those are the only steps that produce a valid_total_loss and a
checkpoint, so the step cap wants to sit *just past* one of them -- a cap of
140,000 stops 328 steps short of the second and quietly yields half the
comparison points it looks like it should.

  141,000  ~4 h, two validation points. Reaches step 140,328, which is exactly
           where the 12 h control run stopped, so both of its validation
           losses have a counterpart. This is the default.
   71,000  ~2 h, one validation point. Half the GPU-hours; enough to confirm
           the throughput and see the early descent, not enough to see whether
           the two curves stay together.

Slurm bills elapsed time, not requested time, so the walltime in the sbatch is
not what costs anything -- this is.
"""

import os

import torch

import snap2midi as s2m

# Same argument as run_hft_paper.py: Sony pin torch==1.10.1, which enabled TF32
# by default on Ampere, so the paper's A100 run used it. Keeping it is fidelity,
# not a speedup -- and it is also the control the fast-attention change was
# measured against (scripts/hft_variant_ab.py --mode grad).
torch.set_float32_matmul_precision("high")

MAX_STEPS = int(os.environ.get("HFT_TRIAL_STEPS", 141_000))

s2m.trainer.Trainer().train_hft(
    base_path="/data/upf105/resh000979/hft_maestro",

    # DIFFERS: its own save_dir and logger version. The real run's chain resumes
    # from last.ckpt in save_dir/hft_paper -- a trial writing there would hand
    # the 19-day run a checkpoint from a different configuration, and it would
    # resume from it without complaint.
    save_dir="/data/upf105/resh000979/save_dir/hft_trial",
    logger_name="tensorboard",
    logger_version="hft_trial",

    num_workers=16,

    # DIFFERS: 4 per device on 2 devices, where the paper run does 8 on 1.
    # Effective batch is 8 either way and so is the optimizer step count -- the
    # loss terms reduce with mean() over equal element counts per rank and DDP
    # averages them, so mean(mean(A), mean(B)) = mean(A + B) exactly. Sony's
    # EXE-TRAINING-MAESTRO.sh uses -batch 8, so effective 8 is the paper value;
    # 8 per device would be 16 effective and a real hyperparameter change.
    batch_size=4,
    devices=2,
    strategy="ddp",

    fast_attention=True,
    compile_model=True,

    # Stop on a step count rather than the walltime, so the run covers the same
    # span of the control curve however fast it turns out to be.
    max_steps=MAX_STEPS,

    # ---- everything below is run_hft_paper.py, unchanged ----
    epochs=20,
    augment=True,
    augment_asset_root="/data/upf105/resh000979/mtg_ir_datasets",
    reverb_level="rms",
    val_check_interval=0.25,
    plateau_per_validation=True,
    ckpt_every_n_steps=5000,

    # From scratch, deliberately. Resuming would start the curve at 156,000
    # where the control has already diverged from its own initial conditions,
    # and the whole point is to compare against a from-zero descent.
    resume_path=None,
)
