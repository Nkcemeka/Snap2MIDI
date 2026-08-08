"""The full 20-epoch MAESTRO run at the paper's settings, with the speedups.

This is run_hft_trial.py promoted from a timing trial to the real run. The trial
is the starting point rather than run_hft_paper.py because the trial is the
configuration that has actually been executed end to end on this cluster --
2 ranks, fast attention, torch.compile, torchrun -- and it ran smoothly. Every
training hyperparameter here is run_hft_paper.py's, so the model being trained
is the paper's; only the machinery around it is the trial's.

Measured, not projected: 7.5 days for 20 epochs.

That number is the reason this file exists in the shape it does. res_gpu caps a
job at 7 days, so 7.5 does NOT fit in one job -- it misses by half a day, which
is the worst possible margin because it looks like it fits until it doesn't.
The run is therefore a chain, and the two settings that make a chain work are
the two this file changes from the trial:

    resume_path="last"   the trial sets None, deliberately, because it exists to
                         produce a from-zero curve. Inheriting that here would
                         make job 2 silently restart from step 0 and the chain
                         would never finish -- with no error anywhere, just a
                         loss curve that keeps starting over.
    max_steps            the trial stops at 141,000 to bound its cost. Removed,
                         so the run stops on epochs=20 instead.

and two that keep the chain's output readable as one run:

    save_dir             its own directory. Sharing the trial's would resume
                         from the trial's last.ckpt -- a half-epoch model from a
                         from-scratch run -- and sharing the paper run's would
                         hand a differently-configured chain its checkpoints.
    logger_version       fixed, so the jobs in the chain write one tensorboard
                         curve instead of one per job.

Checkpoint selection is the paper's, unchanged
----------------------------------------------
Verified against github.com/sony/hFT-Transformer, `training/m_training.py`:
validation runs once per division, and the best model is whichever division has
the lowest validation loss --

    if best_loss_valid > epoch_loss_valid:
        best_loss_valid = epoch_loss_valid
        best_epoch = epoch; best_div = div

with no F1 and no mir_eval anywhere in that decision. Their released MAESTRO
model is `model_016_003.pkl`, i.e. epoch 16 of 20, division 3 of 4.

val_check_interval=0.25 reproduces their 4-divisions-per-epoch cadence, the
checkpoint callback monitors `valid_total_loss`, and save_top_k=-1 keeps all
80 candidates rather than pruning during the run. So selecting the minimum
`valid_total_loss` afterwards is exactly the authors' rule. Ranking on decoded
note F1 instead is better statistics and is what CHECKPOINT_VALIDATION.md argues
for elsewhere -- but it is a declared *deviation* from the paper, not a
correction, and keeping every candidate is what leaves both options open.

Before the first job: `save_dir` must not already contain last.ckpt, or this
resumes from whatever wrote it and "finishes" having trained nothing.
"""

import torch

import snap2midi as s2m

# Fidelity, not speed. Sony pin torch==1.10.1, and every torch before 1.12
# enabled TF32 matmul by default on Ampere -- so their A100 run used it and
# torch 2.x would not. Omitting this departs from the paper rather than matching
# it. Also the control the fast-attention change was measured against
# (scripts/hft_variant_ab.py --mode grad).
torch.set_float32_matmul_precision("high")

s2m.trainer.Trainer().train_hft(
    base_path="/data/upf105/resh000979/hft_maestro",

    # DIFFERS from the trial: its own directory, not save_dir/hft_trial and not
    # save_dir/hft_paper. See the module docstring -- either would be resumed
    # from without complaint.
    save_dir="/data/upf105/resh000979/save_dir/hft_fast",
    logger_name="tensorboard",
    # DIFFERS from the trial: fixed and its own. Every job in the chain reopens
    # this same version, so tensorboard shows one curve rather than one per job.
    logger_version="hft_fast",

    num_workers=16,

    # ---- the speedups, exactly as the trial ran them ----
    #
    # 4 per device on 2 devices, where run_hft_paper.py does 8 on 1. Effective
    # batch is 8 either way and so is the optimizer step count: the loss terms
    # reduce with mean() over equal element counts per rank and DDP averages
    # them. Verified rather than assumed -- scripts/hft_ddp_equivalence.py
    # measures the 2x4 gradient against the 1x8 gradient and finds them equal to
    # float32 accumulation noise (and to ~6e-14 in float64, which is what proves
    # the float32 residue is rounding and not a reduction bug). Sony's
    # EXE-TRAINING-MAESTRO.sh uses -batch 8, so effective 8 is the paper value;
    # 8 per device would be 16 effective and a real hyperparameter change.
    batch_size=4,
    devices=2,
    strategy="ddp",

    # Gradient perturbation from both, measured on a real batch against the
    # controls in the same run: fast attention 2.4%, +compile 5.0%, against
    # 4.6% for turning TF32 off -- a precision change the run already makes
    # deliberately -- and 120-140% between two different minibatches. Also
    # verified under DDP together, which neither single-factor test covers,
    # because Dynamo splits a compiled graph at gradient-bucket boundaries only
    # when both are on.
    #
    # fast_attention is numerically redundant once compile is on (identical
    # gradients either way -- Inductor already fuses the attention). Kept
    # because this is the configuration that was trialled and timed, and
    # nothing has measured what dropping it costs in memory or throughput.
    fast_attention=True,
    compile_model=True,

    # ---- everything below is run_hft_paper.py, unchanged ----
    epochs=20,
    augment=True,
    augment_asset_root="/data/upf105/resh000979/mtg_ir_datasets",
    # audiomentations peak-normalises the reverb output, which would corrupt
    # hFT's velocity targets rather than merely leak a shortcut. "rms" preserves
    # excerpt energy, as every other arm here does.
    reverb_level="rms",

    # Sony's MAESTRO run shards four ways and validates after each shard. This
    # reproduces that cadence, which matters twice over: ReduceLROnPlateau steps
    # 80 times over 20 epochs rather than 20 against a patience of 10 (so the LR
    # can actually decay), and the checkpoint candidates land at the same
    # per-division granularity the authors selected their released model from.
    val_check_interval=0.25,
    plateau_per_validation=True,

    # DIFFERS from the trial, which sets None. This is what makes the chain a
    # chain: starts fresh when save_dir has no checkpoint, resumes when it does,
    # so the same script is correct for every job in the chain.
    resume_path="last",

    # The only thing standing between a walltime kill and lost work. Requeueing
    # does not work on this cluster (see scripts/hft_fast.sbatch), so each job is
    # killed abruptly at 7 days and the next resumes from last.ckpt, losing
    # whatever came after the last rolling save. At the trial's measured rate
    # 5000 steps is ~7 min, so the whole run gives up well under half an hour
    # across its handoffs.
    ckpt_every_n_steps=5000,
)
