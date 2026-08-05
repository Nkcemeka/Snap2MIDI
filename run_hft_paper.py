"""hFT-Transformer on MAESTRO v3 at the paper's settings (Toyama et al., ISMIR 2023).

EPOCHS=1 measures one epoch, ~45 h. Set it to 20 for the full run and resubmit:
resume_path="last" means the run continues from the epoch you already paid for,
so epoch 1 is a decision point, not a throwaway.

Only batch_size and epochs differ from train_hft's defaults -- the other 20 of
the paper's 22 settings are already the defaults, which is why this is short.

The one deliberate deviation from the paper is Edwards-style augmentation, the
same one run_hpp_paper.py makes. The published hFT numbers are therefore not a
matched control; the augmented HPPNet arm is.

Gotcha when re-timing: a second run against a save_dir that already holds
last.ckpt will resume and finish almost immediately, having measured nothing.
Point save_dir somewhere fresh to measure again.
"""

import torch

import snap2midi as s2m

# Fidelity, not speed. Sony pin torch==1.10.1, and every torch before 1.12
# enabled TF32 matmul by default on Ampere -- so their A100 run used it and
# torch 2.x would not. Omitting this departs from the paper rather than matching
# it, and costs a large multiple of throughput on an H100.
torch.set_float32_matmul_precision("high")

EPOCHS = 1

s2m.trainer.Trainer().train_hft(
    base_path="/data/upf105/resh000979/hft_maestro",
    save_dir="/data/upf105/resh000979/save_dir/hft_paper",
    logger_name="tensorboard",
    # Fixed, not the default version_N. This run is a chain of ~38 walltimes and
    # every resumed job would otherwise open its own version_N, which
    # tensorboard shows as a separate run -- one curve in 38 pieces.
    logger_version="hft_paper",
    num_workers=16,

    batch_size=8,        # train_hft defaults to 4
    epochs=EPOCHS,       # paper: 20

    augment=True,
    augment_asset_root="/data/upf105/resh000979/mtg_ir_datasets",
    # Also the default, named because it is the one augmentation choice that
    # departs from Edwards: audiomentations peak-normalises the reverb output,
    # which would corrupt hFT's velocity targets rather than merely leak a
    # shortcut. "rms" preserves excerpt energy, as every other arm here does.
    reverb_level="rms",

    # Sony's MAESTRO run shards four ways and validates after each shard, so it
    # validates and steps ReduceLROnPlateau 80 times over 20 epochs. Once per
    # epoch gives 20, against a default patience of 10 -- the LR would never
    # decay. Neither flag is needed for MAPS, which Sony run undivided.
    val_check_interval=0.25,
    plateau_per_validation=True,

    # Starts fresh when there is no checkpoint, resumes when there is, so the
    # same script works for every job in the chain. One epoch outlasts any
    # walltime here, so there will be a chain.
    resume_path="last",
)
