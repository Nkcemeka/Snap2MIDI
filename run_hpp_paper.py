"""HPPNet-sp on MAESTRO v3 at the paper's settings, with augmentation added.

Wei et al. (ISMIR 2022) train HPPNet clean, so there is exactly one deliberate
deviation here: Edwards-style room-IR and background-noise augmentation on the
training excerpts. Everything else is the paper's -- batch 4, Adam at 6e-4
decaying 0.98 every 10k, gradient clipping at 3, 20.48 s excerpts at 16 kHz, a
352-bin CQT on the 50 fps grid, and the sp variant with separate acoustic models
for onset and for frame/offset/velocity.

The comparison this run answers is Table 4's, not Table 3's: train on MAESTRO,
evaluate on MAPS, where the published no-augmentation HPPNet-sp reaches frame F1
87.56 and note F1 86.63. In-domain MAESTRO test (93.15 / 97.18, Table 3) is
worth reporting too, but augmentation is expected to cost a little there.

There is no clean in-house baseline at this configuration, so the reference is
the published figure rather than a matched control. Sub-point differences are
not interpretable -- this port is not the authors' binary.
"""

import snap2midi as s2m

s2m.trainer.Trainer().train_hpp(
    base_path="data/hpp_maestro",
    model_type="sp",
    save_dir="save_dir/hpp_augmented",

    # Paper settings. Passed explicitly rather than left to the defaults so the
    # run is legible on its own, and so a later change to the defaults cannot
    # silently reinterpret what was run here.
    batch_size=4,
    lr=0.0006,
    sequence_length=327680,
    sample_rate=16000,
    hop_length=320,
    bins_per_semitone=4,
    learning_rate_decay_rate=0.98,
    learning_rate_decay_steps=10000,
    clip_gradient_norm=3,
    seed=42,

    # A ceiling on global_step, which under sp advances twice per batch -- once
    # per subnet optimizer. 1e6 is therefore the 500k updates at the top of the
    # paper's 200k-500k range. EarlyStopping on val_loss/all is what actually
    # ends the run.
    iterations=1_000_000,

    # reverb_level="rms" is also the default; it is named here because it is the
    # one augmentation choice that departs from Edwards. audiomentations peak-
    # normalises the reverb output to 0.5 regardless of input, which on real
    # items makes "did reverb fire?" readable from loudness alone at AUC 0.93
    # and swings the gain from 1.68x to 940x. HPPNet predicts velocity, so that
    # would corrupt a target rather than only leak a shortcut. "rms" preserves
    # excerpt energy, as Kaldi's wav-reverberate does, and matches every kong
    # arm already run here.
    augment=True,
    augment_asset_root="/home/simone-chieppa/Desktop/mtg_ir_datasets",
    reverb_level="rms",

    num_workers=4,
    logger_name="csv",
)
