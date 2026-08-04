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

Model selection is deliberately not Lightning's. Wei et al. dump on a fixed
stride and choose afterwards on validation F1; ranking checkpoints on
val_loss/all instead selects on a scalar that is 71% frame loss when the
reported figure is note F1, and throws the other candidates away. This run
therefore keeps every checkpoint and selects offline -- see
scripts/eval_hpp_paper.py. results/hpp_augmented_valloss_selection.json
records what the earlier val_loss-ranked phase had chosen.
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
    # per subnet optimizer. 400k is therefore 200k batches, the bottom of the
    # paper's 200k-500k range, and here it is a target rather than a ceiling:
    # early_stopping is off, so this is what ends the run.
    #
    # The first phase stopped itself at 134960 batches, below the range the
    # paper reports. That was EarlyStopping firing on val_loss/all after 25
    # checks, on a curve whose check-to-check noise (+/-2%) had grown larger
    # than the gap between the best check and the tail (~1.4%) -- it stopped on
    # noise, and it stopped short.
    iterations=400_000,

    # Wei et al. rank nothing: their train.py writes model-{i}.pt every 2000
    # iterations, keeps all of them, and logs mir_eval note/frame F1 at
    # validation. The model is chosen afterwards, on F1.
    #
    # This run's first phase instead ranked on val_loss/all and kept the best
    # five. That is the wrong scalar for the comparison being made: val_loss/all
    # is 71% frame loss and 10% onset loss, while the headline figure is note
    # F1, which rides on the onset head. Worse, it is destructive -- the five
    # survivors all sit in batches 116162-134960, and every earlier candidate
    # is gone.
    #
    # 8000 global steps is 4000 batches, so ~16 checkpoints (3.1 GB) to reach
    # the target. Selection happens offline afterwards, on MAESTRO validation
    # note F1, across these plus the five that survived phase one.
    checkpoint_every_n_steps=8_000,
    early_stopping=False,

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

    # Third session of one continuous run. 2026-08-01 started clean and took
    # SIGTERM at 52056 batches when the Cursor remote-SSH server shut down its
    # process tree; 2026-08-03 resumed from here and ran to the EarlyStopping
    # at 134960. That handoff cost ~1446 batches of re-treading -- one elevated
    # validation check, recovered within three -- and left the LR schedule
    # continuous, so the batch count is cumulative and the schedule is on spec.
    #
    # Lightning restores the optimizers and the LR decay from here. It will not
    # restore the EarlyStopping counter (no such callback now) or the ranked
    # ModelCheckpoint state (different callback key), which is the point: the
    # new periodic checkpointer starts clean and deletes nothing.
    #
    # Named checkpoint rather than last.ckpt, and not for style. Lightning's
    # version counter only skips the -vN suffix when the callback's own
    # restored state already names the file it is about to overwrite. The new
    # checkpointer has a different callback key, so nothing was restored, and
    # the first save wrote last-v1.ckpt while the stale phase-one last.ckpt sat
    # there untouched -- a resume after any crash would have silently rewound
    # to step 269920 and thrown the run away. Phase one's file is now
    # phase1-last-step269920.ckpt, so last.ckpt is this run's alone and is the
    # right resume target from here on.
    #
    # Remove this line to start clean -- left in place it silently resumes,
    # and the checkpoint it names outlives whatever is in save_dir.
    resume_path="save_dir/hpp_augmented/hpp-step=344000.ckpt",

    # wandb, as the kong runs use, so the arms are watchable side by side.
    logger_name="wandb",
)
