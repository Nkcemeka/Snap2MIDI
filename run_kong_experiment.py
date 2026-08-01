"""Does augmentation improve out-of-distribution transcription?

Two runs, identical except the `augment` flag, trained on MAESTRO and evaluated
on MAPS. MAPS is the point: the model never sees it, and Edwards et al. measure
augmentation's benefit there (82.4 -> 86.4 note-onset F1). In-domain the benefit
is small or negative, so evaluating on MAESTRO test would likely show nothing
and that would be an artifact of the evaluation set, not a result.

Settings follow Edwards' ablation protocol (Tables III and IV) on the axes that
define the experiment -- lr 5e-4, full augmentation, MAESTRO train, MAPS test --
but not on batch size, which the hardware decides. Edwards used batch 32; that
needs ~38 GB of activations in float32 and the GPU here has 31.4, so this runs
batch 8. Float32 is kept rather than trading it for a larger batch: bfloat16
perturbs the log-mel front end by a mean of 0.86 nats, which would degrade both
arms of the comparison.

Step count is a floor, not a budget. 56,000 steps is 448,000 excerpts, half of
the ~900k that both Edwards' ablation (28,000 x 32) and train_kong's own default
(200,000 x 4) settle on. If the augmented-vs-clean gap on MAPS is ambiguous at
56,000, both runs resume from `last.ckpt` and continue -- StepLR decays on step
count alone, so extending a run does not distort the schedule.

Since both arms share every setting, the comparison between them is valid
regardless; only the absolute F1 stops being comparable to the paper.

Reverb uses level="rms" -- the excerpt's energy is preserved, as Kaldi's
wav-reverberate does by default. audiomentations instead peak-normalises the
convolution to 0.5 regardless of input, which on real items raises the mean
log-mel by 3.5 nats and makes "did reverb fire?" predictable from loudness alone
at AUC 0.93. That is a deviation from Edwards, who inherited the audiomentations
behaviour, and it is deliberate.

Usage:
    python run_kong_experiment.py baseline
    python run_kong_experiment.py augmented
    python run_kong_experiment.py evaluate save_dir/kong_baseline/<ckpt>.ckpt
"""

import sys

import snap2midi as s2m

MAESTRO = "./data/kong_maestro/"
MAPS_TEST = "./data/kong_maps/test"
ASSETS = "/home/simone-chieppa/Desktop/mtg_ir_datasets"

COMMON = dict(
    base_path=MAESTRO,
    batch_size=8,           # float32 activations peak at 18.9 GB of 31.4; 16 OOMs
    # 440k excerpts, 0.78 epochs; extendable, see above. A multiple of val_steps
    # so the run ends on a validation and last.ckpt is the final step -- the
    # checkpoint callback only saves there.
    iterations=55000,
    lr=5e-4,
    num_workers=8,          # loader does 270 items/s augmented, the GPU eats 17.5
    val_steps=5000,
    log_steps=200,          # ~90 s between metric points, so a stall shows up fast
    logger_name="wandb",
    seed=1234,
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    trainer = s2m.trainer.Trainer()

    if mode == "baseline":
        trainer.train_kong(**COMMON, save_dir="./save_dir/kong_baseline",
                           experiment_name="KongBaseline")
    elif mode == "augmented":
        trainer.train_kong(**COMMON, save_dir="./save_dir/kong_augmented",
                           experiment_name="KongAugmented",
                           augment=True, augment_asset_root=ASSETS,
                           reverb_level="rms")
    elif mode == "evaluate":
        if len(sys.argv) < 3:
            print("evaluate needs a checkpoint path")
            return 2
        scores = s2m.evaluator.Evaluator().evaluate_kong(MAPS_TEST, sys.argv[2])
        print(scores)
    else:
        print(f"unknown mode {mode!r}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
