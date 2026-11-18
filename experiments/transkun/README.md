# TransKun

TransKun is trained from its official repository,
https://github.com/Yujia-Yan/Transkun, at commit `c5cb837`, with
`transkun.patch` applied:

```
git clone https://github.com/Yujia-Yan/Transkun.git
cd Transkun
git checkout c5cb837
git apply /path/to/Snap2MIDI/experiments/transkun/transkun.patch
pip install -e .
pip install "numpy<2"   # TransKun's dependencies upgrade numpy; nnAudio2 needs numpy 1
```

Snap2MIDI must be importable too (`pip install -e /path/to/Snap2MIDI`): the
augmentation is Snap2MIDI's own. Set `transkun_repo` in
`experiments/config.yaml` to the patched clone.

## Pretraining on MAESTRO

TransKun reads 44.1 kHz audio and neither resamples nor checks the rate, so
the 2017 and 2018 MAESTRO v3 recordings (48 kHz) are first resampled to
44.1 kHz; the other years are 44.1 kHz already. Then, from the patched clone,
build the train / validation / test files and train on 2 GPUs (`--batchSize`
is per GPU, so 24 excerpts per update):

```
python -m transkun.createDatasetMaestro /path/to/maestro_44k \
    /path/to/maestro_44k/maestro-v3.0.0.csv /path/to/transkun_meta

python -m transkun.train /path/to/runs/transkun_pretrain/ckpt.pt \
    --nProcess 2 \
    --datasetPath /path/to/maestro_44k \
    --datasetMetaFile_train /path/to/transkun_meta/train.pickle \
    --datasetMetaFile_val /path/to/transkun_meta/val.pickle \
    --modelConf transkun/pretrained/2.0.conf \
    --batchSize 12 --max_lr 4e-4 --weight_decay 1e-2 --nIter 500000 \
    --gradClippingQuantile 0.8 --dataLoaderWorkers 8 \
    --augment --augmentAssetRoot /path/to/augmentation_assets \
    --reverbLevel rms --augmentSeed 0 --augmentTrackIdMode stem \
    --seed 1234 --allow_tf32
```

These are the settings of Yan & Duan (2024) apart from the augmentation and
TF32. `--nIter` sizes the learning-rate schedule but does not stop training;
the released checkpoint is the run's state at iteration 477,150. Training
resumes from `ckpt.pt` if it exists.

## Fine-tuning on GAPS

Build TransKun's files for GAPS from the WAVs of `experiments/data/prepare.py`
(44.1 kHz, in the study's split), then fine-tune from the pretrained
checkpoint. The full regime is plain `transkun.train`, every parameter
trained:

```
python scripts/createDatasetGAPS.py /path/to/gaps_44k /path/to/gaps_meta

mkdir -p /path/to/runs/transkun_gaps_full   # transkun.train does not create it
python -m transkun.train /path/to/runs/transkun_gaps_full/ckpt.pt \
    --nProcess 1 \
    --initWeightsFrom /path/to/transkun_478k_ep153.pt \
    --datasetPath /path/to/gaps_44k \
    --datasetMetaFile_train /path/to/gaps_meta/train.pickle \
    --datasetMetaFile_val /path/to/gaps_meta/val.pickle \
    --modelConf transkun/pretrained/2.0.conf \
    --batchSize 12 --max_lr 3e-5 --weight_decay 1e-2 \
    --nIter 10520 --maxEpochs 21 --saveEveryEpoch \
    --gradClippingQuantile 0.8 --dataLoaderWorkers 8 --seed 1234 --allow_tf32
```

`--nIter 10520` is 20 epochs of the 240 training tracks (about 526 updates each);
`--maxEpochs 21` leaves room for the 500-step warm-up, and `--saveEveryEpoch`
keeps `ckpt.pt.ep000.pt` to `ckpt.pt.ep020.pt` for checkpoint selection.

The probe, LoRA and LoRA (temporal) regimes run the same command through
`scripts/train_regime.py`, which applies the regime of
`experiments/adapt/regimes.py` before the optimizer is built, at ten times
the learning rate:

```
mkdir -p /path/to/runs/transkun_gaps_lora
TK_REGIME=lora SNAP2MIDI_ROOT=/path/to/Snap2MIDI \
python scripts/train_regime.py /path/to/runs/transkun_gaps_lora/ckpt.pt \
    <the arguments above, with --max_lr 3e-4>
```

`TK_REGIME` is `probe`, `lora` or `lora_temporal`; `TK_REGIME=lora` with
`--max_lr 3e-5` is the learning-rate control. LoRA checkpoints are merged
before scoring:

```
python /path/to/Snap2MIDI/experiments/adapt/regimes.py merge \
    ckpt.pt.ep013.pt merged/ckpt.pt.ep013.pt
```

## What the patch changes

| File | Change |
|---|---|
| `transkun/AugmentatorEdwards.py` (new) | Exposes Snap2MIDI's augmentation pipeline to TransKun, so all six models train with the same augmentation. |
| `transkun/train.py` | `--augment` uses that pipeline (`--augmentLegacy` keeps TransKun's own); `--seed` replaces the wall-clock seed; for fine-tuning, `--initWeightsFrom` (start a new run from pretrained weights, with a fresh optimizer and learning-rate schedule), `--maxEpochs`, `--saveEveryEpoch`, and a learning-rate warm-up that runs once rather than at the start of every epoch. |
| `transkun/Data.py` | The augmentation is seeded per excerpt; `createDataset` passes its pedal setting as `extendSustainPedal` (upstream it lands in the list of pedal controllers); only `createDatasetGAPS.py` calls it. |
| `transkun/TrainUtil.py`, `transkun/ModelTransformer.py` | Validation also reports note onset precision, recall and F1. |
| `scripts/createDatasetGAPS.py` (new) | Builds TransKun's train / validation / test files for GAPS from `experiments/data/prepare.py`'s output. |
| `scripts/train_regime.py` (new) | Runs `transkun.train` under the probe, LoRA and LoRA (temporal) regimes of `experiments/adapt/regimes.py`. |

Transcription and evaluation code is unchanged: the paper's TransKun numbers
are computed by `experiments/evaluate/score.py`.

The MAESTRO pretraining ran before the fine-tuning changes were added, with
the augmentation changes alone. One fine-tuning change also affects a
pretraining run: originally the learning-rate schedule did not advance during
the first 500 steps of every epoch; with the patch this happens once, at the
start of training.
