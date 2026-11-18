# Piano-to-guitar transfer: paper experiments

Code and results for *A Systematic Evaluation of Piano to Guitar Transfer
Learning in Automatic Music Transcription* (Chieppa, Nkama, Serra, Rocamora,
submitted to ICASSP 2027). Supplementary material:
https://nkcemeka.github.io/Snap2MIDI/

## Models

Five models are re-implemented in this library (`snap2midi/`): Onsets and
Frames (OAF), OAFv2, HPPNet-sp (HPP), Kong et al. (Kong) and hFT-Transformer
(hFT).

TransKun (TK) is trained and evaluated from its official repository,
https://github.com/Yujia-Yan/Transkun at commit `c5cb837`, with the patch in
`transkun/` applied. `snap2midi/models/transkun` is not used in the paper.

## Setup

Python 3.12 and a CUDA GPU. From the repository root:

```
pip install -e .     # the library and requirements.txt
```

`data/prepare.py` needs ffmpeg. TransKun is installed from its patched clone
as `transkun/README.md` says; its dependencies upgrade numpy to 2, which
nnAudio2 does not support, so numpy is re-pinned afterwards.

Copy `config.example.yaml` to `config.yaml` and fill in the paths: the
datasets as released, the folders the prepared copies and feature stores go
to, the augmentation assets (`room_ir/` and `bg_noise/`, the files listed in
`snap2midi/augment/manifests/`), the checkpoints and the patched TransKun
clone.

## Data

The study's fixed lists are in `data/splits/`:

- `gaps_split.csv`: the 297 GAPS tracks used, as 240 train / 27 validation /
  30 test, with each track's score-alignment quality. The validation tracks
  were drawn from GAPS' train split; the list is used as is, because a new
  draw on a different copy of GAPS gives a different split.
- `goat_di_items.txt`: the 170 GOAT direct-input performances (172 minus
  `item_96` and `item_110`, whose aligned MIDI is about twice the audio length).
- `<set>_items.txt`: the pieces scored on each other test set.

With the paths set in `config.yaml`, run in this order:

```
python experiments/data/prepare.py --dataset all   # check every dataset; needs ffmpeg
python experiments/data/extract.py --model all --dataset all   # feature stores
```

`prepare.py` checks each dataset against the copy the study used, decodes
GAPS to WAV in the fixed split, converts GuitarSet's annotations to MIDI and
pairs Studio MAESTRO with its MIDI (details in its docstring). `extract.py`
writes one feature store per model and dataset. A store that was
interrupted is refused on the next run rather than skipped: delete it and
run again.

## Pretraining

```
python experiments/pretrain/pretrain.py --model <model>
```

trains a model on MAESTRO v3 with the augmentation pipeline of
`snap2midi/augment/`. TransKun: `transkun/README.md`.

## Evaluation

All six models are scored by one script:

```
python experiments/evaluate/score.py --model <model> --dataset <dataset> --checkpoint <checkpoint>
```

Every model is scored against the MIDI files of the test set with the same
mir_eval note metrics; the paper reports note F1 without offsets (onset
within 50 ms, correct pitch). A reference or estimated note that repeats a
note of the same pitch at the same onset counts once. Each model's output is
read on its own frame grid at the centre of the frame interval. Each model
decodes at its own decision constant unless `--threshold` (TransKun:
`--note-bias`) is given. The output file holds every piece's scores and
their mean. hFT and TransKun keep their transcriptions under `<work>/est/`,
so an interrupted run resumes where it stopped.

## Adaptation regimes

Each pretrained model is fine-tuned on the 240 GAPS training tracks under
four regimes that differ only in which weights train (`adapt/regimes.py`):
full, probe, LoRA and LoRA (temporal). TransKun is fine-tuned from its
repository (`transkun/README.md`); the other five with

```
python experiments/adapt/adapt.py --model <model> --regime <regime> --init <pretrained checkpoint>
```

For the five library models, full fine-tuning uses one tenth of the
pretraining learning rate (Kong: 1e-5); probe and LoRA use ten times that. `--full-rate` trains LoRA at the
full fine-tuning rate, the supplementary learning-rate control.

All regimes of a model train for the same budget: OAF and OAFv2 40,000
steps, HPP 60,000, Kong 100,000, hFT 8 epochs, TransKun 20 epochs.
Checkpoints are saved at fixed intervals. A regime is extended by half its
budget if the best validation F1 (at the model's own constant or calibrated)
over its last fifth of checkpoints beats the best before them by more than
one paired standard error over the 27 validation tracks; the test is
repeated once more after an extension. Only Kong LoRA was extended
(to 150,000 steps). The learning-rate control follows the final budget of its
model's LoRA run. `adapt.py` holds each run's budget.

## Checkpoint selection

Each fine-tuning run is scored on the 27 GAPS validation tracks at every
saved checkpoint and every decision constant (thresholds 0.05 to 0.95 in
steps of 0.05; TransKun: note biases -3 to 2 in steps of 0.25):

```
python experiments/select/grid.py --model <model> --series <model>_<regime> \
    --checkpoints <run folder> --out <grids>/<model>_<regime>.json
python experiments/select/pick.py --grids <grids>
```

`pick.py` takes the checkpoint with the highest mean note F1 at the model's
own constant (native) and the checkpoint and constant with the highest mean
note F1 together (calibrated). A tie goes to the earlier checkpoint.
`pick.py --extension` runs the extension test above. The pretrained models
are calibrated the same way, with their pretrained checkpoint as the only
candidate.

## Results

`results/` holds what the paper's tables are computed from:

- `regime_select/validation/`: the validation grid of every fine-tuning run;
  `regime_select/picks_argmax.json`: the checkpoints and constants chosen
  from them; `regime_select/zeroshot_validation/` and
  `regime_select/zeroshot_control.json`: the calibration grids and constants
  of the pretrained models.
- `regime_eval/`: the per-piece scores of every chosen checkpoint on GAPS
  test, GuitarSet and GOAT DI, at the model's own constant (`native`) and
  calibrated (`joint`).
- `zeroshot/`: the per-piece scores of the pretrained models on every test
  set, at their own constant and calibrated.

Every file is the output of `score.py` or `grid.py`; its header records
the model, checkpoint and settings it was made with.

## Checkpoints

The six pretrained models and the selected fine-tunes, with their checksums
and where to download them: `checkpoints.md`.
