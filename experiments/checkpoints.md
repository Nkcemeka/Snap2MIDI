# Checkpoints

The model files behind the results in `results/`: the six pretrained
models of Tables 1 and 2 and the 36 selected fine-tunes of Table 3, 42
files, 3.5 GB. Download: *(location to be added when the files are
hosted)*. `SHA256SUMS` next to the files verifies a download with
`sha256sum -c`. Every file is weights only (the state dict, the
hyper-parameters the loaders read, the epoch or step it was saved at); the
LoRA files have the adapters merged into the weights. Each file was checked
tensor by tensor against the original before release.

A file is used with `evaluate/score.py --checkpoint <file>`, at the
threshold (TransKun: `--note-bias`) of the column it reproduces.

## Pretrained (Tables 1 and 2)

| Model | File | Own decoding | Calibrated on GAPS validation | MB |
|---|---|---|---|---|
| OAF | `pretrained/oaf/oaf-step=670000.ckpt` | threshold 0.5 | 0.35 | 27 |
| OAFv2 | `pretrained/oafv2/oafv2-step=670000.ckpt` | threshold 0.5 | 0.45 | 124 |
| HPP | `pretrained/hpp/hpp-step=400000.ckpt` | threshold 0.4 | 0.7 | 190 |
| Kong | `pretrained/kong/kong-endpoint-step=200000.ckpt` | threshold 0.3 | 0.75 | 99 |
| hFT | `pretrained/hft/hft_ep17_last.ckpt` | threshold 0.5 | 0.8 | 22 |
| TransKun | `pretrained/transkun/transkun_478k_ep153.pt` | bias 0 | -1.25 | 113 |

The TransKun file holds its
last and best weights; the evaluator and TransKun's own `transcribe.py` take
the best. hFT's calibrated threshold is with `--keep-zero-velocity`.

## Fine-tuned on GAPS (Table 3)

The files are laid out as `<model>_<regime>/<file>` (regimes `full`,
`probe`, `lora`, `lora_temporal`), with the original file names, so the
checkpoint and threshold of every Table 3 column are the ones
`results/regime_select/picks_argmax.json` records (`native` and `joint`);
where the two picks differ, both files are released. Adapted hFT checkpoints
are decoded with `--keep-zero-velocity`.
