"""Check hFT's two dataset backends agree, and that neither has drifted.

`HFTDataset` can read either the waveform slab written by the current extractor
("audio", the default, and the only one augmentation works with) or the
pre-rewrite npz spectrogram store ("legacy_feature"). An unaugmented run has to
mean the same thing through both, otherwise "reproduce the old baseline" is not
a claim anyone can rely on.

The equality that matters is the spectrogram. The audio backend recomputes the
mel from the waveform per item; the legacy backend slices a mel that was
computed once over the whole track. Those are the same arithmetic in a different
batching, so only float32 noise should separate them -- and a mistake here is
silent, because one frame of drift is 16 ms at hop 256, well inside the 50 ms
onset tolerance.

Labels are reported but not asserted equal. Each backend reads the labels stored
beside its own features, and a store written before commit 7724d29 holds labels
from the old pretty_midi `_midi2note` rather than Sony's mido one. The ~1.8%
disagreement that produces is a property of the two stores, not of the dataset
class, so it is measured and shown rather than treated as a failure.

Two differences are expected and excused:

  * the fft_bins // hop_sample frames after each track ends. The audio slab
    keeps the track's real decaying tail there; the old feature slab asserted
    silence. That region carries no labels and the new behaviour is the more
    faithful one.
  * nothing else.

Usage:
    python scripts/verify_hft_dataset_equivalence.py
    python scripts/verify_hft_dataset_equivalence.py --base-path data/hft_maps --probes 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snap2midi.models.hft.hft_dataset import HFTDataset

TOLERANCE = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path("data/hft_maps"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--probes", type=int, default=300)
    args = parser.parse_args()

    config = dict(base_path=str(args.base_path), n_div_train=1, n_div_val=1,
                  n_div_test=1, margin_b=32, margin_f=32, num_frame=128,
                  n_bins=256, n_slice=16, seed=1234)

    new = HFTDataset(config, split=args.split)
    try:
        old = HFTDataset({**config, "feature_source": "legacy_feature"},
                         split=args.split)
    except FileNotFoundError as exc:
        print(f"No legacy store to compare against: {exc}")
        return 2

    print(f"audio backend  : {len(new):,} items")
    print(f"legacy backend : {len(old):,} items")
    if len(new) != len(old):
        print("FAIL: the two stores do not describe the same excerpts")
        return 1
    if not np.array_equal(new.frames, old.frames):
        print("FAIL: dataset_idx differs between the stores, so any comparison "
              "would not be like-for-like")
        return 1
    print("  same length and same dataset_idx -> like-for-like\n")

    # Track ends, so the excused tail frames can be identified.
    raw = old._load_idx(0)
    breaks = np.flatnonzero(np.diff(raw) != 1) + 1
    track_end = [int(b[0]) + len(b) for b in np.split(raw, breaks)]

    n_item = config["margin_b"] + config["num_frame"] + config["margin_f"]
    tail = new.meta["fft_bins"] // new.meta["hop_sample"]

    probes = sorted(set(np.linspace(0, len(new) - 1, args.probes).astype(int).tolist()))
    worst_clean = worst_tail = 0.0
    n_tail = 0
    failures = []
    label_names = ("onset", "offset", "frames", "velocity")
    label_agree = {k: [0, 0] for k in label_names}

    for p in probes:
        a, b = new[p], old[p]
        if a[0].shape != b[0].shape or a[0].dtype != b[0].dtype:
            failures.append((p, f"shape/dtype {tuple(a[0].shape)}/{a[0].dtype} "
                                f"vs {tuple(b[0].shape)}/{b[0].dtype}"))
            continue

        diff = (a[0] - b[0]).abs().numpy()          # [mel_bins, n_item]
        start = int(new.frames[p]) - config["margin_b"]
        frames = np.arange(start, start + n_item)
        excused = np.zeros(n_item, dtype=bool)
        for end in track_end:
            excused |= (frames >= end) & (frames < end + tail)

        if excused.any():
            n_tail += 1
            worst_tail = max(worst_tail, float(diff[:, excused].max()))
        if (~excused).any():
            here = float(diff[:, ~excused].max())
            worst_clean = max(worst_clean, here)
            if here > TOLERANCE:
                failures.append((p, f"max|diff| {here:.3e}"))

        for name, x, y in zip(label_names, a[1:], b[1:]):
            label_agree[name][0] += int((x == y).sum())
            label_agree[name][1] += x.numel()

    print(f"probes: {len(probes)}  ({n_tail} include frames just past a track end)")
    print(f"spec, every frame that must match : {worst_clean:.3e}  "
          f"(tolerance {TOLERANCE:.0e})")
    print(f"spec, the {tail} frames after a track end : {worst_tail:.3e}  "
          f"(may differ by design)")
    for p, why in failures[:5]:
        print(f"   item {p}: {why}")

    print("\nlabels (reported, not asserted -- see module docstring):")
    for name in label_names:
        same, total = label_agree[name]
        print(f"  {name:<9} {100 * same / total:6.2f}% of cells agree")

    if failures:
        print("\nFAIL: the two backends disagree on the spectrogram.")
        return 1
    print("\nPASS: both backends produce the same spectrogram for the same excerpt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
