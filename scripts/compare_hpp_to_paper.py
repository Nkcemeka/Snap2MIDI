"""Put the augmented HPPNet-sp run next to Wei et al. (ISMIR 2022).

Reads the per-piece JSONs written by eval_hpp_paper.py and prints the four
F1s the paper tabulates, with the per-piece 95% CI of the mean beside each.
The CI is what makes the deltas readable: MAESTRO's test split is 177 pieces
and MAPS' is 60, so a gap smaller than the interval is not a finding about
augmentation, it is the split.

The published column is HPPNet-sp: Table 3 for MAESTRO, Table 4 for MAPS.
Those are single numbers from a different codebase, so they carry no interval
of their own -- the comparison is one-sided and sub-point differences are not
interpretable.

Usage: .venv/bin/python scripts/compare_hpp_to_paper.py
"""

import json
from pathlib import Path

import numpy as np

# HPPNet-sp, Wei et al. 2022. MAESTRO from Table 3, MAPS from Table 4.
PAPER = {
    "maestro": {"frame_f1": 93.15, "note_no_offset_f1": 97.18,
                "note_f1": 83.80, "note_vel_f1": 82.24},
    "maps": {"frame_f1": 87.56, "note_no_offset_f1": 86.63,
             "note_f1": 64.39, "note_vel_f1": 59.77},
}

# Left is our key, right is the paper's column heading.
ROWS = [
    ("frame_f1", "Frame F1"),
    ("note_no_offset_f1", "Note F1"),
    ("note_f1", "Note w/ Offset F1"),
    ("note_vel_f1", "Note w/ Offset+Vel F1"),
]


def main():
    for dataset in ("maestro", "maps"):
        path = Path(f"results/hpp_augmented_{dataset}_per_piece.json")
        data = json.loads(path.read_text())
        scores = data["scores"]
        n = len(data["pieces"])

        print(f"\n{dataset.upper()} test -- {n} pieces, threshold "
              f"{data['threshold']}, {Path(data['checkpoint']).name}")
        print(f"{'':24s} {'ours':>16s} {'paper':>8s} {'delta':>8s}")
        for key, label in ROWS:
            values = 100 * np.array(scores[key])
            mean = values.mean()
            # 1.96 sigma / sqrt(n): the pieces are independent draws, and this
            # is the spread of the mean, not of the pieces.
            ci = 1.96 * values.std(ddof=1) / np.sqrt(n)
            paper = PAPER[dataset][key]
            print(f"{label:24s} {mean:7.2f} +/-{ci:5.2f} {paper:8.2f} "
                  f"{mean - paper:+8.2f}")


if __name__ == "__main__":
    main()
