"""Reconstruct val_loss/all across both HPP sessions, resume included.

The augmented sp run is two processes: 2026-08-01 ran clean to ~52k batches
before SIGTERM, and 2026-08-03 resumed from its last.ckpt and ran to the
EarlyStopping at ~135k. Judging convergence from the second session alone
reads a plateau that may just be the tail of one long curve, so this stitches
the two together from the local .wandb stores.

Values are pulled out of the protobuf by pattern rather than through the wandb
API, which would want the runs re-synced. Each logged value appears twice in
the store (history record, then summary update), so consecutive duplicates are
collapsed.

Usage: .venv/bin/python scripts/hpp_val_trajectory.py
"""

import re
from pathlib import Path

import numpy as np

RUNS = [
    ("2026-08-01 (clean start, SIGTERM)", "wandb/run-20260801_194332-o1u1o8ok"),
    ("2026-08-03 (resumed, EarlyStopping)", "wandb/run-20260803_154455-caf9u5mh"),
]

# 'val_loss/all', field tag \x82\x01, one length byte, then the ASCII float.
PATTERN = re.compile(rb"val_loss/all\x82\x01(.)([0-9.eE+-]+)")


def values(run_dir: str) -> list[float]:
    store = next(Path(run_dir).glob("*.wandb"))
    data = store.read_bytes()
    out = []
    for length, number in PATTERN.findall(data):
        if len(number) != length[0]:
            continue  # length byte disagrees -- not a value record
        v = float(number)
        if not out or v != out[-1]:
            out.append(v)
    return out


def main():
    combined = []
    for label, run_dir in RUNS:
        v = values(run_dir)
        combined.append((label, v))
        print(f"\n{label}")
        print(f"  checks       : {len(v)}")
        print(f"  first / last : {v[0]:.6f} / {v[-1]:.6f}")
        print(f"  best         : {min(v):.6f} at check {int(np.argmin(v)) + 1}")

    all_v = [x for _, v in combined for x in v]
    print(f"\ncombined: {len(all_v)} checks, best {min(all_v):.6f} "
          f"at check {int(np.argmin(all_v)) + 1}")

    # How much of the total improvement each quarter of the run bought. A run
    # still worth extending puts real gains in its last quarter.
    span = all_v[0] - min(all_v)
    print(f"\nimprovement from first check ({all_v[0]:.6f}) to best: {span:.6f}")
    quarters = np.array_split(np.array(all_v), 4)
    running = all_v[0]
    for i, q in enumerate(quarters, 1):
        best_so_far = min(running, q.min())
        gained = running - best_so_far
        print(f"  quarter {i}: best {q.min():.6f}  gained {gained:.6f} "
              f"({100 * gained / span:5.1f}% of total)")
        running = best_so_far


if __name__ == "__main__":
    main()
