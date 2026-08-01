"""Rebuild the reverb manifests to follow Edwards et al. rather than substitute
for them.

Edwards draws reverb from "14 impulse responses from echothief.com". EchoThief
is a library of deliberately dramatic spaces -- caves, tunnels, stairwells,
underpasses, fortresses, glaciers -- collected for unusual reverberation, not
for realism.

The pool this replaces was assembled from MIT, AachenIR and OPENAIR and then
filtered to exclude "outdoor spaces, vehicles, bathrooms and pools", on the
reasoning that they "do not correspond to any plausible piano recording
environment". That filter rules out precisely what EchoThief specialises in. The
result was a training pool dominated by the MIT survey's domestic rooms --
bedrooms, kitchens, living rooms, median RT60 0.57 s capped at 2.0 s -- against
Edwards' median 1.09 s reaching 3.47 s. Not a variation on his distribution but
close to its opposite, and reverb is the stage his ablation values most (+2.8 F1
alone, -3.6 when removed).

Two judgement calls, both forced by what the paper does not say:

  * Which 14. The paper does not identify them, so any 14 of the library's 115
    would be our choice rather than his. All 115 are used instead: same source,
    same character, a superset of whatever 14 he drew, and no arbitrary
    selection to defend. Each space is still seen thousands of times over a
    28k-step run.
  * No filtering. The RT60 band and space-type exclusions are deliberately not
    applied here -- Edwards applied none, and the point of this pool is to stop
    substituting our judgement for his. Note that reverb does not move onsets;
    a direct-sound-aligned IR leaves each onset where it was and adds energy
    after it, so a long tail blurs offsets and masks quiet onsets rather than
    shifting the onsets the headline metric scores.

The previous pool becomes the held-out robustness set. That is a better split
than the one it replaces: training now uses the paper's distribution, and
evaluation measures generalisation to realistic rooms the model has never seen.
The two pools are disjoint by construction, being different source datasets.

Verified before adoption: EchoThief survives the direct-sound alignment in
SpaceUniformImpulseResponse. Median onset-to-peak gap 0.31 ms, median DRR change
from trimming +0.23 dB, no truncated or empty impulse responses. More files have
a gap over 1 ms than in the old pool (43/115 vs 17/177), which is expected of
large spaces where a reflection can outweigh the direct arrival, but the DRR
effect stays negligible -- the same conclusion section 7 of the notes reached.

Usage:
    python scripts/build_edwards_reverb_manifests.py
    python scripts/build_edwards_reverb_manifests.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "snap2midi" / "augment" / "manifests"
DEFAULT_ASSET_ROOT = Path("~/Desktop/mtg_ir_datasets").expanduser()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            space, _, rel = line.partition("\t")
            rows.append((space, rel))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    echo_root = args.asset_root / "room_ir" / "EchoThief"
    if not echo_root.exists():
        print(f"EchoThief not found at {echo_root}.\n"
              f"Download EchoThiefImpulseResponseLibrary.zip from "
              f"https://www.echothief.com/downloads/ and unpack its category "
              f"folders there.")
        return 2

    # One measurement per space in this library, so the space column is just the
    # category and file name. Sampling per space then reduces to sampling per
    # file, which is what Edwards does over his fourteen.
    train = sorted(
        (f"EchoThief/{p.parent.name}/{p.stem}",
         f"EchoThief/{p.parent.name}/{p.name}")
        for p in echo_root.rglob("*.wav")
    )

    # The old train and test pools merge into the new held-out set: none of it
    # is used for training any more, so there is nothing to keep separate.
    test = sorted(set(read_manifest(MANIFEST_DIR / "room_ir_train.txt")
                      + read_manifest(MANIFEST_DIR / "room_ir_test.txt")))

    print(f"  train : {len(train):>4} impulse responses, "
          f"{len({s for s, _ in train}):>4} spaces  (EchoThief)")
    print(f"  test  : {len(test):>4} impulse responses, "
          f"{len({s for s, _ in test}):>4} spaces  (MIT / AachenIR / OPENAIR)")

    missing = [rel for _, rel in train + test
               if not (args.asset_root / "room_ir" / rel).exists()]
    if missing:
        print(f"  {len(missing)} manifest entries do not exist on disk, e.g. {missing[0]}")
        return 1
    print("  every entry resolves on disk")

    if args.dry_run:
        print("\n  --dry-run, nothing written")
        return 0

    for name, rows in (("room_ir_train.txt", train), ("room_ir_test.txt", test)):
        path = MANIFEST_DIR / name
        backup = path.with_suffix(".txt.pre_echothief")
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
            print(f"  backed up {name} -> {backup.name}")
        path.write_text("".join(f"{s}\t{r}\n" for s, r in rows))
        print(f"  wrote {name} ({len(rows)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
