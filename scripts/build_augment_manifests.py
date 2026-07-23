"""
Build the augmentation asset manifests from the analysis produced by
analyze_augment_assets.py.

Two filters are applied to the room impulse responses:

  1. Space type. Outdoor spaces, vehicles, bathrooms and pools are excluded:
     they do not correspond to any plausible piano recording environment.
     Concert halls and churches are KEPT -- MAESTRO is recorded in a concert
     hall, so those are the target acoustics rather than outliers.

  2. RT60 band. Below the lower bound the IR is effectively anechoic and the
     convolution is a no-op, which silently lowers the effective augmentation
     rate below the configured probability. Above the upper bound the tail
     smears note onsets while the labels stay at their original times, which
     is label noise rather than augmentation.

Background noise is filtered for stationarity: impulsive material (horns,
door chimes, dropped objects) adds spurious onset evidence at the SNRs used
here, so recordings with a wide short-term energy spread are dropped.

Training draws only from the train manifests; the test manifests are reserved
for the robustness evaluation, so measured robustness is generalization rather
than recall of the augmentation distribution.

The background noise split is taken from the source dataset, which already
separates whole TUT recordings (no recording contributes segments to both
sides). The impulse responses are re-split here: the source split happens to
place every plausible OPENAIR venue on one side and every extreme one on the
other, so after filtering the evaluation pool would contain no hall or church
acoustics at all. The re-split is stratified by RT60 so both pools span the
same reverberation range, and is performed at the level of physical spaces --
microphone positions of one room, and the left/right channels of one stereo
capture, all stay on the same side, otherwise a "held out" IR is the same room
measured slightly differently.

Usage:
    python scripts/build_augment_manifests.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# Space types that cannot plausibly host a recorded piano performance.
# Matched case-insensitively against the full relative path.
EXCLUDE_KEYWORDS = [
    # Outdoor / open air
    "outside", "outdoor", "streetsof", "bikepath", "field", "forest",
    "skatepark", "playground", "soccerfield", "driveway", "backyard",
    "frontyard", "fronyyar", "suburanbackyard", "suburbanbackyard",
    "parkinglot", "doorstep", "porch", "balcony", "bridge",
    "national-park", "wheldrake-wood", "trollers-gill", "monument",
    "courtyard", "waveguide-web-example",
    # Wet rooms
    "bathroom", "shower", "swimmingpool",
    # Vehicles / transit
    "car_", "_car_", "train_", "trainstation", "subwaystation", "tramstop",
    "innocent-railway-tunnel",
    # Underground / non-architectural
    "mine", "maes-howe", "cave", "reactor",
    # Commercial spaces unrelated to music performance
    "supermarket", "supermerket", "departmentstore", "toystore",
    "drycleaners", "icecreamparlor", "sandwichshop", "samdwichshop",
    "fastfood", "hospital", "doctorsoffice", "docrorsoffice", "mallfoodcourt",
]

# RT60 bounds in seconds: small room through concert hall.
RT60_MIN = 0.30
RT60_MAX = 2.50

# Stationarity bounds for background noise.
CREST_MAX_DB = 25.0
FRAME_STD_MAX_DB = 4.5

# Fraction of impulse response spaces held out for the robustness evaluation.
IR_TEST_FRACTION = 0.25
IR_SPLIT_SEED = 42


def space_id(record: dict) -> str:
    """Identify the physical space an impulse response was measured in.

    The three sources are organised differently: MIT contributes one space per
    file, AachenIR measures a handful of rooms from many microphone positions,
    and OPENAIR nests named sub-spaces (captured as L/R pairs) under a venue.
    """
    parts = Path(record["path"]).parts  # split/source/...
    source, tail = record["source"], parts[2:]

    if source == "AachenIR":
        return f"AachenIR/{tail[0]}"  # room directory
    if source == "OPENAIR":
        stem = Path(tail[-1]).stem
        for suffix in ("-channel_L", "-channel_R"):
            stem = stem.removesuffix(suffix)
        return f"OPENAIR/{tail[0]}/{stem}"
    return f"MIT/{Path(tail[-1]).stem}"


def split_ir_spaces(records: list[dict]) -> dict[str, str]:
    """Assign each space to train or test, stratified by RT60.

    Spaces are ranked by their median RT60 and split into terciles; the same
    proportion is held out from each tercile so neither pool ends up
    systematically drier or wetter than the other.
    """
    import random
    import statistics

    spaces: dict[str, list[float]] = {}
    for rec in records:
        spaces.setdefault(space_id(rec), []).append(rec["rt60_s"])

    ranked = sorted(spaces, key=lambda s: statistics.median(spaces[s]))
    rng = random.Random(IR_SPLIT_SEED)

    assignment = {}
    for tercile in range(3):
        group = ranked[tercile * len(ranked) // 3:(tercile + 1) * len(ranked) // 3]
        group = sorted(group)
        rng.shuffle(group)
        n_test = round(len(group) * IR_TEST_FRACTION)
        for i, space in enumerate(group):
            assignment[space] = "test" if i < n_test else "train"
    return assignment


def excluded_by_keyword(path: str) -> str | None:
    """Return the matching exclusion keyword, or None if the path is kept."""
    lowered = path.lower()
    return next((kw for kw in EXCLUDE_KEYWORDS if kw in lowered), None)


def filter_irs(records: list[dict]) -> tuple[list[dict], Counter]:
    """Apply the space-type and RT60 filters to the impulse responses."""
    kept, rejected = [], Counter()
    for rec in records:
        keyword = excluded_by_keyword(rec["path"])
        if keyword is not None:
            rejected[f"space type ({keyword})"] += 1
        elif rec["rt60_s"] < RT60_MIN:
            rejected["RT60 too short (near-anechoic)"] += 1
        elif rec["rt60_s"] > RT60_MAX:
            rejected["RT60 too long (onset smearing)"] += 1
        else:
            kept.append(rec)
    return kept, rejected


def filter_noise(records: list[dict]) -> tuple[list[dict], Counter]:
    """Drop impulsive noise recordings, keeping stationary ambience."""
    kept, rejected = [], Counter()
    for rec in records:
        if rec["crest_factor_db"] > CREST_MAX_DB:
            rejected["crest factor too high (impulsive)"] += 1
        elif rec["frame_db_std"] > FRAME_STD_MAX_DB:
            rejected["energy too variable (non-stationary)"] += 1
        else:
            kept.append(rec)
    return kept, rejected


def write_manifest(path: Path, records: list[dict], split: str) -> int:
    """Write the relative paths for one split, sorted for reproducibility."""
    rows = sorted(r["path"] for r in records if r["split"] == split)
    path.write_text("\n".join(rows) + "\n")
    return len(rows)


def report(name: str, records: list[dict], kept: list[dict], rejected: Counter) -> None:
    print(f"\n{name}: kept {len(kept)}/{len(records)} "
          f"({100 * len(kept) / len(records):.1f}%)")
    for reason, count in rejected.most_common():
        print(f"    -{count:4d}  {reason}")
    by_source = Counter(r["source"] for r in kept)
    print(f"    sources: {dict(by_source)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("scripts/augment_asset_stats.json"))
    parser.add_argument("--out", type=Path, default=Path("snap2midi/augment/manifests"))
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    ir_kept, ir_rejected = filter_irs(stats["room_ir"])
    noise_kept, noise_rejected = filter_noise(stats["bg_noise"])

    # Re-split the surviving impulse responses; see the module docstring.
    assignment = split_ir_spaces(ir_kept)
    for rec in ir_kept:
        rec["split"] = assignment[space_id(rec)]

    report("room_ir", stats["room_ir"], ir_kept, ir_rejected)
    report("bg_noise", stats["bg_noise"], noise_kept, noise_rejected)
    print(f"    spaces: {len(assignment)} "
          f"({sum(v == 'test' for v in assignment.values())} held out)")

    print()
    for name, kept in (("room_ir", ir_kept), ("bg_noise", noise_kept)):
        for split in ("train", "test"):
            manifest = args.out / f"{name}_{split}.txt"
            count = write_manifest(manifest, kept, split)
            print(f"  {manifest}  ({count} files)")


if __name__ == "__main__":
    main()
