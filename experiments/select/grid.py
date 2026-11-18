#!/usr/bin/env python
"""The selection grid of one fine-tuning run: the note F1 of every checkpoint
on each of the 27 GAPS validation tracks at every threshold (TransKun: every
note bias), in the layout of results/regime_select/validation/ that pick.py
reads.

Each checkpoint runs once per track and its output is decoded at every value
(evaluate/models.py: at_thresholds, hft_grid, transkun_grid), so a grid costs
about one evaluation per checkpoint rather than one per value. Every piece is
scored as evaluate/score.py scores it: against the track's MIDI file,
duplicates counted once, estimates read with the same rule, OAF onsets at the
frame centre. OAF reads its whole-track validation store
gaps_ft_val (experiments/data/extract.py), since its gaps_ft/val holds
excerpts.

Checkpoints are listed in training order: the step in the file name, or for
hFT, whose names carry the epoch and the validation loss (several per epoch),
the global_step stored inside each checkpoint (kept in the grid's "steps"; of
two hFT files at one step, left by a resumed run, the later is kept and the
other listed in "dropped").
The grid is rewritten after every checkpoint (atomically) and a checkpoint
already in it is skipped, so an interrupted run resumes; a grid made with
other settings is refused. Each checkpoint's SHA-256 is kept in the grid
("sha256"), and a resume stops if a file of the same name differs from the
one the grid scored (another run saved under the same names). Nothing from
GAPS test, GuitarSet or GOAT is read.

Usage:
    python experiments/select/grid.py --model oafv2 --series oafv2_lora \
        --checkpoints <runs>/oafv2_gaps_lora/merged --out <work>/grids/oafv2_lora.json
    python experiments/select/grid.py --model hft --series hft_full --keep-zero-velocity \
        --checkpoints <runs>/hft_gaps_full --out <work>/grids/hft_full.json
then
    python experiments/select/pick.py --grids <work>/grids
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "evaluate"))
import config  # noqa: E402
from pick import NATIVE, _order_key  # noqa: E402

MODELS = tuple(NATIVE)
STORE_MODELS = ("oaf", "oafv2", "hpp", "kong")
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]      # 0.05 .. 0.95
BIASES = [round(-3.0 + 0.25 * i, 2) for i in range(21)]      # -3.0 .. 2.0
PATTERN = {"hft": "hft-epoch=*.ckpt", "transkun": "ckpt.pt.ep*.pt"}
MARKER = ".complete"


def hft_steps(ckpts):
    """({file name: global_step}, {dropped name: kept name}) of hFT checkpoints,
    the step read from inside each file (the names do not give the order within
    an epoch).

    A resumed hFT run restarts from its last periodic checkpoint, replays the
    steps after it and saves again at the end of the epoch, so two files can hold
    the same global_step with different weights. The training went on from the
    later one; the earlier one is an abandoned branch. Of two files at one step
    the most recently written is kept and the other dropped."""
    import torch
    by_step = {}
    for c in ckpts:
        step = torch.load(c, map_location="cpu", weights_only=False, mmap=True).get("global_step")
        if step is None:
            raise SystemExit(f"{c} stores no global_step: its place in training is unknown")
        by_step.setdefault(int(step), []).append(c)
    steps, dropped = {}, {}
    for step, files in by_step.items():
        files.sort(key=lambda c: (c.stat().st_mtime, c.name))
        if len(set(c.stat().st_mtime for c in files)) != len(files):
            raise SystemExit(f"checkpoints at step {step} share a write time: "
                             + ", ".join(c.name for c in files))
        steps[files[-1].name] = step
        for c in files[:-1]:
            dropped[c.name] = files[-1].name
            print(f"step {step}: {c.name} dropped, {files[-1].name} written later")
    return steps, dropped


def validation_notes(args, cfg, ckpt, values):
    """(piece, reference, {value: estimate}) of every validation track, as the
    model's function yields them (reference: the model's own, replaced by the
    MIDI in score_checkpoint)."""
    import models
    from utils import test_items
    if args.model in STORE_MODELS:
        dataset, split = ("gaps_ft_val", "test") if args.model == "oaf" else ("gaps_ft", "val")
        store = cfg["stores"] / f"{args.model}_{dataset}"
        if not (store / MARKER).exists():
            raise SystemExit(f"{store} has no {MARKER} marker: extract it first with "
                             f"experiments/data/extract.py --model {args.model} --dataset {dataset}")
        extra = {"centre_onsets": True} if args.model == "oaf" else {}
        return getattr(models, args.model)(ckpt, store / split, values, args.limit, **extra)
    rows = test_items("gaps_val", cfg)
    if args.model == "hft":
        return models.hft_grid(ckpt, rows, "gaps_val", values, args.keep_zero_velocity, args.limit)
    return models.transkun_grid(ckpt, rows, "gaps_val", values, cfg["transkun_repo"],
                                args.state, args.limit)


def score_checkpoint(args, cfg, ckpt, values, reference):
    """{piece: {value: F1}}, every track against its MIDI file."""
    from metrics import note_f1
    from utils import merge_overlaps
    scores = {}
    for piece, _, est in validation_notes(args, cfg, ckpt, values):
        if piece in scores:
            raise SystemExit(f"{piece} comes in several excerpts: not a whole-track store")
        ref = reference(piece)
        scores[piece] = {v: note_f1(ref, merge_overlaps(est[v])) for v in values}
    return scores


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--series", required=True, help="<model>_<regime>, e.g. oaf_lora")
    ap.add_argument("--checkpoints", required=True, type=Path,
                    help="The run's checkpoint folder (LoRA: its merged/).")
    ap.add_argument("--pattern", default=None, help="Default: the model's checkpoint names.")
    ap.add_argument("--out", required=True, type=Path, help="The grid file, <series>.json.")
    ap.add_argument("--keep-zero-velocity", action="store_true",
                    help="hFT: keep notes decoded with velocity 0 (the adapted checkpoints' decoder).")
    ap.add_argument("--state", default="auto", choices=["auto", "best", "last"],
                    help="TransKun: the state of a checkpoint (auto: best if present).")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N tracks (a test run).")
    args = ap.parse_args()
    cfg = config.load(args.config)
    if not args.series.startswith(args.model + "_"):
        raise SystemExit(f"series {args.series} is not a {args.model} run")

    values = BIASES if args.model == "transkun" else THRESHOLDS
    pattern = args.pattern or PATTERN.get(args.model, f"{args.model}-step=*.ckpt")
    ckpts = list(args.checkpoints.glob(pattern))
    if not ckpts:
        raise SystemExit(f"no checkpoint matching {pattern} in {args.checkpoints}")
    names = [c.name for c in ckpts]
    if len(set(names)) != len(names):
        raise SystemExit("two checkpoints share a file name")
    steps, dropped = hft_steps(ckpts) if args.model == "hft" else ({}, {})
    ckpts = [c for c in ckpts if c.name not in dropped]
    ckpts.sort(key=lambda c: (steps[c.name], c.name) if steps else _order_key(c.name))

    from utils import midi_reference
    reference = midi_reference("gaps_val", cfg)
    header = dict(series=args.series, model=args.model, regime=args.series.split("_", 1)[1],
                  operating_point="note_bias" if args.model == "transkun" else "threshold",
                  native=NATIVE[args.model], reference="midi",
                  keep_zero_velocity=args.keep_zero_velocity, limit=args.limit,
                  values=[str(v) for v in values],
                  order="global_step" if args.model == "hft" else "name")
    grid = json.loads(args.out.read_text()) if args.out.exists() else dict(header, f1={})
    for key, value in header.items():
        if grid.get(key) != value:
            raise SystemExit(f"{args.out} was made with {key}={grid.get(key)!r}, not {value!r}: "
                             "use another --out")
    if dropped:
        grid["dropped"] = {**grid.get("dropped", {}), **dropped}

    print(f"{args.series}: {len(ckpts)} checkpoints x {len(values)} values -> {args.out}")
    for ckpt in ckpts:
        digest = sha256(ckpt)
        if ckpt.name in grid["f1"]:
            recorded = grid.get("sha256", {}).get(ckpt.name)
            if recorded != digest:
                raise SystemExit(f"{args.out} scored a {ckpt.name} that is not this file "
                                 f"({'no fingerprint' if recorded is None else 'different SHA-256'}): "
                                 "use another --out for this run")
            print(f"[done] {ckpt.name}")
            continue
        per_piece = score_checkpoint(args, cfg, ckpt, values, reference)
        pieces = sorted(per_piece)
        if grid.get("pieces", pieces) != pieces:
            raise SystemExit(f"{ckpt.name} scored other pieces than the grid's")
        grid["pieces"], grid["n_pieces"] = pieces, len(pieces)
        grid["f1"][ckpt.name] = {str(v): [per_piece[p][v] for p in pieces] for v in values}
        grid["sha256"] = {**grid.get("sha256", {}), ckpt.name: digest}
        if steps:
            grid["steps"] = {**grid.get("steps", {}), ckpt.name: steps[ckpt.name]}
            grid["checkpoints"] = sorted(grid["f1"], key=lambda n: (grid["steps"][n], n))
        else:
            grid["checkpoints"] = sorted(grid["f1"], key=_order_key)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        part = args.out.with_suffix(".json.part")
        part.write_text(json.dumps(grid, indent=1))
        part.replace(args.out)
        best = max((sum(f) / len(f), v) for v, f in grid["f1"][ckpt.name].items())
        print(f"{ckpt.name}: best {100 * best[0]:.2f} at {best[1]}")
    missing = [n for n in grid["checkpoints"] if n not in names]
    if missing:
        print(f"note: the grid also holds {len(missing)} checkpoints not in {args.checkpoints}")


if __name__ == "__main__":
    main()
