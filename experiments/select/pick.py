#!/usr/bin/env python
"""Choose each fine-tune's checkpoint from its GAPS-validation grid.

A series is one fine-tuning run (a model and a regime, e.g. oaf_lora). Its
grid holds the note F1 of every saved checkpoint on each of the 27 GAPS
validation tracks, at every threshold of the paper's grid (TransKun: every
note bias). The rule is argmax of mean note F1, with two columns:

  native   the best checkpoint at the model's own threshold (TransKun: bias 0)
  joint    the best checkpoint and threshold together, scored at that threshold

A tie goes to the checkpoint first in the grid's ranking order. Nothing from
GAPS test, GuitarSet or GOAT is read.

The 1x LoRA controls (<model>_lora_fullrate) are trained at their model's
final LoRA budget: each is picked only among its checkpoints at or before the
last checkpoint of the <model>_lora grid in the same folder (recorded as
"budget_cap" in the picks).

The default --grids folder, results/regime_select/validation/, holds the
study's 30 grids (select/grid.py); the picks made from them are
results/regime_select/picks_argmax.json.

A grid file holds: series, model, regime, operating_point ("threshold" or
"note_bias"), native (the model's own value), n_pieces, checkpoints (file
names in training order) and f1[checkpoint][value] = the note F1 of each
validation track. `--scores` builds one from the per-piece files that
evaluate/score.py --split val writes, one file per checkpoint and threshold
(bias), in any folder and under any names: the model, checkpoint and
threshold are read from each file's header.

The extension check (--extension) applies the pre-registered budget rule to
the four main regimes of every model (full, probe, lora, lora_temporal; the
1x LoRA controls follow their model's LoRA budget and are not tested): with
the checkpoints in training order (the grid's "checkpoints"), the window is
the last ceil(0.2 n) of them. In each column (native, joint) the best
configuration in the window is compared with the best before it, track by
track; the regime is extended by 50% of its base budget if, in either
column, the mean difference exceeds one paired standard error over the 27
tracks (z > 1). The check runs after each round, at most twice.

Usage:
    python experiments/select/pick.py
    python experiments/select/pick.py --extension --round 1 --out extension_round1.json
    python experiments/select/pick.py --series oaf_lora,hft_probe --out picks.json
    python experiments/select/pick.py --scores <folder of score.py files> --name kong_full
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402

GRIDS = config.EXPERIMENTS / "results" / "regime_select" / "validation"
RULE = "argmax mean note F1 on GAPS validation (27 tracks)"
# The models' own operating points (evaluate/models.py THRESHOLD; TransKun: bias 0).
NATIVE = {"oaf": 0.5, "oafv2": 0.5, "hpp": 0.4, "kong": 0.3, "hft": 0.5, "transkun": 0.0}


def _order_key(name):
    """Training order from the file name: step or epoch number, then the name."""
    m = re.search(r"step=?(\d+)", name) or re.search(r"epoch[=_]?(\d+)|\.ep(\d+)\.", name)
    return (int(next(g for g in m.groups() if g)) if m else 0, name)


def grid_from_scores(folder, name):
    """A grid in the released layout from score.py's per-piece files."""
    files = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))]
    if not files:
        raise SystemExit(f"no score files in {folder}")
    models = {f["model"] for f in files}
    if len(models) != 1:
        raise SystemExit(f"{folder}: files of several models {sorted(models)}")
    model = models.pop()
    op = "note_bias" if model == "transkun" else "threshold"
    pieces = sorted(files[0]["pieces"])
    f1 = {}
    for f in files:
        if sorted(f["pieces"]) != pieces:
            raise SystemExit(f"{folder}: not every file scores the same pieces")
        by_piece = dict(zip(f["pieces"], f["scores"]["note_no_offset_f1"]))
        ckpt = Path(f["checkpoint"]).name
        if str(f[op]) in f1.setdefault(ckpt, {}):
            raise SystemExit(f"{folder}: {ckpt} at {op} {f[op]} scored twice")
        f1[ckpt][str(f[op])] = [by_piece[p] for p in pieces]
    values = {tuple(sorted(v)) for v in f1.values()}
    if len(values) != 1:
        raise SystemExit(f"{folder}: the checkpoints were not all scored at the same {op}s")
    if str(NATIVE[model]) not in next(iter(values)):
        raise SystemExit(f"{folder}: no file at the model's own {op} {NATIVE[model]}")
    return dict(series=name, model=model, regime=name.split("_", 1)[1] if "_" in name else "",
                operating_point=op, native=NATIVE[model], n_pieces=len(pieces),
                checkpoints=sorted(f1, key=_order_key), f1=f1)


def pick(grid):
    """The native and joint picks of one series' grid."""
    order = grid["checkpoints"]
    native = grid["native"]
    f1 = {n: {float(t): np.array(v) for t, v in grid["f1"][n].items()} for n in order}
    nat = max(order, key=lambda n: f1[n][native].mean())
    jnt = max(order, key=lambda n: max(v.mean() for v in f1[n].values()))
    jt = max(f1[jnt], key=lambda t: f1[jnt][t].mean())
    return dict(model=grid["model"], regime=grid["regime"],
                operating_point=grid["operating_point"], n_checkpoints=len(order),
                n_pieces=grid["n_pieces"], rule=RULE,
                native=dict(checkpoint=nat, value=native,
                            gaps_val_f1=100 * float(f1[nat][native].mean())),
                joint=dict(checkpoint=jnt, value=jt,
                           gaps_val_f1=100 * float(f1[jnt][jt].mean())),
                argmax_is_last_checkpoint=dict(native=nat == order[-1], joint=jnt == order[-1]))


CONTROL = "_lora_fullrate"     # the 1x LoRA controls: <model>_lora_fullrate


def _step(grid, name):
    """The training step of a checkpoint: the grid's "steps" (hFT), else the file name."""
    if "steps" in grid:
        return grid["steps"][name]
    if grid["model"] == "hft":
        raise SystemExit(f"{grid['series']}: an hFT grid without \"steps\" has no step order")
    return _order_key(name)[0]


def cap_control(grid, lora):
    """A 1x control's grid cut to the checkpoints at or before the last of its
    model's LoRA run, so that the control is picked at the LoRA's final budget;
    and a record of the cut."""
    last = lora["checkpoints"][-1]
    limit = _step(lora, last)
    keep = [n for n in grid["checkpoints"] if _step(grid, n) <= limit]
    if not keep:
        raise SystemExit(f"{grid['series']}: no checkpoint at or before {lora['series']}'s {last}")
    cut = dict(lora_series=lora["series"], lora_last_checkpoint=last, step=limit,
               excluded=len(grid["checkpoints"]) - len(keep))
    return dict(grid, checkpoints=keep), cut


MAIN_REGIMES = ("full", "probe", "lora", "lora_temporal")
WINDOW, Z = 0.2, 1.0


def _best(f1, names, values):
    """(per-track F1, checkpoint, value) of the best mean in names x values; a
    tie goes to the first in training order, as in pick()."""
    return max(((f1[n][v], n, v) for n in names for v in values), key=lambda b: b[0].mean())


def extension(grid):
    """The extension check of one series' grid: per column, the best in the
    last ceil(WINDOW n) checkpoints against the best before them."""
    order = grid["checkpoints"]
    k = math.ceil(WINDOW * len(order))
    if k >= len(order):
        raise SystemExit(f"{grid['series']}: {len(order)} checkpoints, no early part to compare")
    early, late = order[:-k], order[-k:]
    f1 = {n: {float(t): np.array(v) for t, v in grid["f1"][n].items()} for n in order}
    columns = {}
    for col, values in (("native", [float(grid["native"])]), ("joint", sorted(f1[order[0]]))):
        (e, en, ev), (l, ln, lv) = _best(f1, early, values), _best(f1, late, values)
        d = l - e
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        z = float(d.mean() / se) if se > 0 else 0.0
        columns[col] = dict(early=dict(checkpoint=en, value=ev, f1=100 * float(e.mean())),
                            late=dict(checkpoint=ln, value=lv, f1=100 * float(l.mean())),
                            gain=100 * float(d.mean()), se=100 * se, z=z, triggers=z > Z)
    return dict(model=grid["model"], regime=grid["regime"], n_checkpoints=len(order),
                window=k, first_in_window=late[0], n_pieces=grid["n_pieces"],
                extend=any(c["triggers"] for c in columns.values()), **columns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--grids", type=Path, default=GRIDS,
                    help="Folder of <series>.json grids (default: the current protocol's).")
    ap.add_argument("--series", default="all", help="A name, a comma list, or all.")
    ap.add_argument("--out", type=Path, default=None, help="Default: <work>/picks.json")
    ap.add_argument("--scores", type=Path, default=None,
                    help="Build the grid of one run from a folder of score.py --split val files "
                         "(one per checkpoint and threshold) and pick from it; the grid is "
                         "written next to --out as <name>.json.")
    ap.add_argument("--name", default=None, help="--scores: the series name (<model>_<regime>).")
    ap.add_argument("--extension", action="store_true",
                    help="Run the extension check on the main regimes instead of picking.")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2],
                    help="--extension: the round recorded in the output (1: base budget).")
    args = ap.parse_args()
    work = config.load(args.config)["work"]

    if args.scores:
        name = args.name or args.scores.name
        grids = {name: grid_from_scores(args.scores, name)}
        out = args.out or work / "picks.json"
        grid_path = out.parent / f"{name}.json"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.write_text(json.dumps(grids[name], indent=1))
        print(f"grid of {len(grids[name]['checkpoints'])} checkpoints x "
              f"{len(next(iter(grids[name]['f1'].values())))} values written to {grid_path}")
    else:
        available = sorted(p.stem for p in args.grids.glob("*.json"))
        if not available:
            raise SystemExit(f"no grids in {args.grids}")
        names = available if args.series == "all" else args.series.split(",")
        for n in names:
            if n not in available:
                raise SystemExit(f"no grid for {n} in {args.grids}")
        grids = {n: json.loads((args.grids / f"{n}.json").read_text()) for n in names}
        out = args.out or work / "picks.json"

    if args.extension:
        if args.scores:
            raise SystemExit("--extension reads --grids, not --scores")
        main_series = {n: g for n, g in grids.items() if n.split("_", 1)[1] in MAIN_REGIMES}
        if not main_series:
            raise SystemExit("no grid of a main regime (full, probe, lora, lora_temporal)")
        skipped = sorted(set(grids) - set(main_series))
        result = dict(round=args.round, rule=dict(window=WINDOW, z=Z, columns=["native", "joint"]),
                      skipped=skipped, series={})
        print(f"extension check, round {args.round}: window last ceil({WINDOW} n), z > {Z}")
        print(f"{'':32}{'native':^27}{'joint':^27}")
        print(f"{'series':24}{'n':>4}{'win':>4}" + f"{'early':>8}{'late':>6}{'gain':>6}{'z':>6} " * 2
              + "  extend")
        for n, grid in main_series.items():
            e = result["series"][n] = extension(grid)
            cells = "".join(f"{c['early']['f1']:>8.2f}{c['late']['f1']:>6.2f}{c['gain']:>+6.2f}"
                            f"{c['z']:>+6.2f}{'*' if c['triggers'] else ' '}" for c in (e["native"], e["joint"]))
            print(f"{n:24}{e['n_checkpoints']:>4}{e['window']:>4}{cells}  {'YES' if e['extend'] else 'no'}")
        if skipped:
            print(f"not tested (not a main regime): {', '.join(skipped)}")
        out = args.out or work / f"extension_round{args.round}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        print(f"wrote {out}")
        return

    picks = {}
    print(f"{'series':24}{'n':>4}  {'native':42}{'@':>6}{'F1':>7}  {'joint':42}{'@':>6}{'F1':>7}")
    for n, grid in grids.items():
        cut = None
        if n.endswith(CONTROL):
            lora_path = args.grids / f"{n[:-len(CONTROL)]}_lora.json"
            if not lora_path.exists():
                raise SystemExit(f"{n} is picked at its LoRA's budget, and there is no {lora_path}")
            grid, cut = cap_control(grid, json.loads(lora_path.read_text()))
        e = picks[n] = pick(grid)
        if cut:
            e["budget_cap"] = cut
        print(f"{n:24}{e['n_checkpoints']:>4}  {e['native']['checkpoint'][:42]:42}"
              f"{e['native']['value']:>6.2f}{e['native']['gaps_val_f1']:>7.2f}  "
              f"{e['joint']['checkpoint'][:42]:42}{e['joint']['value']:>6.2f}"
              f"{e['joint']['gaps_val_f1']:>7.2f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picks, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
