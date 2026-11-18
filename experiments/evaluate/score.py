#!/usr/bin/env python
"""Score one checkpoint on one dataset, piece by piece.

The single evaluator behind every number in the paper, the same for all six
models: each model's function in models.py runs the model on the test set
and yields the reference and estimated notes of every piece, and every piece
is scored here with the same mir_eval note metrics (metrics.py). The paper
reports note F1 without offsets (`note_no_offset_f1`). The result is a
per-piece file (metrics.write_per_piece).

snap2midi.Evaluator is the package's general-purpose API and returns rounded
means; the paper needs every piece's score, so it is not used here.

Test sets: OAF, OAFv2, HPP and Kong read the feature store
<stores>/<model>_<dataset>/<split>/ of experiments/data/extract.py (refused
without its .complete marker); hFT and TransKun read the audio and reference
MIDI files of the test set directly, under the same names (the lists are
built by utils.test_items from the datasets of config.yaml), and keep their
transcriptions, one MIDI per piece, in --estimates (default
<work>/est/<model>_<set>_<checkpoint>), so an interrupted run resumes.

Reference: every model is scored against the notes of the set's MIDI files,
read with one rule (utils.read_midi_notes: same-pitch overlaps cut, so a
duplicated note counts once; sustain pedal extended on the piano sets), and
every estimate is read with the same rule. OAF's onsets are read at the centre
of their frame, since its labels floor onsets to the frame (models.oaf).

Usage:
    python experiments/evaluate/score.py --model kong --dataset smd \
        --checkpoint <pretrained>/kong.ckpt
    python experiments/evaluate/score.py --model hft --dataset guitarset \
        --checkpoint <pretrained>/hft.ckpt
    python experiments/evaluate/score.py --model oafv2 --dataset gaps_ft \
        --threshold 0.45 --checkpoint <pretrained>/oafv2.ckpt
"""

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from metrics import note_metrics, write_per_piece  # noqa: E402

MODELS = ("oaf", "oafv2", "hpp", "kong", "hft", "transkun")
STORE_MODELS = ("oaf", "oafv2", "hpp", "kong")
MARKER = ".complete"


def on_midi(notes, reference):
    """The reference of every piece replaced by the notes of its MIDI file
    (utils.midi_reference), and the estimate read with the same rule
    (utils.merge_overlaps): every model scored against the same notes."""
    from utils import merge_overlaps
    seen = set()
    for piece, _, est in notes:
        if piece in seen:
            raise SystemExit(f"{piece} comes in several excerpts (an OAF training split): "
                             "excerpts cannot be scored against the whole MIDI; use a store "
                             "that keeps whole tracks")
        seen.add(piece)
        yield piece, reference(piece), merge_overlaps(est)


def score(notes):
    """Every piece scored with note_metrics."""
    pieces, scores = [], {}
    for piece, ref, est in notes:
        pieces.append(piece)
        for key, value in note_metrics(ref, est).items():
            scores.setdefault(key, []).append(float(value))
    return pieces, scores


def notes(args, cfg):
    """The chosen model's notes on the chosen test set, and what to record."""
    import models
    from utils import set_name, test_items
    threshold = models.THRESHOLD.get(args.model) if args.threshold is None else args.threshold
    name = set_name(args.dataset, args.split)
    if name is None:
        raise SystemExit(f"{args.dataset}/{args.split} is not a test set of utils.test_items")
    if args.model in STORE_MODELS:
        store = cfg["stores"] / f"{args.model}_{args.dataset}"
        if not (store / MARKER).exists():
            raise SystemExit(f"{store} has no {MARKER} marker: extract it first with "
                             f"experiments/data/extract.py --model {args.model} --dataset {args.dataset}")
        items = store / args.split
        if not items.is_dir() or not any(items.iterdir()):
            raise SystemExit(f"{items} is empty or missing")
        print(f"items      : {items}\nthreshold  : {threshold}")
        extra = {"centre_onsets": True} if args.model == "oaf" else {}
        return (getattr(models, args.model)(args.checkpoint, items, threshold, args.limit, **extra),
                {"split": args.split, "threshold": threshold, **extra})

    rows = test_items(name, cfg)
    estimates = args.estimates or (cfg["work"] / "est" /
                                   f"{args.model}_{name}_{args.checkpoint.stem.replace('=', '')}")
    print(f"items      : {name}, {len(rows)} pieces\nestimates  : {estimates}")
    if args.model == "hft":
        print(f"threshold  : {threshold}")
        return (models.hft(args.checkpoint, rows, name, estimates, threshold,
                           args.keep_zero_velocity, args.limit),
                {"split": args.split, "threshold": threshold,
                 "keep_zero_velocity": args.keep_zero_velocity, "estimates": str(estimates)})
    print(f"note bias  : {args.note_bias}")
    return (models.transkun(args.checkpoint, rows, name, estimates,
                            cfg["transkun_repo"], args.note_bias, args.state, args.limit),
            {"split": args.split, "note_bias": args.note_bias, "state": args.state,
             "estimates": str(estimates)})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--dataset", required=True,
                    help="A store name of experiments/data/extract.py (e.g. smd, gaps_ft, "
                         "goat_di); hFT and TransKun read the same sets from the files.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decoding threshold; default the model's own (TransKun: see --note-bias).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N pieces (sorted), for a quick check.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <work>/scores/<model>_<dataset>_<checkpoint>.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="The split of the set (evaluation sets: test).")
    ap.add_argument("--estimates", type=Path, default=None,
                    help="hFT, TransKun: the folder for the transcriptions.")
    ap.add_argument("--keep-zero-velocity", action="store_true",
                    help="hFT: keep notes decoded with velocity 0 (the adapted checkpoints' decoder).")
    ap.add_argument("--note-bias", type=float, default=0.0,
                    help="TransKun: added to every interval score before decoding (default 0).")
    ap.add_argument("--state", default="auto", choices=["auto", "best", "last"],
                    help="TransKun: the state of a training checkpoint (auto: best if present).")
    args = ap.parse_args()
    cfg = config.load(args.config)
    if not args.checkpoint.is_file():
        raise SystemExit(f"no such checkpoint: {args.checkpoint}")

    print(f"model      : {args.model}\ncheckpoint : {args.checkpoint}")
    model_notes, settings = notes(args, cfg)
    from utils import midi_reference, set_name
    model_notes = on_midi(model_notes, midi_reference(set_name(args.dataset, args.split), cfg))
    pieces, scores = score(model_notes)
    out = args.out or (cfg["work"] / "scores" /
                       f"{args.model}_{args.dataset}_{args.checkpoint.stem.replace('=', '')}.json")
    header = {"model": args.model, "dataset": args.dataset,
              "checkpoint": str(args.checkpoint), "reference": "midi", **settings}
    summary = write_per_piece(out, header, pieces, scores)

    print(f"\n{args.model} x {args.dataset}: {len(pieces)} pieces")
    for key in sorted(summary):
        print(f"  {key:28s} {100 * summary[key]:6.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
