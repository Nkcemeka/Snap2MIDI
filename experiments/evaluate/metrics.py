"""The note metrics every model is scored with, and the per-piece result file.

One function scores one piece for all six models: mir_eval note precision,
recall and F1 at its defaults (50 ms onset tolerance, offset within 20% of the
note or 50 ms, velocity within 0.1), without offsets, with offsets, and with
offsets and velocity. The paper reports `note_no_offset_f1`.
"""

import json
from pathlib import Path

import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel


def note_metrics(ref, est) -> dict:
    """`ref` and `est` are (intervals, pitches, velocities): intervals in
    seconds (n x 2), pitches in Hz, velocities on the scale of the model's
    reference (mir_eval rescales the estimate to it)."""
    (ri, rp, rv), (ei, ep, ev) = ref, est
    out = {}
    p, r, f, _ = prf(ri, rp, ei, ep, offset_ratio=None)
    out.update(note_no_offset_precision=p, note_no_offset_recall=r, note_no_offset_f1=f)
    p, r, f, _ = prf(ri, rp, ei, ep)
    out.update(note_precision=p, note_recall=r, note_f1=f)
    p, r, f, _ = prf_vel(ri, rp, rv, ei, ep, ev, velocity_tolerance=0.1)
    out.update(note_vel_precision=p, note_vel_recall=r, note_vel_f1=f)
    return {k: float(v) for k, v in out.items()}


def note_f1(ref, est) -> float:
    """note_metrics' `note_no_offset_f1` alone (the same call), for grids."""
    (ri, rp, _), (ei, ep, _) = ref, est
    return float(prf(ri, rp, ei, ep, offset_ratio=None)[2])


def write_per_piece(path: Path, header: dict, pieces: list, scores: dict):
    """The per-piece result file: piece names, one list per metric in the same
    order, and the means."""
    summary = {k: float(np.mean(v)) for k, v in scores.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **header,
        "pieces": pieces,
        "summary": summary,
        "scores": {k: [float(x) for x in v] for k, v in scores.items()},
    }, indent=1))
    return summary
