#!/usr/bin/env python
"""Crash-tolerant hFT evaluation, and an A/B table between two checkpoints.

Why this exists rather than a shell loop over `evaluator.evaluate_hft`. On
2026-08-07 two full-test-set evaluations died on this workstation with a native
fault and no Python traceback:

    traps: python[1331576] general protection fault ip:599574 ... (piece 135/177)
    python[1334573]: segfault at 1d9 ip 0000000000599564 ... (piece  56/177)

Same instruction in the CPython binary, 16 bytes apart, at two unrelated
pieces -- heap or refcount corruption from a C extension, surfacing wherever
the timing happens to put it. No package changed between the Aug 5 run that
completed 177 pieces and those failures, so it is latent rather than a
regression in a dependency.

The response is not to fix CPython. It is to stop a fault from costing the
whole run:

  * Work is split into chunks, each a directory of symlinks that
    `evaluate_test`'s `Path(test_dir).glob("*.npz")` picks up unmodified.
  * Every finished chunk is appended to a JSONL and fsynced immediately, so a
    crash loses at most the chunk in flight.
  * A worker subprocess evaluates chunks in sequence and is restarted from the
    JSONL when it dies. Model load is per chunk regardless (`evaluate_hft`
    reloads from the checkpoint each call), so restarting costs one interpreter
    startup, not the accumulated work.
  * A chunk that faults twice is quarantined by name and the run continues.
    Partial coverage with the failing files named beats no result at all.
  * Workers run under PYTHONFAULTHANDLER=1, so the next fault prints the Python
    stack that was executing instead of only a raw instruction pointer.

Aggregation across chunks is exact, not an approximation. `evaluate.py:71`
reduces each metric with an unweighted `np.mean` over per-piece values, so the
full-set mean is the chunk means weighted by piece count. The one loss of
precision is the library's `round(..., 3)` on each chunk mean, which leaves
<=0.0005 of rounding noise on the aggregate -- reported in the output, and
worth remembering when reading a delta near it.

Usage
-----
    # A/B two checkpoints over the MAESTRO test split
    scripts/eval_hft_ab.py \
        save_dir/hft-epoch=00-valid_total_loss=0.1754.ckpt \
        save_dir/hft-epoch=00-valid_total_loss=0.1784.ckpt

    # one checkpoint, smaller chunks (less lost per fault, more model reloads)
    scripts/eval_hft_ab.py save_dir/some.ckpt --chunk-size 5

Re-running resumes: chunks already in the JSONL are skipped. Delete the JSONL
under --out-dir to force a clean evaluation.

Evaluation parameters are `evaluate_hft`'s defaults, which are the paper's
(n_slice=16, thresholds 0.5). This script deliberately exposes no knobs for
them -- a checkpoint comparison is only meaningful if both sides ran identical
settings.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Metric families in the order a reader wants them: note-level first, since
# note F1 is what the paper reports and what a checkpoint decision turns on.
KEY_ORDER = ("note_", "note_vel_", "frame_")


def chunk_tag(path: Path) -> str:
    """A filesystem-safe identity for a checkpoint, stable across runs."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in path.stem)


def build_chunks(test_dir: Path, work_dir: Path, size: int) -> list[Path]:
    """Split the test set into directories of symlinks, `size` pieces each.

    Deterministic: the same sorted glob and the same size give the same chunks,
    which is what lets a resumed run trust chunk ids recorded by an earlier one.
    Symlinks rather than copies -- the test set is ~14 GB.
    """
    files = sorted(test_dir.glob("*.npz"))
    if not files:
        sys.exit(f"[eval_hft_ab] no .npz under {test_dir}")

    chunks = []
    for i in range(0, len(files), size):
        cdir = work_dir / f"chunk_{i // size:03d}"
        if cdir.exists():
            for link in cdir.iterdir():
                link.unlink()
        else:
            cdir.mkdir(parents=True)
        for f in files[i:i + size]:
            (cdir / f.name).symlink_to(f.resolve())
        chunks.append(cdir)
    return chunks


def done_chunks(jsonl: Path) -> dict[str, dict]:
    """Chunk results already on disk. Tolerates a line truncated by a crash."""
    out: dict[str, dict] = {}
    if not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue          # a fault mid-write; the chunk simply reruns
        out[rec["chunk"]] = rec
    return out


def run_worker(ckpt: Path, chunks: list[Path], jsonl: Path) -> int:
    """Evaluate `chunks` in a subprocess, appending one JSON line per chunk."""
    payload = json.dumps({
        "ckpt": str(ckpt),
        "chunks": [str(c) for c in chunks],
        "jsonl": str(jsonl),
    })
    env = os.environ | {
        "PYTHONFAULTHANDLER": "1",              # Python stack on a fatal signal
        "NNAUDIO_DISABLE_CITATION_REMINDER": "1",
    }
    return subprocess.run(
        [sys.executable, __file__, "--worker", payload], env=env).returncode


def worker_main(payload: str) -> None:
    """Subprocess entry point. Kept in this file so both modes share the layout."""
    import snap2midi as s2m

    cfg = json.loads(payload)
    jsonl = Path(cfg["jsonl"])
    evaluator = s2m.evaluator.Evaluator()

    for cdir in map(Path, cfg["chunks"]):
        n = len(list(cdir.glob("*.npz")))
        scores, frame_scores = evaluator.evaluate_hft(str(cdir), cfg["ckpt"])
        rec = {
            "chunk": cdir.name,
            "n": n,
            "files": sorted(p.name for p in cdir.glob("*.npz")),
            "scores": scores,
            "frame_scores": frame_scores,
        }
        # Append and fsync before touching the next chunk: the whole design
        # rests on this line having hit the disk when the fault arrives.
        with jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[worker] {cdir.name}: {n} pieces", file=sys.stderr)


def evaluate(ckpt: Path, chunks: list[Path], jsonl: Path,
             max_attempts: int) -> tuple[dict[str, dict], list[str]]:
    """Drive workers until every chunk is done or quarantined.

    Returns the completed records and the names of files in chunks that faulted
    `max_attempts` times.
    """
    attempts: dict[str, int] = {}
    quarantined: list[str] = []

    while True:
        done = done_chunks(jsonl)
        pending = [c for c in chunks
                   if c.name not in done and c.name not in
                   {q for q in attempts if attempts[q] >= max_attempts}]
        if not pending:
            break

        print(f"[eval_hft_ab] {ckpt.name}: {len(done)}/{len(chunks)} chunks done, "
              f"{len(pending)} to go", file=sys.stderr)
        rc = run_worker(ckpt, pending, jsonl)
        if rc == 0:
            continue

        # The worker died. Whatever it was on when it fell over is the first
        # pending chunk still absent from the JSONL.
        after = done_chunks(jsonl)
        crashed = next((c for c in pending if c.name not in after), None)
        if crashed is None:
            print(f"[eval_hft_ab] worker exited {rc} with nothing outstanding; "
                  "stopping", file=sys.stderr)
            break

        attempts[crashed.name] = attempts.get(crashed.name, 0) + 1
        state = ("quarantined" if attempts[crashed.name] >= max_attempts
                 else f"retry {attempts[crashed.name]}/{max_attempts}")
        print(f"[eval_hft_ab] worker exited {rc} on {crashed.name} -- {state}",
              file=sys.stderr)
        if attempts[crashed.name] >= max_attempts:
            quarantined += sorted(p.name for p in crashed.glob("*.npz"))

    return done_chunks(jsonl), quarantined


def aggregate(records: dict[str, dict]) -> tuple[dict[str, float], int]:
    """Piece-count-weighted mean over chunk means -- exact, see module docstring."""
    totals: dict[str, float] = {}
    n_total = 0
    for rec in records.values():
        n = rec["n"]
        n_total += n
        for group in ("scores", "frame_scores"):
            for key, value in rec[group].items():
                totals[key] = totals.get(key, 0.0) + value * n
    if not n_total:
        return {}, 0
    return {k: v / n_total for k, v in totals.items()}, n_total


def sort_keys(keys) -> list[str]:
    """note_ before note_vel_ before frame_, alphabetical within each family."""
    # Longest prefix first: "note_vel_pitch" also starts with "note_", and
    # matching that one first would file the velocity metrics under notes.
    by_length = sorted(KEY_ORDER, key=len, reverse=True)

    def rank(k: str) -> tuple[int, str]:
        for prefix in by_length:
            if k.startswith(prefix):
                return (KEY_ORDER.index(prefix), k)
        return (len(KEY_ORDER), k)

    return sorted(keys, key=rank)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="*", type=Path,
                    help="one checkpoint to evaluate, or two to compare")
    ap.add_argument("--test-dir", type=Path,
                    default=Path("data/hft_maestro/audio/test"))
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="pieces per chunk; smaller loses less per fault "
                         "but reloads the model more often (default 10)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/hft_eval_ab"))
    ap.add_argument("--work-dir", type=Path, default=Path(".eval_chunks"),
                    help="where the symlink chunk directories live")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="faults tolerated per chunk before quarantine")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_main(args.worker)
        return

    if not 1 <= len(args.checkpoints) <= 2:
        ap.error("give one checkpoint to evaluate, or two to compare")
    for ckpt in args.checkpoints:
        if not ckpt.exists():
            sys.exit(f"[eval_hft_ab] no such checkpoint: {ckpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(args.test_dir, args.work_dir, args.chunk_size)
    total_pieces = sum(len(list(c.glob("*.npz"))) for c in chunks)
    print(f"[eval_hft_ab] {total_pieces} pieces in {len(chunks)} chunks "
          f"from {args.test_dir}", file=sys.stderr)

    results = []
    for ckpt in args.checkpoints:
        jsonl = args.out_dir / f"{chunk_tag(ckpt)}.jsonl"
        records, quarantined = evaluate(ckpt, chunks, jsonl, args.max_attempts)
        means, n = aggregate(records)
        results.append((ckpt, means, n, quarantined, jsonl))

    # ---- report ----
    width = max(len(k) for _, m, _, _, _ in results for k in m) + 2
    print()
    for ckpt, means, n, quarantined, jsonl in results:
        cover = 100.0 * n / total_pieces if total_pieces else 0.0
        print(f"{ckpt.name}: {n}/{total_pieces} pieces ({cover:.1f}%)  -> {jsonl}")
        if quarantined:
            print(f"  quarantined after {args.max_attempts} faults: "
                  f"{', '.join(quarantined)}")
    print()

    if len(results) == 1:
        (_, means, _, _, _) = results[0]
        for key in sort_keys(means):
            print(f"{key:<{width}}{means[key]:>9.4f}")
    else:
        (ca, ma, na, _, _), (cb, mb, nb, _, _) = results
        print(f"{'metric':<{width}}{'A':>9}{'B':>9}{'delta':>9}{'delta %':>9}")
        print(f"{'':<{width}}{'-' * 36}")
        for key in sort_keys(set(ma) | set(mb)):
            a, b = ma.get(key), mb.get(key)
            if a is None or b is None:
                continue
            pct = 100.0 * (b - a) / a if a else float("nan")
            print(f"{key:<{width}}{a:>9.4f}{b:>9.4f}{b - a:>+9.4f}{pct:>+8.2f}%")
        print(f"\nA = {ca.name}\nB = {cb.name}")
        if na != nb:
            print(f"\nWARNING: different coverage ({na} vs {nb} pieces). The two "
                  "columns are not over the same set and the delta is not a "
                  "like-for-like comparison.")

    print("\nAggregate is a piece-weighted mean of chunk means, each rounded to "
          "3 dp by evaluate.py:71, so it carries <=0.0005 of rounding noise.")


if __name__ == "__main__":
    main()
