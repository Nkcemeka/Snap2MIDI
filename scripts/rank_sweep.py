"""Rank an HPP or Kong checkpoint sweep on decoded note F1, with CIs.

Reads the per-piece JSONs eval_hpp_paper.py and eval_kong_paper.py write under
results/sweep/ and answers the questions each sweep exists for:

  HPP  H2 -- which checkpoint wins on the metric the paper selects on, note F1
             on the validation split, rather than on val_loss/all
       H3 -- is F1 still climbing at 200k batches, or flat? That decides whether
             to extend to the 500k top of the paper's range or stop here
       H4 -- is the winner the earliest survivor, i.e. might the peak lie in the
             region save_top_k deleted?

  Kong K2 -- same ranking question against valid_total_loss, whose five
             survivors span 0.09% and are therefore ranked by noise
       K3 -- if no candidate wins significantly, fall back to the paper's own
             rule: Kong et al. do no selection, their model is the endpoint

A gap smaller than the confidence interval is a tie, not a win: with 40 pieces
the interval is wide, and breaking ties on the mean is how a run ends up
reporting noise.

Usage:
    .venv/bin/python scripts/rank_sweep.py --model kong
    .venv/bin/python scripts/rank_sweep.py --model hpp [--pattern ...]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# The paper's "Note F1", for both papers, is onset-only -- mir_eval's no-offset
# variant.
RANK_KEY = "note_no_offset_f1"

MODELS = {
    # steps_per_batch: HPP-sp advances Lightning's global_step twice per batch,
    # so its step numbers are double the paper's iteration count. Kong advances
    # once, and its steps are iterations already.
    "hpp": {"pattern": "maestro-val_first40", "steps_per_batch": 2},
    "kong": {"pattern": "kong_maestro-val_first40", "steps_per_batch": 1},
}


def load(pattern: str):
    rows = []
    for path in Path("results/sweep").glob(f"{pattern}_*.json"):
        d = json.loads(path.read_text())
        # step=?(\d+): HPP's paths carry 'hpp-step=400000.ckpt' and Kong's
        # 'kong-step=step=193868-loss=...', so the '=' has to be optional
        # rather than absent -- without it every file is skipped and the loader
        # reports "no results found".
        m = re.search(r"step=?(\d+)", d["checkpoint"])
        if not m:
            continue
        scores = d["scores"]
        n = len(d["pieces"])
        row = {"step": int(m.group(1)), "n": n, "path": path.name}
        # Kong's filenames carry the selection loss; HPP's do not. Where it is
        # there, the ranking can be compared directly against the rule it
        # replaces.
        loss = re.search(r"loss=([\d.]+)\.ckpt", d["checkpoint"])
        row["loss"] = float(loss.group(1)) if loss else None
        for k in (RANK_KEY, "note_f1", "note_vel_f1", "frame_f1"):
            v = 100 * np.array(scores[k])
            row[k] = v.mean()
            row[k + "_ci"] = 1.96 * v.std(ddof=1) / np.sqrt(n)
        rows.append(row)
    return sorted(rows, key=lambda r: r["step"])


def ties_with(rows: list, target: dict) -> list:
    """Every candidate whose interval overlaps target's -- i.e. not beaten."""
    cut = target[RANK_KEY] - target[RANK_KEY + "_ci"]
    return [r for r in rows if r[RANK_KEY] + r[RANK_KEY + "_ci"] >= cut]


def report_hpp(rows: list, best: dict):
    # H3: compare the run's last third against the middle third. A model still
    # learning puts its best in the last third; a converged one scatters.
    third = max(1, len(rows) // 3)
    early, late = rows[:-2 * third], rows[-third:]
    if early and late:
        print(f"\nH3: mean {RANK_KEY} over first {len(early)} candidates "
              f"{np.mean([r[RANK_KEY] for r in early]):.2f}, "
              f"over last {len(late)} {np.mean([r[RANK_KEY] for r in late]):.2f}")

    # H4: the deleted region sits below the earliest survivor. If the winner is
    # that survivor, the peak may have been in what save_top_k destroyed.
    if best["step"] == rows[0]["step"]:
        print("\nH4 TRIGGERED: the winner is the earliest surviving checkpoint. "
              "The peak may lie in the region save_top_k deleted (below "
              f"{rows[0]['step'] // 2} batches) -- a from-scratch rerun with "
              "periodic checkpointing is now justified.")
    else:
        print(f"\nH4 clear: winner is not the earliest survivor "
              f"({rows[0]['step']}), so the deleted region demonstrably did "
              "not hold the best checkpoint.")


def report_kong(rows: list, best: dict):
    # K3: Kong et al. run to 200k and report that model -- no selection at all.
    # So where the sweep cannot separate the candidates, the endpoint is the
    # faithful choice, not whichever mean happens to be highest.
    endpoint = rows[-1]
    tied = ties_with(rows, best)
    if endpoint["step"] == best["step"]:
        print(f"\nK3: the endpoint {endpoint['step']} is also the F1 winner -- "
              "the paper's rule and the sweep agree, report it.")
    elif any(r["step"] == endpoint["step"] for r in tied):
        print(f"\nK3: {best['step']} leads on the mean but its interval "
              f"overlaps the endpoint {endpoint['step']} "
              f"({endpoint[RANK_KEY]:.2f} +/-{endpoint[RANK_KEY + '_ci']:.2f} "
              f"vs {best[RANK_KEY]:.2f} +/-{best[RANK_KEY + '_ci']:.2f}) -- a "
              f"tie. Report the endpoint {endpoint['step']}, which is Kong et "
              "al.'s own rule, and say why.")
    else:
        print(f"\nK3: {best['step']} beats the endpoint {endpoint['step']} "
              "outside the interval -- a real difference, so selection on "
              "validation F1 is justified over the paper's endpoint rule. "
              "Report it as a deviation.")

    # The rule being replaced: min valid_total_loss. Where that lands in the F1
    # ranking is the evidence that the loss was the wrong selector.
    scored = [r for r in rows if r["loss"] is not None]
    if scored:
        loss_pick = min(scored, key=lambda r: r["loss"])
        order = sorted(rows, key=lambda r: -r[RANK_KEY])
        rank = 1 + order.index(loss_pick)
        print(f"\nval-loss pick was step {loss_pick['step']} "
              f"(loss {loss_pick['loss']:.4f}); on note F1 it ranks {rank} of "
              f"{len(rows)} at {loss_pick[RANK_KEY]:.2f} "
              f"+/-{loss_pick[RANK_KEY + '_ci']:.2f}"
              + ("" if loss_pick["step"] == best["step"] else
                 f", {best[RANK_KEY] - loss_pick[RANK_KEY]:+.2f} behind the leader"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default="hpp")
    ap.add_argument("--pattern", default=None,
                    help="Sweep filename prefix. Defaults to the model's "
                         "40-piece maestro-val sweep.")
    args = ap.parse_args()

    spec = MODELS[args.model]
    pattern = args.pattern or spec["pattern"]
    rows = load(pattern)
    if not rows:
        raise SystemExit(f"no results/sweep/{pattern}_*.json found")

    per_batch = spec["steps_per_batch"]
    print(f"{args.model}: {len(rows)} candidates, {rows[0]['n']} pieces each, "
          f"ranked on {RANK_KEY}\n")
    print(f"{'step':>8s} {'iters':>8s} {'note F1':>16s} {'w/offset':>9s} "
          f"{'w/off+vel':>10s} {'frame':>8s}")
    for r in rows:
        print(f"{r['step']:8d} {r['step'] // per_batch:8d} "
              f"{r[RANK_KEY]:9.2f} +/-{r[RANK_KEY + '_ci']:4.2f} "
              f"{r['note_f1']:9.2f} {r['note_vel_f1']:10.2f} {r['frame_f1']:8.2f}")

    best = max(rows, key=lambda r: r[RANK_KEY])
    print(f"\nwinner on {RANK_KEY}: step {best['step']} "
          f"({best['step'] // per_batch} iterations), "
          f"{best[RANK_KEY]:.2f} +/-{best[RANK_KEY + '_ci']:.2f}")

    tied = ties_with(rows, best)
    print(f"statistical ties with it: {len(tied)} of {len(rows)} "
          f"-- steps {min(t['step'] for t in tied)}..{max(t['step'] for t in tied)}")

    (report_hpp if args.model == "hpp" else report_kong)(rows, best)


if __name__ == "__main__":
    main()
