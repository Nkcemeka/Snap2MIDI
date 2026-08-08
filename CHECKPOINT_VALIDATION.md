# Checkpoint validation: HPP and Kong

What has to be checked before either model's checkpoint is used as *the* result.
Written 2026-08-04, after finding that both runs selected their checkpoint on a
validation **loss** while both papers select on decoded **F1**.

## The three rules

1. **Select on validation, report on test.** Ranking candidates on the test
   split is choosing the number being reported.
2. **Rank on decoded note F1, not on validation loss.** Both papers monitor
   mir_eval F1. Our loss-based ranking is, by variance, ~88% frame loss for HPP
   and ~102% for Kong, with the onset head — which drives note F1 — at −0.6%
   and 1.6% respectively. It ranks on nearly the opposite of what is reported.
3. **A gap inside the confidence interval is a tie.** MAESTRO test is 177
   pieces, MAPS 60. Report per-piece CIs and do not break ties on the mean.

## Status

| | HPP (`save_dir/hpp_augmented`) | Kong (`save_dir/kong_paper_aug`) |
|---|---|---|
| training | **done** — 400,000 global = 200,000 batches | finished, 199,999 iterations |
| candidates on disk | **22** (232,324–400,000 global) | 5 (166,651–193,868) |
| currently reported | `hpp-step=247748` | `step=171651` → **`step=193868`** (K3, 2026-08-05) |
| how it was picked | min `val_loss/all` | min `valid_total_loss` |
| paper's own rule | dump every 2k, pick on val F1 | none — the 200k endpoint |

---

## HPP

- [x] **H1. Confirm the run reached its budget.** PASSED 2026-08-04 19:28:
      `Trainer.fit stopped: max_steps=400000 reached`, unit exited `inactive`,
      newest checkpoint `hpp-step=400000` = 200,000 batches exactly. The newest `hpp-step=*.ckpt`
      should report `global_step` 400,000 = 200,000 batches (sp advances
      global_step twice per batch). Anything less means it was interrupted
      again — and it has been, three times now, so check rather than assume.
      Known interruptions: two SIGTERMs from SSH teardown (08-01, 08-03) and a
      **worker SIGSEGV on 08-04 at step 344,000** (apport core dump in
      `/var/crash/_usr_bin_python3.12.1000.crash`, 782 MB — delete it once it
      is no longer wanted, the disk is at 93%). That last one cost at most
      4,000 batches because checkpoints are now every 4,000; under the old
      top-5 scheme it could have cost far more.

- [ ] **H2. Rank every candidate on the validation split**, on a fixed 40-piece
      subset so the comparison is paired and cheap (~3 min each, ~1 h total):

      for c in save_dir/hpp_augmented/hpp-step=*.ckpt; do
        .venv/bin/python scripts/eval_hpp_paper.py \
          --dataset maestro-val --limit 40 --checkpoint "$c"
      done

      Results land in `results/sweep/`. Rank on `note_no_offset_f1` — that is
      the paper's "Note F1". Then aggregate:

      .venv/bin/python scripts/rank_sweep.py --model hpp

      which prints the ranking with CIs, flags how many candidates are
      statistical ties, and evaluates the H3 and H4 triggers below.
      (`rank_sweep.py` replaced `rank_hpp_sweep.py` on 2026-08-05 — same
      numbers, now with a `--model kong` arm for K2.)

- [ ] **H3. Plot note F1 against iteration.** This is the check that decides
      whether 200k was the right budget, and it is the reason every checkpoint
      is being kept.
      - still climbing at 200k → extend to 500k (top of the paper's range) with
        evidence rather than by guess; needs a relaunch at
        `iterations=1_000_000`, see the note on `last.ckpt` below
      - flat over the last third → converged; 200k is inside the paper's range,
        spend the GPU on the clean control arm instead

- [ ] **H4. Trigger check — is the peak in the deleted region?** `save_top_k=5`
      destroyed every checkpoint before batch 116,162. If the sweep's winner is
      the *earliest* survivor (`hpp-step=232324`), that is evidence the curve
      was still rising when the deletions happened, and a from-scratch rerun
      with periodic checkpointing becomes justified. A winner in the middle or
      late closes the question.

- [ ] **H5. Confirm the top three on the full 137-piece validation split**
      (~10 min each, no `--limit`). If their CIs overlap, prefer the later
      checkpoint — it is the more paper-faithful choice at equal performance.

- [ ] **H6. Only then, evaluate the winner on test** — both splits, and only
      once:

      .venv/bin/python scripts/eval_hpp_paper.py --dataset maestro
      .venv/bin/python scripts/eval_hpp_paper.py --dataset maps
      .venv/bin/python scripts/compare_hpp_to_paper.py

- [ ] **H7. Sanity-check the winner's decode config** before believing it:
      threshold 0.4, frame rate 50 fps (16 kHz / hop 320), `model_type="sp"`.
      A 31.25 fps default would silently skew every note time.

---

## Kong

Two blockers have to be cleared before any Kong comparison means anything.

- [x] **K0a. Fix the rounding.** DONE 2026-08-05. `evaluate()` rounded every
      metric to **2 decimals**; at that precision 171,651 and 193,868 tie no
      matter what the truth is. Now 4, matching `evaluate_pedal()`, which
      already used 4.

- [x] **K0b. Per-piece Kong eval.** DONE 2026-08-05 — `scripts/eval_kong_paper.py`,
      same shape and metric names as `scripts/eval_hpp_paper.py`
      (`{checkpoint, test_path, thresholds, pieces, summary, scores}`), so CIs
      are computable and rule 3 applies. It imports the decode from the library
      rather than reimplementing it; verified against `evaluate()` on the first
      two `maestro-val` pieces — **all 12 metrics identical to 4 decimals**
      (note F1 0.7639, no-offset 0.9927, frame 0.8775).

      Two deliberate departures from `evaluate()`: the file list is **sorted**,
      so `--limit N` hands every candidate the same pieces and the sweep is
      paired; and a checkpoint that decodes nothing scores 0 instead of
      crashing. Output goes to `results/sweep/` for a sweep run, and to
      `results/<run>_<dataset>_per_piece.json` otherwise — named from the
      checkpoint's own directory, so the A/B arms in K4 need no extra flag.

      Cost: **~11.6 s/piece** measured. 137-piece val ≈ 27 min per checkpoint
      (~2 h 15 for all five); a 40-piece subset ≈ 8 min each (~40 min total).

      Note the frame row is not comparable to HPP's: this thresholds the frame
      head directly, the HPP script rasterises frames back from decoded notes.

          .venv/bin/python scripts/eval_kong_paper.py \
            --dataset maestro-val --limit 40 --checkpoint <ckpt>

- [ ] **K1. Evaluate `step=193868`** — the surviving checkpoint nearest the
      paper's 200k endpoint, which is the checkpoint Kong et al.'s procedure
      implies — and compare against the reported `step=171651`. ~50 min.
      Note the exact-200k model was never written to disk; 193,868 (96.9% of
      the budget) is the closest that exists.

- [x] **K2. Rank all five survivors on note F1.** DONE 2026-08-05,
      `bash scripts/sweep_kong_val.sh 40` → 40-piece `maestro-val` subset,
      46 min, then `.venv/bin/python scripts/rank_sweep.py --model kong`:

      | step | note F1 (no offset) | w/offset | w/off+vel | frame |
      |---|---|---|---|---|
      | 166651 | 96.84 ±0.97 | 81.03 | 79.53 | 88.90 |
      | 171651 *(val-loss pick)* | 96.86 ±0.97 | 81.01 | 79.59 | 88.88 |
      | 176651 | 96.89 ±0.95 | 81.16 | 79.69 | 88.96 |
      | 186651 | 96.91 ±0.95 | 81.10 | 79.67 | 88.89 |
      | **193868** *(endpoint)* | **96.92 ±0.93** | 81.14 | 79.67 | 88.97 |

      **F1 does not separate them either.** All five tie: the whole spread is
      0.08 against a ±0.95 interval. F1 is monotone in step, so the paired
      per-piece differences were checked too (same 40 pieces, so a paired test
      is the sharper one): 193868 − 166651 = +0.084 ±0.057, significant but
      trivial; 193868 − 171651 = +0.059 ±0.063, still a tie even paired.

      So the val-loss selection **did not measurably cost anything** — it was
      unjustified, not harmful. Say it that way in the writeup rather than
      implying a rescued result. No full 137-piece confirmation is needed:
      nothing is being selected on this sweep (see K3).

- [x] **K3. Pick by the paper's rule, because the sweep is a tie.** DECIDED
      2026-08-05: report **`step=193868`**, not `171651`. Kong et al. do no
      selection at all — their model is the 200k endpoint — and 193868 is the
      closest surviving checkpoint to it (96.9% of budget; the exact-200k model
      was never written to disk). It also happens to lead the F1 ranking, so
      the paper's rule and the sweep point the same way. The reason to switch
      is the rule, not the 0.06 F1, which is noise.

- [ ] **K4. Re-select the 55k A/B arms too** (`kong_baseline`,
      `kong_augmented`). Both were ranked by the same loss, so their *relative*
      comparison is less compromised — but both were picked on a frame-driven
      metric, and the A/B is the only augmentation control that currently
      exists for Kong.

---

## Record for the writeup

- Which checkpoints were candidates, and that the pool has **mixed provenance**:
  HPP's ≤135k-batch candidates survived a val-loss filter, the >135k ones are
  unfiltered periodic dumps. Selection is clean; the early candidate pool is not
  a random sample.
- That HPP's run is four sessions with three interruptions — two SIGTERMs and
  one worker SIGSEGV — costing ~1,446 batches of re-treading at the first
  handoff and at most 4,000 at the last, with the LR schedule continuous
  throughout.
- `results/hpp_augmented_valloss_selection.json` — what the superseded
  val-loss ranking had chosen, kept so the change of rule is auditable.
- That **frame F1 is not directly comparable to either paper**: this repo
  rasterises frames back from decoded notes rather than thresholding the frame
  head. Report it, caveat it, do not claim wins or losses on it.
- That there is **no clean in-house control** for either model at paper budget,
  so cross-domain gains cannot be attributed to augmentation — only compared to
  the published no-augmentation figure.

## Gotcha when relaunching any HPP training

Lightning's version counter only reuses the `last.ckpt` name when the
checkpointer's restored state already names that file. After a callback change
nothing is restored, so it writes `last-v1.ckpt` and leaves the stale
`last.ckpt` in place — and a crash-resume then silently rewinds to it. Before
relaunching: rename the existing `last.ckpt` out of the way and point
`resume_path` at the newest `hpp-step=*.ckpt`.
