# hFT pre-flight: what to check before the 19-day MAESTRO run

Written 2026-08-05, after the 12 h trial job (Slurm 3100103) measured
**22.6 h/epoch** on an H100 PCIe. Twenty epochs is **~452 h ≈ 19 days**, which at
a 12 h walltime is **~38 chained jobs**.

The trial proved the model trains: all eight loss terms fell between step 70,164
and 140,329, and the 2nd stage beat the 1st on every head. It proved nothing
about whether a *chain* of 38 jobs works, and that is where the money is.

Companion to `CHECKPOINT_VALIDATION.md`, whose three rules apply here unchanged.
Rule 2 in particular: hFT's `select_ckpt` monitors `valid_total_loss`, which at
step 140,329 was 55% frames and 25% velocity, with **onset only ~5%** — the same
"ranks on nearly the opposite of what is reported" problem found in HPP and Kong.
`save_top_k=-1` means selection can be deferred, so this is not a launch blocker,
but the eval tooling has to exist before the run ends.

---

## Gate A — will it finish at all?

Failing any of these means the run cannot complete, no matter how correct the
code is. All are one-liners. Do them first.

- [ ] **A1. GPU-hour budget.** 452 GPU-hours on one H100. If `upf105_b` has less
      than that left, the run stops partway and 19 days buy an unusable partial
      result.

      sshare -A upf105_b -u resh000979
      sacctmgr show assoc user=resh000979 format=Account,Partition,GrpTRESMins,MaxWall

- [ ] **A2. Maximum walltime on the partition.** 12 h is what the sbatch asks
      for; it may not be the cap. Every extra hour per job removes a resume
      cycle, and each cycle costs ~8 min of setup plus the risk in Gate B.

      sinfo -p res_gpu -o "%P %l %L %D %G"

      At 12 h → 38 jobs. At 24 h → 19. At 48 h → 10. Take the longest offered.

- [ ] **A3. Job-count / concurrency limits.** Some QOS cap queued jobs per user.
      A 38-job chain that can only hold 2 in the queue needs babysitting or a
      submit loop.

      sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxWall

- [ ] **A4. Disk quota on /data/upf105.** `save_top_k=-1` with
      `val_check_interval=0.25` writes **4 checkpoints per epoch × 20 = 80**, at
      66 MB each = **5.3 GB**, plus the rolling checkpoint and `last.ckpt`.
      Note the code comment at `train_hft.py:214` estimates 1.3 GB — it assumes
      one checkpoint per epoch and is 4× low now that quarter-epoch validation
      is on.

      lfs quota -h -u resh000979 /data 2>/dev/null || quota -s
      du -sh /data/upf105/resh000979/*

- [ ] **A5. Dataset is fully staged and stays staged.** 38 jobs over 19 days is
      long enough to hit a scratch-purge policy. Confirm `/data/upf105/...` is
      project storage, not scratch with a 14-day sweep.

---

## Gate B — will the chain actually make progress?

This is the dangerous gate. Every failure here looks *exactly* like success:
jobs run, logs fill, checkpoints appear — and the model never advances. Nothing
in the trial exercised any of it, because the trial ran once, from scratch, and
was killed.

- [ ] **B1. `last.ckpt` vs `last-v1.ckpt` — the callback-change trap.**

      **This is now live, because the `LearningRateMonitor` added on 2026-08-05
      changed the callback list.**

      `CHECKPOINT_VALIDATION.md:149` records the mechanism: Lightning only
      reuses the `last.ckpt` name when the checkpointer's restored state already
      names that file. After a callback change nothing is restored, so it writes
      **`last-v1.ckpt`** and leaves the stale `last.ckpt` in place.

      `_resolve_resume_path` (`train_hft.py:173`) looks for exactly `last.ckpt`.
      If the trap fires, **every one of the 38 jobs resumes from step 148,000**,
      trains for 12 h, writes `last-v1.ckpt`, and the next job rewinds again.
      Nineteen days, zero progress, and the logs look perfectly healthy.

      Before the first real job:

      ls -la /data/upf105/resh000979/save_dir/hft_paper/
      mv /data/upf105/resh000979/save_dir/hft_paper/last.ckpt \
         /data/upf105/resh000979/save_dir/hft_paper/last.ckpt.pre_lrmonitor

      then point `resume_path` at the newest explicit checkpoint
      (`hft-rolling-step=00148000.ckpt`) for the *first* job only, and revert to
      `"last"` once B2 confirms a clean `last.ckpt` is being written again.

- [ ] **B2. Resume actually resumes.** 30-minute job, then confirm the restored
      step is ~148,000 and not 0:

      grep -E "Restored all states|Restoring states" logs/hft_<jobid>.out

      python - <<'PY'
      import torch, glob
      for f in sorted(glob.glob('/data/upf105/resh000979/save_dir/hft_paper/*.ckpt')):
          c = torch.load(f, map_location='cpu', weights_only=False)
          print(f.split('/')[-1], 'global_step=', c['global_step'], 'epoch=', c['epoch'])
      PY

      Then check **no `last-v1.ckpt` appeared**. If one did, B1 fired.

- [ ] **B3. Requeue on SIGUSR1.** Never once exercised — the trial had
      `--signal`, `--requeue` and `--open-mode` commented out on purpose.
      Submit with `--time=00:20:00` and all three enabled. A requeued job shows
      **multiple rows** under one job ID:

      sacct -j <jobid> --format=JobID,State,Elapsed,Start,End,ExitCode

      If you see a single row ending `CANCELLED ... DUE TO TIME LIMIT`, the
      signal never reached Python and each job in the chain silently discards
      everything since its last rolling checkpoint.

- [ ] **B4. LR scheduler state survives resume.** `ReduceLROnPlateau` carries
      `best` and `num_bad_epochs`. If those reset every job, the plateau counter
      never reaches patience=10 and **the LR never decays for the whole run** —
      the exact failure `PlateauPerValidation` (`train_hft.py:44`) exists to
      prevent. Now checkable, thanks to the `LearningRateMonitor` added today:
      after B2's resume job, the `lr-Adam` series must be continuous across the
      job boundary, not restarting at 1e-4 each time.

- [ ] **B5. Epoch counter survives resume.** `EpochUpdateCallback`
      (`train_hft.py:14`) feeds `epoch` into the augmentation seed. If it resets
      to 0 on each resumed job, every job applies **identical damage to
      identical excerpts** — augmentation silently degrades to a fixed
      pre-corrupted dataset, with no crash and no warning. Print
      `trainer.current_epoch` at resume, or read `epoch=` from B2's dump.

- [ ] **B6. Two logger directories, one job.** The trial produced
      `logs/HFT/version_0` (empty) and `logs/HFT/version_1` from **two different
      PIDs** on gpu11, 131 s apart. Unexplained. The `logger_version="hft_paper"`
      fix pins the directory, so it no longer scatters the curve — but if two
      writers still open the same directory, find out why before trusting the
      curve. Confirm the next run produces exactly one `logs/HFT/hft_paper/`.

---

## Gate C — will the output be usable?

- [ ] **C1. The train→eval round trip works, on a checkpoint that already
      exists.** The single highest-value check here. If `evaluate_hft` cannot
      load a checkpoint this training produces — config mismatch, key mismatch,
      anything — you find out on day 19 instead of today. Run it on 2–3 pieces,
      not the full split:

      python - <<'PY'
      import snap2midi as s2m
      print(s2m.evaluator.Evaluator().evaluate_hft(
          "/data/upf105/resh000979/hft_maestro/audio/test",
          "/data/upf105/resh000979/save_dir/hft_paper/hft-epoch=00-valid_total_loss=0.1754.ckpt",
      ))
      PY

      Half an epoch in, don't expect the paper's ~0.95 note F1 — but it must be
      clearly off the floor. A near-zero F1 alongside a healthy loss means the
      decode path is broken, and no amount of further training fixes it.

- [ ] **C2. Selection tooling exists before the run ends.** There is no
      `scripts/eval_hft_paper.py`. Without it, 80 checkpoints arrive with no way
      to rank them on note F1, and the temptation is to fall back to
      `valid_total_loss` — the mistake `CHECKPOINT_VALIDATION.md` was written to
      stop. Port `scripts/eval_hpp_paper.py`, including its per-piece collection
      so rule 3's confidence intervals are available.

- [ ] **C3. The new logging is actually recording.** After the first short job,
      the tag list must now include `train_*` **and** an LR series:

      python - <<'PY'
      import glob
      from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
      for d in sorted(glob.glob('/home/resh000979/Snap2MIDI/logs/HFT/*')):
          ea = EventAccumulator(d); ea.Reload()
          print(d, sorted(ea.Tags()['scalars']))
      PY

      Expect `train_total_loss` with many points (not 1 per epoch) and
      `lr-Adam`. If either is missing, the fixes did not take effect.

- [ ] **C4. `CSVLogger`/`TensorBoardLogger` accept the `version=` keyword** in
      the pinned Lightning version. Verified only by syntax check locally, never
      executed — this machine has no Lightning install:

      python -c "from snap2midi.utils.train_utils import pl_logger; \
                 print(pl_logger('tensorboard', project_name='HFT', version='hft_paper').log_dir)"

      Expect `./logs/HFT/hft_paper`. A `TypeError` here kills every job in the
      chain at startup.

---

## Gate D — is it training on the right data?

Lower risk: the trial's falling losses are weak evidence these are already fine.
Cheap enough to confirm rather than assume.

- [ ] **D1. Split sizes match MAESTRO v3.** Expect 962 train / 137 validation /
      177 test.

      for s in train val test; do echo -n "$s: "; \
        ls /data/upf105/resh000979/hft_maestro/audio/$s | wc -l; done

- [ ] **D2. Dataset/slab integrity on the MAESTRO base path** (the verifiers
      default to MAPS):

      python scripts/verify_hft_dataset_equivalence.py \
        --base-path /data/upf105/resh000979/hft_maestro --split train

- [ ] **D3. Augmentation does what Edwards specifies, and `reverb_level="rms"`
      is in force.** This is the run's one deliberate deviation from the paper —
      if the reverb path peak-normalises instead, the **velocity targets are
      corrupted**, which is worse than a leaked shortcut.

      python scripts/verify_augmentation.py hft

- [ ] **D4. Validation is never augmented.** A validation set that drifts with
      the augmentation seed makes every checkpoint comparison meaningless, and
      all 80 candidates unrankable against each other.

---

## Changes to make before submitting

1. `run_hft_paper.py:30` — `EPOCHS = 20`
2. `scripts/hft_paper.sbatch` — uncomment `--signal=USR1@300`, `--requeue`,
   `--open-mode=append`
3. `scripts/hft_paper.sbatch` — raise `--time` to the Gate A2 maximum
4. **`ckpt_every_n_steps`: 2000 → ~20000.** The sizing rule at
   `train_hft.py:245` is `(walltime / step_time) / 8`; at the measured 3.9 it/s
   a 12 h job is ~168k steps, so 2000 is 10× tighter than intended. It is
   currently writing 66 MB every ~8.5 min — ~85 writes per job — and that
   overhead is *inside* the measured 3.9 it/s.
5. `resume_path` — see B1. Explicit checkpoint for the first job, `"last"`
   thereafter.

## Go / no-go

Do not submit the chain until **A1, A2, A4, B1, B2, B3, C1, C4** have passed.
Those eight are the ones where failure costs days rather than minutes. The rest
can run alongside the first job.
