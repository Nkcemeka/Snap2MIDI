# hFT pre-flight: what to check before the 19-day MAESTRO run

Written 2026-08-05, after the 12 h trial job (Slurm 3100103) measured
**22.6 h/epoch** on an H100 PCIe. Twenty epochs is **~452 h ≈ 19 days**, which at
a 12 h walltime is **~38 chained jobs**.

The trial proved the model trains: all eight loss terms fell between step 70,164
and 140,329, and the 2nd stage beat the 1st on every head. It proved nothing
about whether a *chain* of 38 jobs works, and that is where the money is.

Companion to `CHECKPOINT_VALIDATION.md`, with one correction to it. Sony's
training code selects on **validation loss**, not note F1:

    if best_loss_valid > epoch_loss_valid:
        best_loss_valid = epoch_loss_valid
        best_epoch = epoch; best_div = div

(`training/m_training.py`, github.com/sony/hFT-Transformer). So monitoring
`valid_total_loss` is paper-faithful, and the comment at `train_hft.py:212`
claiming otherwise is wrong. Ranking the candidates on note F1 instead is better
statistics — at step 140,329 the loss was 55% frames, 25% velocity and only ~5%
onset — but it is a declared **deviation**, not a correction. Say so in the
writeup.

The released MAESTRO model is `model_016_003.pkl` = epoch 16, division 3 of
20 × 4. So the authors' own best validation loss was ~85% of the way through,
which is the only evidence anywhere that 20 epochs is enough and 16 is roughly
where it stops helping. `EXE-TRAINING-MAESTRO.sh` hardcodes `-epoch 20` with no
justification given.

## Status 2026-08-05

| | result |
|---|---|
| A2 walltime | **PASS** — `res_gpu` is 7 days, `upf105_b`→`res_b` has no MaxWall override |
| A3 job limits | **PASS** — `res_b` allows 100 jobs |
| A4 disk | **PASS** — 1.1 PB free, no enforced quota |
| B1 `last-v1.ckpt` trap | **PASS** — did not fire; `last.ckpt` rewritten (+64 B of LR-monitor state) |
| B2 resume | **PASS** — 148,000 → 154,000, 4.0 it/s, matches the trial's 3.9 |
| B3 requeue | **DROPPED** — broken here, see below; replaced by a dependency chain |
| B4 LR logged | **PASS** — `lr-Adam` present at 1e-4 |
| B5 epoch counter | **PASS** — `epoch=0` preserved across resume |
| C1 eval round trip | **PASS** — 0.880 note F1 (no offset) over 177 test pieces at 4× coverage |
| C3 logging | **PASS** — one `logs/HFT/hft_paper`, `train_*_step` at 140 pts |
| C4 `version=` keyword | **PASS** — verified against Lightning 2.6.5 |
| D1 splits | **BLOCKED** — no test split on the cluster; see below |

Two things C1 exposed that are not on the original checklist:

- **There is no test split on Pirineus.** `audio/` holds `train` and `val` only,
  both single collated `.npy` slabs, and `evaluate.py:24` globs `*.npz`. Nothing
  on that machine is evaluable. Raw MAESTRO v3 does exist at
  `/data/upf105/resh000973/maestrov3_44100` (same group), so extraction is
  possible — ~13 GB of test npz, ~226 GB peak per `extract_maestro_hft.py:22`.
  Training is unaffected; **reporting is not**. Start this before day 19.
- **Mid-epoch resume is not data-exact.** Lightning warns the dataloader is not
  resumable. With `shuffle=True` (`train_hft.py:140`) the effect is a fresh
  permutation rather than skipped data, so it is statistically harmless across
  2 handoffs — but `deterministic=True` no longer implies a bit-identical rerun.
  Record it in the writeup the way HPP's three interruptions are recorded.

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

- [x] **B3. Requeue on SIGUSR1 — DROPPED, does not work here.** Tested twice
      (jobs 3101512, 3101523), both died at the 15-minute signal with exit 1
      rather than requeueing. Two independent causes, stacked:

      OSError: [Errno 18] Invalid cross-device link:
        '/scratch/upf105/.../tmpXXXX' -> '/data/upf105/.../hpc_ckpt_1.ckpt'

      Lightning saves that checkpoint atomically (temp file, then rename); Slurm
      puts `TMPDIR` on `/scratch` while `save_dir` is on `/data`, and `rename()`
      across filesystems is `EXDEV`. Exporting `TMPDIR` onto `/data` does not
      rescue it: the dataloader workers also receive SIGUSR1, run the same
      handler, and hit `CUDA error: initialization error` because a worker has no
      CUDA context — killing the job on `DataLoader worker exited unexpectedly`.

      Note the failure is **worse than not requeueing**: it leaves a truncated
      `hpc_ckpt_*.ckpt` (18.8 MB of 66 MB) in `save_dir`, which is precisely
      where Lightning looks on the next start.

      Replaced by a `--dependency=afterany` chain (see the sbatch header), which
      uses the resume path B2 already validated. Cost: the work since the last
      rolling checkpoint, bounded by `ckpt_every_n_steps=5000` to ~21 min per
      handoff, ~42 min across the run. Fixing requeue properly would need a
      `worker_init_fn` making workers ignore SIGUSR1 — not worth a DataModule
      change days before a 19-day run.

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

All done as of 2026-08-05, commit following this document:

1. `run_hft_paper.py` — `EPOCHS = 20`
2. `scripts/hft_paper.sbatch` — `--time=7-00:00:00`
3. `scripts/hft_paper.sbatch` — `--signal` / `--requeue` / `--open-mode` stay
   **commented out**, permanently, per B3
4. `scripts/hft_paper.sbatch` — `export TMPDIR` onto `/data`. Not needed by the
   current path, but `/scratch` vs `/data` is a latent `EXDEV` trap for any
   future atomic write into `save_dir`
5. `run_hft_paper.py` — `ckpt_every_n_steps=5000`, plus the passthrough it needs
   in `Trainer.train_hft`. Note this **tightens** the default rather than
   loosening it: the sizing rule at `train_hft.py:245` assumes a signal-based
   clean shutdown, which does not exist here, so the rolling checkpoint is now
   the only thing standing between a walltime kill and lost work
6. `resume_path` stays `"last"` — B1 showed the trap does not fire

## Go / no-go

Gates cleared: **A2, A3, A4, B1, B2, B4, B5, C1, C3, C4.** B3 dropped as
unavailable. Remaining before launch: **D2** and **D3**, the data and
augmentation verifiers.

Not blocking the launch, but blocking the *result*: **C2** (no
`scripts/eval_hft_paper.py` exists, so 80 checkpoints would arrive with no way
to rank them on note F1) and the **missing test split**. Both need to be done
before the run ends; neither needs to be done before it starts.

## Launch

    J1=$(sbatch --parsable scripts/hft_paper.sbatch)
    J2=$(sbatch --parsable --dependency=afterany:$J1 scripts/hft_paper.sbatch)
    J3=$(sbatch --parsable --dependency=afterany:$J2 scripts/hft_paper.sbatch)

Before submitting, clear the test artefacts from `save_dir`:
`hpc_ckpt_1.ckpt.CORRUPT` and `last.ckpt.backup`. Neither is matched by
Lightning's `hpc_ckpt_*.ckpt` glob, so both are inert — but a directory holding
a file named `.CORRUPT` at launch invites confusion three weeks later.
