# Augmentation work — status and open decisions

Working notes for `snap2midi/utils/augmentator.py` and its integration into the six
models. Written 2026-07-26. Read the "Start here tomorrow" section first.

---

## Start here tomorrow

**§3.1 decided (2026-07-27): option (b), the map-style rewrite.** A second bug
in the same `__iter__` settled it — the shuffle is frozen. `self.g` is seeded in
the parent process but `randperm` runs in the worker's copy, so with
`persistent_workers=False` every epoch re-forks from identical state and yields
byte-identical order. Reproduced: `num_workers=0` varies per epoch,
`num_workers=4` (the real config, `train_hft.py:6`) does not. Combined with
`n_slice=16`'s fixed stride, hFT has trained on the same 1/16 of excerpts in the
same order for all 50 epochs — so §8's "n_slice … unaffected by any of the
above" is wrong, the two interact. `RandomSampler` fixes both for free, and
`HFTDivDataset` has only two call sites, so the churn argument for (a) is thin.

**§3.3, §3.4, §3.5 all done (2026-07-27).** Extraction writes audio, the dataset
is map-style over mapped slabs, MAPS is re-extracted, and hFT trains again.
Everything is verified against the June data — see each section.

**§3.2 done** — hFT augmentation is on, epoch plumbing verified inside the
workers. **§6 resolved** — `reverb_level` flag implemented, default faithful.
**§7 resolved** — trimming is correct, no change needed.

**Remaining, in order:**

1. **§5 — kong.** The reproduction run, and the thing everything else waits on.
   **Correction: the target is 86.4, not 88.4.** Edwards' 88.4 needs eight audio
   sources (original MAESTRO + Studio MAESTRO + six Pianoteq renderings) at
   200k steps. Training on MAESTRO alone with full augmentation gives **86.4**
   (Table IV) against **82.4** unaugmented (Table III), both at only 28,000
   steps — far cheaper than planned. That +4.0 swing validates asset
   substitution, manifests, alignment and seeding end to end.
   *Blocked on:* MAESTRO is only partly unpacked — `~/Downloads/maestro-v3.0.0`
   holds the 2004 folder only (9.1 GB of a 108 GB zip), and
   `maestro-v3.0.0.csv` is missing, which `_get_maestro_train_val_test`
   (`base_mode.py:407`) asserts on.
2. **§6 ablation** — rerun kong with `reverb_level="rms"` once (1) has landed,
   then set the default from the result.
3. **§4** — OAF. ~40 lines, no re-extraction.
4. **§5 cont.** — oafv2 / hpp wiring.
5. **§8** — smaller items, including the `n_slice` per-epoch rotation.
6. **Retrain an hFT baseline.** There isn't one: the `save_dir` checkpoints do
   not load and were trained on pre-`7724d29` labels.

**Not part of the augmentation work, but found while doing it and worth
tracking:**

- `data/hft_maps/feature/` is 17 GB of superseded data. Safe to delete now that
  the rewrite is verified.
- `frames_to_note` (`utilities.py:302`) uses a bare `assert False` when a decoded
  note has zero duration, aborting an entire evaluation run. Fine on a
  well-trained model; it will bite during an ablation where one configuration
  transcribes badly.
- The `save_dir` checkpoints no longer load (§3.5). There is currently **no
  usable hFT baseline** — one has to be retrained before any augmented result
  can be compared to anything.

---

## 1. What the augmentator does

Edwards et al., *A Data-Driven Analysis of Robust Automatic Piano Transcription*
(arXiv:2402.01424), Fig. 1. Five stages, each fired independently at p=0.5:

1. seven-band parametric EQ, gain −10 to +5 dB
2. background noise, SNR 17.5–25 dB
3. pitch shift, ±10 cents
4. seven-band parametric EQ, independent draw
5. reverb, convolution with one room impulse response

Confirmed from the paper (§IV-A): augmentation is applied **per training example
(~10 s), on the fly**, not per track and not precomputed. Their base model is
Kong et al.'s — the same architecture as our `kong`.

Verified working: stage fire rates over 400 excerpts are 0.497 / 0.492 / 0.477 /
0.542 / 0.477. Seeding from `(track_id, excerpt_start, epoch)` is reproducible
across separate `Augmentator` instances and across `num_workers=0` vs `4`. The
ambient `random` / `numpy` / `torch` RNG streams are left untouched.

---

## 2. Done

| | |
|---|---|
| **transkun hook** | Fixed. Was passing a `(N,1)` array into a mono-only call (crashed) and passing no identity (silently unseeded). Now squeezes at the call site, keeps a stereo assertion, and threads `track_id` / `excerpt_start` / `epoch`. `epoch` is set in `build_chunks`, which already runs once per epoch under `reload_dataloaders_every_n_epochs=1`. |
| **Pickle failure** | Fixed. `SpaceUniformImpulseResponse` held an `lru_cache` with no class-level fallback, so the whole `Augmentator` was unpicklable — fine under fork, fatal under `ddp_spawn` or notebooks. Now follows the same idiom as both audiomentations parents: uncached staticmethod, shadowed by the cached version, dropped in `__getstate__`. |
| **IR direct-sound alignment** | Verified, not changed. Raw IRs delay onsets by a median of 5.9 ms and up to 90.2 ms, putting 5 of 177 outside the 50 ms tolerance on their own. After alignment: 0.0 ms for all 177. |
| **hFT frame equivalence** | Proven. `scripts/verify_hft_frame_equivalence.py`. Per-item mel reproduces the stored whole-track feature to 1.9e-6 (float32 noise). Self-validating: a 1-frame shift reads 7.8, a half-hop shift 5.2, so the check demonstrably has teeth. |
| **Pitch shift timing** | Checked, fine. On real piano audio the stage shifts onsets by a mean of 6 ms, worst 20 ms, against a 50 ms tolerance. An EQ-only control measures 0.0 ms, confirming the method. Non-zero but safe — worth one line in the docstring, which currently implies it is exactly time-preserving. |

---

## 3. hFT — the main piece of work

**Decision made:** hFT gets on-the-fly augmentation like every other model, *not*
a baked-in pass at extraction. The paper augments per ~10 s example, so a
one-draw-per-track bake-in would be a real deviation, and it would tie the
nuisance variable to track identity.

hFT currently stores a log-mel per track and slices it by frame index, so the
waveform is gone by training time. The fix is to store audio instead and compute
the mel per item.

### Why this is cheaper than it first looks

- **Storage shrinks.** The mel at 256 bins / hop 256 is 64 KB per second of
  audio — identical to float32 mono 16 kHz, and double int16. So the 17 GB
  feature store becomes **~8.5 GB** of int16 audio.
- **Speed is free.** An item is 192 frames = 3.07 s of audio. Augmentation costs
  5.8 ms, the mel 48 ms, total 53.8 ms/item. hFT currently runs at ~5.7 it/s
  (523,007 steps / 91,015 s from wandb) — see §3.1 for the worker caveat.
- **Re-extraction is required anyway.** Unrelated to augmentation: commit
  `bdb0e10` changed the division filenames to `dataset_idx000.npz`, but
  `data/hft_maps` was extracted before that with `dataset_idx.npz`. The current
  code **cannot load the current data** — it raises `FileNotFoundError`. So the
  re-extraction cost that was counted against this plan is already sunk.
- **Padding needs no special handling.** `hft_mode.py:167-172` fills the gaps
  between tracks with `log(log_offset)`, which is exactly what silence produces
  through `log(mel + 1e-8)`. An audio slab zero-padded in the same layout
  reproduces the current padding for free. The first/last-item cases in the
  equivalence test exercise this and pass.

### 3.1 BLOCKING DECISION: dataset sharding and storage format

**The problem.** `HFTDivDataset.__iter__` (`hft_dataset.py:31`) splits work
across dataloader workers **by division**. `n_div_train` is 1, so with
`num_workers=4`, worker 0 does everything and workers 1–3 idle. That is
invisible today because loading pre-made spectrograms is nearly free. Once
augmentation and mel move into `__getitem__` at 53.8 ms/item, one effective
worker gives a ceiling of **4.6 batches/s** against a current **5.7 it/s** —
augmentation becomes the bottleneck and costs ~20% (25 h → ~31 h).

**Rejected: shard by division, raise `n_div` to 8.** Needs no dataset change,
but couples data layout to worker count forever (raise `num_workers` later and
workers idle again, silently), and degrades shuffling from global to
within-division.

**Rejected: shard by item, `np.load` the slab per worker.** `HFTDataset.__init__`
loads the whole division and is called *inside* `__iter__`, i.e. after fork, so
each worker gets a private copy — roughly 20 GB across four workers.

**Chosen approach: shard by item, memory-map the slabs.** Store the audio slab
and all four label arrays as `.npy` (not `.npz` — a zip archive cannot be
mapped) and open with `np.load(mmap_mode='r')`. The OS page cache is shared
across worker processes, so memory stays bounded no matter how many workers.

Benchmarked on a 1.5 GB mapped slab with random access:

```
random mmap reads : 71,574 items/s   (0.01 ms/item)
needed            :     ~23 items/s
process RSS       :   0.46 GB for a 1.5 GB file
```

3,000× the required throughput, with memory bounded and shared. This removes
the worker/layout coupling and preserves global shuffling.

**The remaining choice — how far to take it:**

| | **(a) Minimal** | **(b) Clean** |
|---|---|---|
| structure | keep `IterableDataset`, shard by item inside `__iter__` | convert to a map-style `Dataset` over `(div, idx)`, delete custom sharding |
| sharding correctness | still hand-written | handled by PyTorch |
| diff size | small | larger rewrite of `hft_dataset.py` |
| shuffle RNG | close to current | changes |
| risk | the custom `__iter__` is exactly what silently wasted 3 workers | more churn on a model that currently works |

Recommendation: **(b)**. The custom sharding is the bug; deleting it means it
cannot return. But (a) is defensible if minimising churn matters more.

**Cons that apply to either:**

- Shuffled access means random reads across a multi-GB file. Fine on SSD/NVMe
  (benchmarked above), catastrophic on a spinning disk. The assumption is real.
- The benchmark ran warm. The first epoch reads cold; even at a pessimistic
  200 MB/s that is ~2,000 items/s vs 23 needed, so it does not change the
  conclusion, but epoch 1 will be slower.
- `.npy` holds one array per file, so extraction writes more files.
- Extraction should write the slab with `np.lib.format.open_memmap` rather than
  building a 3.7 GB array in RAM and saving it (`hft_mode.py:172` currently does
  the latter).

### 3.2 Epoch plumbing for hFT — DONE

Done 2026-07-27. `EpochUpdateCallback` sets `dataset.epoch =
trainer.current_epoch` on `on_train_epoch_start`; `HFTDataModule.setup` builds
an `Augmentator` for **train only**, at the sample rate recorded in the
extraction meta. New `train_hft` kwargs: `augment`, `augment_asset_root`,
`augment_manifest_dir`. Default is off, so nothing changes unless asked.

**The `persistent_workers` guard is a live check, not a comment.** The callback
only reaches the workers because they re-fork each epoch. The callback therefore
reads `trainer.train_dataloader.persistent_workers` every epoch and raises if it
is ever True while augmentation is on — so the failure surfaces at the config
that caused it rather than as silently frozen augmentation.

**Verified — the hook ordering actually works.** This was the real risk: if
Lightning built the dataloader iterator before `on_train_epoch_start`, workers
would inherit a stale epoch and nothing would raise. Asked the workers directly
by having them report the epoch they saw:

```
trainer epoch | epoch seen inside workers | worker ids
     0        | [0]  ok                   | [0, 1, 2, 3]
     1        | [1]  ok                   | [0, 1, 2, 3]
     2        | [2]  ok                   | [0, 1, 2, 3]
```

Also confirmed the guard fires when `persistent_workers=True` is forced.

**Verified — augmentation behaves.** On real hFT items:

- changes the spec (max|diff| 7–18 in log-mel) and leaves all four label arrays
  bit-identical
- the draw differs at every epoch, at five excerpts spread across the split — so
  it is not frozen
- the same excerpt at the same epoch reproduces exactly (max|diff| 0) from a
  separate `Augmentator` instance
- stage fire rates over 400 real items: **0.497 / 0.520 / 0.475 / 0.512 /
  0.490**, all near the p=0.5 the paper specifies

**Note for §5:** transkun passes the same augmentator to *both* its train and
val datasets (`train_transkun.py:52,63`). Harmless today because
`config["augmentator"]` is hardcoded `None`, but it will augment validation the
moment that is turned on. hFT deliberately does not.

### 3.3 `hft_mode.py` — DONE

Done 2026-07-27. `_get_feature_hft` → `_get_audio_hft`, returning the resampled
waveform; per-track npz stores `audio` + `num_frames` under `audio/{split}/`.
`dataset_idx` stays in frame units and is unchanged.

Decided while implementing:

- **Slab is `total_num_frame × 256 + 2 × 1024` samples, not `× 256`.** Frame `f`
  needs samples from `f × 256 − n_fft/2`, which runs negative at `f = 0`. With
  `audio_pad = n_fft // 2 = 1024` of headroom at each end, sample position of
  frame `f` is `f × 256 + 1024` and an item read is one flat contiguous slice —
  no branch, no edge case. Verified the furthest item ends at exactly
  `total_num_frame`, so the slab is neither short nor over-allocated.
- **float32, not int16.** The mel at 256 bins costs 1024 B/frame and 256 float32
  samples cost 1024 B/frame — *identical*. The train slab is 1.93 GB either way,
  so int16 would save 1 GB and cost bit-exactness against the old baseline.
  §3.5's "known residual" no longer applies.
- **Everything is `.npy` now, written via `open_memmap`.** npz is a zip and
  cannot be mapped; open_memmap also removes the build-in-RAM-then-save step.
- **Fixed a latent drift bug.** The old feature loop advanced `loc_d` by the
  track's own frame count while idx and all four label loops advanced by
  `max(feature, label)`. Any track whose MIDI outlived its audio would offset
  the feature slab against the labels for every track after it. 0/139 train and
  0/60 test MAPS tracks trigger it, so no existing result is affected, but it
  would have fired on MAESTRO or GOAT. All six arrays now share one `_stride()`.
- **`meta/{split}/dataset_meta{div}.json`** records sr / hop / fft_bins /
  audio_pad / total_num_frame so the dataset can check its frame→sample
  arithmetic instead of hardcoding it.
- **One deliberate, benign difference.** The old slab stored exactly
  `num_frames` frames per track and left the rest at `log(log_offset)`. The new
  one stores samples, so the ≤ 8 frames after a track ends see its real decaying
  tail rather than hard silence. That region carries no labels, and the new
  behaviour is the more faithful one — but it means the step-4 old-vs-new
  comparison must exclude track tails.

Downstream updated so nothing breaks: `evaluate.py` computes the whole-track mel
from stored audio via a new shared `inference.mel_from_audio`; `evaluate_hft`
gained the mel params it now needs; `train_hft.py`'s val check and the root
`evaluate_hft.py` point at `audio/` instead of `feature/`.

Set `n_div_train` sensibly at extraction even though item-sharding no longer
requires it.

**Verified: `scripts/verify_hft_slab_layout.py`.** Runs the real extraction over
a few tracks, rebuilds the old feature slab, and compares it against per-item
mels read out of the new audio slab at 40 `dataset_idx` probes. Worst diff
1.9e-6 (float32 noise). Self-validating like its sibling: a 1-frame slip reads
6.5, a half-hop 4.3, so it demonstrably catches the `× 256` off-by-one that
§3.5 flags as this rewrite's one new risk.

### 3.4 `hft_dataset.py` — DONE

Done 2026-07-27, option (b). `HFTDivDataset` deleted; `HFTDataset` is now
map-style over a flat `(div, frame)` index, with all five slabs memory-mapped
and opened lazily per worker (`__getstate__` drops the handles so a spawned
worker cannot pickle gigabytes). `train_hft.py` passes `shuffle=True` and lets
PyTorch shard.

Feature parameters are read from `meta/{split}/dataset_meta{div}.json`, not from
the training config, so there is one source of truth; `margin_b`, `margin_f`,
`num_frame` and `n_bins` from the config are checked against it and raise on
mismatch.

**Two backends, selected by `feature_source`** (added 2026-07-27 on request):

| | `"audio"` (default) | `"legacy_feature"` |
|---|---|---|
| reads | the waveform slab | the pre-rewrite `.npz` spectrogram store |
| augmentation | yes | rejected — no waveform to augment |
| memory | mapped, shared across workers | ~3.5 GB per worker, as before |
| throughput | 413 batches/s | 1055 batches/s |

`legacy_feature` tolerates both division naming schemes (`dataset_idx.npz` and
`dataset_idx000.npz`), so it reads stores written either side of `bdb0e10`. It
keeps the map-style structure, so it gets the sharding and shuffling fixes —
it reproduces the old *data path*, not the old bugs.

**Verified by `scripts/verify_hft_dataset_equivalence.py`:** same length, same
`dataset_idx`, and the spectrogram matches to **1.9e-6** over 300 probes.
Labels are reported rather than asserted, because each backend reads the labels
stored beside its own features — 100.00% / 99.67% / 99.30% / 100.00% agreement
on onset / offset / frames / velocity, which is the `_midi2note` drift in §3.5,
not a dataset bug.

**Verified — returns exactly what the old dataset did.** The old `__getitem__`
arithmetic reimplemented verbatim and fed the same arrays, over 300 probes:
spec matches the June feature slab to **1.9e-6** everywhere it must, and all
four label arrays are **bit-identical including dtype** (0 mismatches).

**Verified — both bugs gone.** On the real DataLoader with `num_workers=4`:
items per worker `[1000, 1000, 1000, 1000]` (was worker 0 doing everything);
three consecutive epochs give three different orders, each a full permutation
(was byte-identical forever).

**Verified — memory bounded.** One train division is 3.58 GB on disk. The old
design loaded all of it per worker: 14.3 GB across four. Mapped, the pages are
file-backed, so they are shared between workers and reclaimable under pressure
rather than four private anonymous copies.

**Verified — it trains.** `fast_dev_run=5` through the real `HFTDataModule` and
`HFT`: five training steps plus a validation pass, all sixteen losses computed.
First time hFT has loaded its data since the `dataset_idx000` rename.

**Throughput: augmentation is free, by a wide margin.** Measured on the real
dataset with 4 workers: **448.9 batches/s** (1,795 items/s). Training runs at
5.7 it/s. Even adding the augmentator's ~5.8 ms/item, the loader stays ~20×
ahead of the GPU. §3.1's worry that augmentation would cost ~20% was based on
the 48 ms mel figure, which is wrong — see §8.

Historical detail, in case it is ever needed:

Read `slab[s·256, +50944)` from the mapped audio slab — the slab already carries
`audio_pad = 1024` of headroom at each end, so this is a flat slice with no edge
handling and no negative index. Augment, mel with `center=False`, transpose.
`item_from_slab()` in `scripts/verify_hft_slab_layout.py` is the reference
implementation of that arithmetic; `item_from_audio()` in the frame-equivalence
script is the same maths without the slab offset.

**The mel transform is already cached** — `inference.mel_transform_hft` is
`lru_cache`d and takes a `center` flag, so the dataset can call
`mel_from_audio(buffer, config, center=False)` directly. Verified to produce
bit-identical output (max|diff| 0.0) to the arithmetic the layout check
validates. Measured single-threaded, which is what a worker gets: constructing
the transform is 1.48 ms against a 1.88 ms forward, so caching nearly halves the
per-item mel cost. Real, but not the disaster an earlier note here implied.

Also read `meta/{split}/dataset_meta{div}.json` and check `hop_sample`,
`fft_bins`, `audio_pad` and `total_num_frame` against what the dataset assumes,
rather than hardcoding them a second time.

Seeding: the slab concatenates tracks, so the dataset knows a global frame index
but not a track name. Seed on `(split, div, global_frame, epoch)` — a stable
excerpt identity works just as well as a real track id.

### 3.5 Re-extract and verify

**Two pre-existing problems found on 2026-07-27, neither caused by the rewrite,
both of which change what verification is possible here:**

1. **The data in `data/hft_maps` has stale labels.** It was extracted
   2026-06-11. Commit `7724d29` (2026-06-25) replaced `_midi2note`'s pretty_midi
   implementation with Sony's mido one, which computes note times slightly
   differently; the ~1e-16 differences flip frame-boundary roundings. Measured on
   `MAPS_MUS-bk_xmas1_ENSTDkAm`: **1.79% of `label_frames` cells** and 0.43% of
   `label_offset` differ from what current code produces. Verified this is not
   the audio rewrite: HEAD's label code and the rewritten file produce
   byte-identical labels on all four arrays and on `notes`.
2. **The `save_dir` checkpoints no longer load.** `HFT.load_from_checkpoint`
   fails with `Unexpected key(s) in state_dict: hft_encoder.scale_freq,
   ...self_attn.scale` — the architecture changed after they were written.
   `hft.py` is untouched by this work.

Together these kill the planned "confirm the loss curve starts where the current
one does" check: the old checkpoints cannot be loaded, and even if they could,
they were trained on different labels. **Verification has to be the exact
old-vs-new item comparison instead** — which is stronger anyway, and is what
`scripts/verify_hft_slab_layout.py` already does for the feature side.

- ~~Re-extract~~ **DONE 2026-07-27.** Full MAPS re-extracted into
  `data/hft_maps` in ~8 min. Purely additive — the June `feature/` tree and the
  pre-rename `dataset_*.npz` files were untouched, so the old data is still
  there to compare against (and to delete once step 2 is working; `feature/` is
  17 GB).

  | split | tracks | frames | idx entries | audio |
  |---|---|---|---|---|
  | train | 139 | 1,881,730 | 1,859,597 | 8.36 h |
  | val | 71 | 1,243,823 | 1,232,502 | 5.53 h |
  | test | 60 | 990,987 | 981,415 | 4.40 h |

  All three splits pass 7 structural checks each (slab length, idx bounds, last
  sample read in range, and shape/dtype of all four label arrays). No
  MIDI-outlives-audio warnings fired, confirming 0/270 tracks hit that case.

  **Regression check against the June feature slab, on the real artefact** (not
  a subset): `dataset_idx` is byte-identical, 1,859,597 entries — same tracks,
  same order, same layout. Across 892 probes spanning all 139 train tracks, 417
  of them touching a track boundary, every frame that must match does so to
  **1.9e-6** (float32 noise). The 8 frames after each track end differ as
  designed. So the re-extracted audio reproduces exactly the feature the
  previous checkpoints were trained on.
- **Rerun `scripts/verify_hft_frame_equivalence.py` against the new data.** It
  currently validates the maths; pointed at the new slab it also validates the
  layout, catching an off-by-one in the `× 256` offset conversion. That is the
  one new risk this rewrite introduces and is not yet covered.
- Short training run; confirm it trains at all and the loss falls. It cannot be
  compared against the existing curve — see above.

**Known residual: resolved.** float32 costs exactly what the mel it replaces
cost (1024 B/frame both ways), so there is no disk saving to trade away and the
baseline stays bit-identical. See §3.3.

**For the writeup:** hFT items are 3.07 s vs Edwards' ~10 s examples. Still
per-example and correct, but a reverb tail occupies a larger fraction of a 3 s
excerpt, so the effective strength of the reverb stage is not identical to
theirs. Worth one sentence.

---

## 4. OAF — augments nothing today

`dataset_oaf.py` returns `audio`, but `oaf.py:214` unpacks it and never uses it;
the model trains on the pre-computed `feature`. Wiring the augmentator in as-is
would damage an array that gets thrown away.

**Decision made:** augment the audio and recompute the mel in `__getitem__`.
OAF is much better placed than hFT — `oaf_mode.py:777` already saves the
per-segment waveform, so **no re-extraction is needed**.

**Cons, in order of how much they matter:**

1. **The stored recipe has a quirk that must be reproduced, not fixed.**
   `handcrafted_features.py:106-121` calls `librosa.feature.melspectrogram`
   (which already returns power, `power=2.0`) and then squares it *again*, so
   the stored feature is `log(power²)` — double the usual dB dynamic range. Your
   checkpoints were trained on that, so it is the correct target. Anyone
   "cleaning it up" would silently halve the input contrast.
   **Mitigation: call `compute_mel` directly. Do not reimplement.**
2. **Frame count must land exactly.** `oaf_mode.py:760` asserts
   `feature.shape[0] == label_frames.shape[0]` at extraction. Off by one frame is
   32 ms at 31.25 fps — under the 50 ms tolerance, so silent.
   **Mitigation: assert the same thing in `__getitem__`.**
   Note `compute_mel` derives `hop_length` two different ways (line 101-104)
   depending on whether it is passed; extraction passes it explicitly from
   `feature_params` (default 512), and `HandcraftedFeatures` must be built with
   the same `window_size=config["max_frame_secs"]` and `frame_rate`. Those live
   in the extraction config, so the dataset needs access — this is most of the
   ~40 lines.
3. **CPU cost is negligible.** Measured: 2.6 ms for a 5 s segment, 5.6 ms for
   20 s, against ~77 ms of augmentation. ~7% overhead.
4. **Dead weight.** The unused `feature` array stays in every npz. Harmless, but
   leave a comment saying which is live.

---

## 5. kong / oafv2 / hpp — straightforward wiring

All three already load a mono float32 waveform and compute their own front end,
so they just need the augmentator called in `__getitem__` with
`(track_id, excerpt_start, epoch)`, plus epoch plumbing per model.

### kong dataloader — fixed 2026-07-27, 54× faster

Found while sizing the reproduction run. `kong_dataset` was costing **98 ms per
item** before any augmentation, which would have made the loader the bottleneck
by a wide margin. Two independent problems, both now fixed, labels verified
**bit-identical** throughout.

**1. The MIDI was re-parsed for every excerpt.** `__getitem__` called
`pretty_midi.PrettyMIDI(midi_path)` — a full parse of a ~10 min performance,
median 6,625 notes, **63 ms** — to label a single 10 s window. With
`hop_size=1.0` each file yields ~600 excerpts, so every file was parsed ~600
times per epoch.

An LRU cache does not fix this: training shuffles globally over 962 files, so
the hit rate is `cache_size / n_files`. 64 entries buys 6.6% at 518 MB per
worker; a useful rate needs ~7.8 GB per worker.

Fixed by parsing once at construction into the minimum `NoteSeg` reads — note
tuples and pre-paired sustain events as numpy arrays. **0.192 MB per
performance against 8.1 MB for a parsed PrettyMIDI, 42× smaller**, so all 962
train files fit in **185 MB**. Built in the parent before the fork, and numpy
buffers rather than Python objects, so workers share it instead of each
refcounting their way into a private copy. The sustain on/off pairing moved to
parse time as well — it scans the whole performance and does not depend on the
segment.

**2. `_get_reg` was 95% of what remained.** Three nested Python loops per pitch
per roll, 178 calls an item. The loops implement a nearest-event assignment with
the boundary at `floor((a + b) / 2)`, which is one `searchsorted` over the
midpoints. **35.1 ms → 1.8 ms.** Verified bit-identical over 213 cases including
200 fuzzed rolls and the degenerate ones the loops handled implicitly — no
events, single event, events at frame 0 and the last frame, adjacent events.

| | parse | labels | total | 4 workers |
|---|---|---|---|---|
| original | 63.0 ms | 35.1 ms | 98.1 ms | 1.3 batches/s |
| + MIDI store | 0 | 35.1 ms | 35.1 ms | 3.6 batches/s |
| + `_get_reg` | 0 | **1.8 ms** | **1.8 ms** | **69.4 batches/s** |

28,000 steps of dataloading: **6.10 h → 0.11 h**.

**Augmentation is now the constraint for kong**, unlike hFT where it is free.
Measured single-threaded on a 10 s / 16 kHz excerpt: **21.0 ms** against 1.8 ms
of labels. At 4 workers that is 5.5 batches/s and ~1.4 h for a 28k-step run;
8 workers halves it. Worth setting `num_workers` accordingly rather than leaving
it at 4.

---

**Do `kong` first.** Edwards used Kong's architecture with ~10 s training
examples, and `extract_kong` already defaults to `window_size=10.0`. So kong +
augmentator is a **direct reproduction of the paper**, with a published number
to check against: **88.4 F1 note-onset on MAPS**. If it lands near that, the
whole pipeline — asset substitution, manifests, alignment, seeding — is
validated end to end. If not, we know something is wrong before building on top
of it. This is the highest-information run available.

---

## 6. Reverb level jump — RESOLVED 2026-07-27

**Implemented as `reverb_level="peak" | "rms"`, default `"peak"`.** Threaded
through `Augmentator` and `train_hft`. `last_applied()` now reports `level` and
the realized `rms_gain`, so the effect is measurable instead of invisible.

### The numbers were wrong in the original note

Measured on 300 real hFT items, not a synthetic tone:

| | `"peak"` (faithful) | `"rms"` (Kaldi-style) |
|---|---|---|
| loudness gap, reverb on − off | **+3.476 nats** (+15.2 dB power) | −0.483 nats |
| AUC predicting "reverb fired" from loudness | **0.929** | 0.417 (chance) |
| realized RMS gain | median **8.38×**, range **1.68–939.98×** | 1.00 exactly |
| room effect on the spectrogram | 3.842 nats | 1.485 nats |

Three corrections to what was written here before:

- It is **+15.2 dB**, not +8.2 dB.
- It is **not a flat offset**. Normalization targets a fixed peak of 0.5, so the
  shift depends on the excerpt's own level — sd 1.72 nats.
- It does **not** "perfectly predict" the stage. AUC 0.929, not ~1.0.

### Two things the measurement added

**61% of what the reverb stage does is level, not room.** The same convolutions
with the same IRs perturb the spectrogram by 3.842 nats under `"peak"` and 1.485
under `"rms"`. The stage Edwards credits with +2.8 F1 is majority level shift by
magnitude.

**It also destroys dynamics across excerpts.** The gain range is 1.68× to
**940×** — a quiet passage is amplified a thousandfold to reach the same peak as
a loud one. Every reverberant excerpt leaves the stage at peak 0.5 regardless of
how loud the music was. hFT, OAF and kong all predict **velocity**, which
depends on level, so this is not only a shortcut but a direct corruption of one
of the four prediction targets on ~50% of training excerpts.

### The literature says preserve the level

| implementation | normalizes | level |
|---|---|---|
| Kaldi `wav-reverberate`, Ko et al. 2017 (ICASSP) | output energy → input energy, **default on** | preserved |
| torchaudio augmentation tutorial | RIR → unit L2 norm, no output scaling | preserved on average |
| audiomentations (0.34.1 **and** current `main`) | output peak → 0.5 | **discarded** |

Kaldi's is the exact operation `"rms"` implements: *"scale so that the signal
energy is the same as the original input signal"*,
`input.Scale(sqrt(power_before_reverb / power_after_reverb))`.

So this is not a deviation from Edwards so much as a library default that
diverges from both dominant reference implementations, which Edwards inherited
without comment. The paper says only that the pipeline "is implemented with the
`audiomentations` package" — there is no training code release, so there is
nothing else to check against.

### Plan

Default stays `"peak"` **until the kong reproduction lands**, because that run
has to be faithful to be a reproduction (§5, target 82.4 → 86.4). Then run kong
with `"rms"` as a single-variable ablation against a validated baseline, and
switch the default to whichever wins. A measured deviation with a citation and a
number behind it is reportable; an unjustified one is not.

---

## 6b. Original options table, kept for the record

When reverb fires it peak-normalises the output to 0.5 regardless of input
level. Measured on a fixed-amplitude tone:

- reverb off: **×0.81** RMS
- reverb on: **×6.56** RMS (range ×1.54–×9.86)

In the log-mel domain that is a flat **+8.2 dB offset on every value**, present
on ~50% of excerpts and perfectly predicting the reverb stage. No model here
normalises per-excerpt loudness (hpp's `AmplitudeToDB(top_db=80)` only clamps
the floor). So a model can learn "loud ⇒ reverberant" instead of robustness.

It is audiomentations' own behaviour and therefore what Edwards ran, so keeping
it preserves the reproduction claim.

| option | for | against |
|---|---|---|
| keep, log the ratio in `last_applied()` | faithful; effect becomes measurable | shortcut remains available |
| rescale reverb output to input RMS | removes the shortcut at source | real deviation, must be declared |
| flag, default faithful | turns it into a reportable ablation | two code paths; only pays off if run |
| normalise loudness on all model inputs | fixes every level effect | invalidates existing checkpoints; blast radius too large — **rule out** |

---

## 7a. Reverb pool switched to Edwards' own — 2026-07-27

**Decision: use EchoThief, the library Edwards actually drew from.**

The old pool was assembled from MIT, AachenIR and OPENAIR and filtered to
exclude outdoor spaces, vehicles, bathrooms and pools as "not plausible piano
recording environments", with RT60 capped at 2.0 s. Measured against what the
paper used, that was not a substitution for his distribution but close to its
inverse:

| | old pool | EchoThief (Edwards) |
|---|---|---|
| composition | MIT 100 spaces (bedrooms, kitchens, living rooms), OPENAIR 19, AachenIR 5 | caves, tunnels, stairwells, underpasses, fortresses, glaciers |
| RT60 median | 0.57 s | **1.09 s** |
| RT60 max | 1.93 s (capped) | **3.47 s** |
| over 2 s | 0 | 10 |

Reverb is the stage Edwards' ablation values most (+2.8 F1 alone, −3.6 when
removed), so this was the largest deviation in the pipeline and the one most
likely to explain a miss against his 86.4.

**Two judgement calls, both forced by the paper's silence:**

- *Which 14.* Not identified in the paper, so any 14 of the library's 115 would
  be our choice, not his. All 115 are used: same source, same character, a
  superset of whatever he drew, no arbitrary selection to defend. Over a 28k-step
  run at p=0.5 each space is still drawn thousands of times.
- *No filtering.* The RT60 band and space-type exclusions are deliberately not
  applied — he applied none, and the point is to stop substituting our judgement.
  Worth noting the RT60 cap was guarding against something that mostly does not
  affect the headline metric: reverb does not move onsets, so a long tail blurs
  offsets and masks quiet onsets rather than shifting the onsets note-onset F1
  scores.

**The old pool becomes the held-out robustness set** — 226 IRs over 166 spaces,
disjoint from EchoThief by construction. Better than the split it replaces:
train on the paper's distribution, evaluate on realistic rooms never seen.

**Verified.** EchoThief survives `SpaceUniformImpulseResponse`'s direct-sound
alignment: median onset-to-peak gap 0.31 ms, median DRR change from trimming
+0.23 dB, zero truncated or empty files. More files exceed a 1 ms gap than
before (43/115 vs 17/177) — expected where a reflection outweighs the direct
arrival in a large space — but the DRR effect stays negligible, the same
conclusion §7 reached for the old pool. End to end on real hFT items: reverb
fires at **0.490**, 92 of 115 spaces drawn over 400 items, spectrogram shape and
all four label arrays unchanged.

Old manifests kept as `room_ir_{train,test}.txt.pre_echothief`.

**Noise pool is unchanged and remains a real deviation.** Edwards uses
`restaurant08.wav` plus four freesound recordings — 65 clips of pub and cafe
babble. Ours is TUT-acoustic-scenes-2016, of which **81% of selected clips carry
more than half their energy below 100 Hz**: engine and traffic rumble, only 3%
speech-band dominated. Because `AddBackgroundNoise` sets SNR broadband, the
in-band SNR also runs **+5.2 dB above nominal at the median** (+7.6 dB at p10),
so the stage is gentler than the paper's as well as different in kind. Bounded
at ~1 F1 by his ablation. Cheap partial fix if wanted: raise
`BAND_FRACTION_MIN` from 0.15 to 0.50, which keeps 98 clips — still more than
his 65 — and cuts the SNR gap under 3 dB.

---

## 7. Reverb pool audit — one open question

Prompted by "check the different reverb rooms are not causing problems". Ran a
first pass; two results are clean, one needs follow-up.

### Clean

**RT60 is realistic.** No pathologically long rooms that would smear onsets:

| split | IRs / spaces | RT60 min | median | p95 | max | >2 s |
|---|---|---|---|---|---|---|
| train | 177 / 124 | 0.30 s | 0.57 s | 1.70 s | 1.93 s | 0 |
| test | 49 / 42 | 0.30 s | 0.64 s | 1.67 s | 1.84 s | 0 |

A concert hall is ~2 s, so this sits sensibly below MAESTRO's own acoustic and
nothing is extreme.

**Train/test spaces are disjoint.** 0 shared spaces, so robustness evaluation
measures generalisation to unseen rooms rather than recall of the training pool.
Early/late energy ratio is healthy in both (median ~+12 dB).

### RESOLVED 2026-07-27 — trimming is fine, and the proposed fix would be worse

Both checks run over all 226 IRs.

| | train (177) | test (49) |
|---|---|---|
| pre-peak energy > 1% | 152 | 40 |
| onset→peak gap, median | **0.19 ms** | **0.06 ms** |
| gap p95 / max | 4.56 / 31.94 ms | 0.62 / 2.69 ms |
| gap > 1 ms | 17 | 2 |
| DRR change from trimming, median | **+0.36 dB** | +0.23 dB |
| DRR change, max | +6.12 dB | +4.26 dB |

**Explanation 1 holds.** The median onset→peak gap is three samples at 16 kHz —
`argmax` lands inside the rising edge of a single pulse, exactly as the benign
reading predicted. Only 19 of 226 files have a gap over 1 ms. Envelope profiles
of the worst files show monotonic rises into the peak, e.g.
`air_binaural_office_1_1_1`: −61 −65 −75 −59 −53 −49 −46 −44 −43 −42 −42 −44
−47 −59 −49 −40 −36 −32 −29 −27 −25 −23 −21 −19 −7 −4 −0 dB. One pulse, one
edge.

**The feared mechanism does not occur.** Trimming was predicted to *lower* DRR
and make rooms sound more distant. Measured, it makes them very slightly
**drier** — median +0.36 dB. Opposite sign, negligible magnitude.

**The proposed fix would be a regression.** Replacing `argmax` with a −20 dB
threshold detector fires 511 samples — **31.9 ms** — before the peak on
`aula_carolina_1_1_4`, latching onto scattered −30 to −44 dB energy. That would
shift the IR's effective time zero by 32 ms against a 50 ms tolerance, where §2
established `argmax` gives 0.0 ms error on all 177. The "fix" would inject onset
delay on exactly the files it was meant to repair.

**Decision: change nothing.** Optionally drop the ~19 files with a gap over 1 ms
(8% of the pool, no meaningful loss of diversity), but +0.36 dB median does not
warrant even that.

### Original open question, kept for the record

`SpaceUniformImpulseResponse._load_aligned_ir` trims everything before
`argmax(|ir|)`. Checking how much energy sits *before* that peak:

| split | IRs with >1% pre-peak energy | worst case |
|---|---|---|
| train | **152 / 177** | 56.4% of total energy, peak at 4.7 ms (`MIT/h153_Office_Foyer_1txts.wav`) |
| test | **40 / 49** | 49.9%, peak at 3.4 ms (`AachenIR/office/air_binaural_office_1_1_1.wav`) |

For the worst files, trimming at the peak discards more than half the impulse
response's energy.

**Two competing explanations, and I have not distinguished them:**

1. **Benign.** The direct sound is a bandlimited pulse spread over several ms by
   the 16 kHz downsampling and the sweep measurement. `argmax` lands a little
   into that pulse, and the "pre-peak energy" is its own rising edge. Trimming
   then removes part of one pulse and nothing meaningful.
2. **A real defect.** The true direct arrival is genuinely earlier and quieter
   in peak amplitude than a strong early reflection. Trimming then deletes the
   direct path entirely, keeping a reflection as time zero — which lowers the
   direct-to-reverberant ratio and makes every room sound more distant than it
   is. The reverb stage would still be *timing*-correct but acoustically wrong.

Note this does **not** contradict the alignment result in §2 — that measured
where the output peak lands (0.0 ms for all 177), which is true either way.

**Concrete check to run:**

- Plot `|ir|` around the peak for the five worst files. A single broadband pulse
  with a rising edge means explanation 1; a distinct earlier arrival separated
  by quiet means explanation 2.
- Compute direct-to-reverberant ratio before and after trimming. A large drop
  indicates the direct path is being removed.
- If explanation 2 holds, replace `argmax` with a threshold-based onset detector
  (first sample exceeding e.g. −20 dB relative to the peak) and re-run
  `scripts/analyze_augment_assets.py` plus the click-train alignment check.

Cost if it turns out to matter: Edwards' ablation puts reverb at +2.8 F1 alone
and −3.6 when removed, so a systematically wrong reverb character is worth up to
a few F1 on out-of-domain evaluation. Worth resolving before any long run.

---

## 8. Smaller items

- **The 48 ms mel figure was wrong — resolved.** §3 budgeted 48 ms for an hFT
  item's mel, which is what made augmentation a 20% cost in §3.1. Measured on a
  192-frame buffer, single-threaded: **1.88 ms**, 25× cheaper. Settled
  end-to-end in §3.4: the real loader does 448.9 batches/s against training's
  5.7 it/s. Augmentation is free. Do not quote the 48 ms figure.
- **Throughput.** Measure each model's current it/s before optimising —
  augmentation is free while workers outpace the GPU. Loader ceiling is
  `num_workers ÷ (batch_size × per-excerpt cost)`. Per-excerpt: transkun 99 ms
  (16 s @ 44.1 kHz), oafv2/hpp ~77 ms, OAF ~83 ms, kong ~37 ms, hFT 54 ms.
- **Memory.** The noise cache is per worker: 64 clips × 30 s is ~120 MB at
  16 kHz but **~340 MB at 44.1 kHz** (transkun). Consider changing
  `noise_cache_size` from a clip count to a megabyte budget so it means the same
  thing at every sample rate.
- **Docstring: the 8 kHz claim.** "every model front end caps at or below 8 kHz"
  is true of current config values, not structurally. All assets are 16 kHz, so
  reverb lowpasses above 8 kHz when it fires. Currently harmless (transkun
  `f_max=8000`, kong `sample_rate//2`=8000 at the 16 kHz default, hpp CQT
  ~4.4 kHz), but raise kong's extraction sample rate and it silently becomes
  false. Reword to name the configs it depends on.
- **Docstring: noise pool substitution.** Edwards' noise is specifically pub,
  cafe and restaurant recordings — human babble, a narrow distribution. Ours is
  TUT-acoustic-scenes-2016, which is much broader. The docstring says the
  filtering makes our pools "behave like his"; for noise it is a different
  *kind* of sound. Low stakes (the ablation puts background noise at ~1 F1) but
  worth one honest sentence.
- **EQ clipping.** +5 dB on a 0.95-peak excerpt gave peaks to 1.59 (10/100 over
  1.0). Fine in float; only matters if anything downstream converts to int16.
  One grep to confirm nothing does, then ignore.
- **hFT n_slice — worth changing, and §3.2 is the moment to do it.**
  `n_slice=16` takes every 16th index, always the *same* every 16th. So of
  1,859,597 possible train excerpts, hFT only ever sees 116,224 — the other
  15/16 are unreachable for the entire run. That compounded with the frozen
  shuffle (fixed now) but survives it: the stride is still fixed.

  Rotating the offset per epoch — `idx[epoch % n_slice :: n_slice]` — gives
  access to all of them across epochs, keeps the epoch length identical, and
  costs nothing. §3.2 introduces `dataset.epoch` anyway, so the mechanism is
  already there.

  Caveat: it is a deviation from hFT's original recipe and it changes the
  baseline as well as the augmented run, so apply it to both or it confounds the
  comparison. Treat as a deliberate, declared change, not a silent one.

---

## 9. Reference numbers

From the paper, for checking our reproduction:

| | MAESTRO | Studio MAESTRO |
|---|---|---|
| full augmentation | 86.4 | 79.0 |
| skip background | 85.4 (−1.0) | 78.8 (−0.2) |
| skip pitch shift | 82.9 (**−3.5**) | 75.6 (**−3.4**) |
| skip reverb | 82.8 (**−3.6**) | 77.1 (−1.9) |
| skip EQ | 86.4 (−0.0) | 77.8 (−1.2) |

Single augmentations alone, MAPS note-onset F1: pitch shift +3.1, reverb +2.8,
background +0.3, EQ −0.3. Their headline: **88.4 F1 on MAPS**.

Pitch shift and reverb carry the pipeline. A useless noise pool costs about one
F1; a useless IR pool about three.
