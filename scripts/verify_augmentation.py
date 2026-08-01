"""Check that augmentation is wired correctly into every model, and breaks none.

Run this before committing GPU-days to an augmented run.  A broken augmentation
pipeline does not usually crash -- it trains, the loss looks plausible, and the
result is merely disappointing months later -- so each model is checked against
the invariants that would otherwise fail silently:

  clean-path      with no augmentator the item is bit-identical to what the
                  model saw before augmentation existed.  This is the one that
                  guarantees an augmented-vs-clean comparison is controlled.
  applied         augmentation actually reaches what the model consumes.  For
                  five models that is the waveform; for OAF it is the feature
                  recomputed from the augmented waveform, since OAF trains on a
                  feature written at extraction time and augmenting its audio
                  alone would change nothing.
  labels-intact   every label array is bit-identical, augmented or not.  The
                  pipeline is label-preserving by construction -- the +/-10 cent
                  pitch shift is tuning jitter, and impulse responses are
                  aligned to their direct sound so reverb adds no delay -- and
                  this is what catches a regression in that.
  reproducible    the same excerpt in the same epoch draws the same
                  augmentation, whatever the batch size or worker count.  That
                  is what lets two architectures be compared on equal terms.
  redrawn         a different epoch draws differently.  Without this the run
                  trains on one fixed pre-corrupted copy of the dataset, which
                  is the failure the epoch plumbing exists to prevent.
  finite          no NaN or inf reaches the model.
  epoch-rollover  three epochs through a real worker-backed loader keep
                  delivering different items.  Two training steps never cross an
                  epoch boundary, so this is where a frozen draw shows up --
                  whether frozen because the epoch never reached the workers or
                  because the excerpt itself was pinned.
  coverage        for the datasets that choose an excerpt per item, several
                  epochs must reach several different excerpts.  Measured
                  unaugmented, since augmented audio differs every epoch even
                  when the underlying excerpt does not -- which is precisely how
                  a frozen crop went unnoticed.
  trains          two real optimizer steps through the public Trainer API with
                  augment=True: the actual datamodule, callbacks, collate and
                  loss, not a reimplementation.

Usage:
    python scripts/verify_augmentation.py            # every model
    python scripts/verify_augmentation.py oaf kong   # a subset
    python scripts/verify_augmentation.py --no-train # skip the training steps
"""

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ASSETS = "/home/simone-chieppa/Desktop/mtg_ir_datasets"

# The smoke stores hold a handful of short MAESTRO tracks; kong and hft use the
# real extractions already on disk.
STORES = {
    "oaf": "data/smoke_oaf",
    "oafv2": "data/smoke_oafv2",
    "hpp": "data/smoke_hpp",
    "transkun": "data/smoke_transkun",
    "kong": "data/kong_maestro",
    "hft": "data/hft_maps",
}

# hft's trainer is epoch-driven with no step cap, and one epoch over the real
# MAPS store runs for tens of minutes. Its dataset checks stay on that store --
# real data is the point -- while the two training steps use a small store
# extracted from the same mini-MAESTRO as the others.
TRAIN_STORES = {"hft": "data/smoke_hft"}

OAF_FEATURE_CONFIG = dict(feature="mel", sample_rate=16000, frame_rate=31.25,
                          max_frame_secs=20.0, n_mels=229, mel_n_fft=2048,
                          hop_length=512)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Results:
    """Collects one row per (model, check) so the summary can be read at once."""

    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, model: str, check: str, status: str, detail: str = "") -> None:
        self.rows.append((model, check, status, detail))
        mark = {PASS: "ok", FAIL: "FAIL", SKIP: "skip"}[status]
        print(f"  [{mark:>4}] {check:<14} {detail}")

    def failed(self) -> list[tuple[str, str, str, str]]:
        return [r for r in self.rows if r[2] == FAIL]


# oafv2 and hpp draw one excerpt per track per epoch; the rest enumerate every
# excerpt by index, so "which excerpt" is not a choice they make and the
# coverage check does not apply to them.
CROPPING = {"oafv2", "hpp"}


# Every model is verified at reverb_level="rms", which is what the trainers
# default to.  The one deliberate "peak" left in this file is the stereo-guard
# below, which asserts that multi-channel datasets refuse it.
def make_augmentator(sample_rate: int, seed: int = 1234):
    from snap2midi.utils.augmentator import Augmentator
    return Augmentator(sample_rate=sample_rate, asset_root=ASSETS, split="train",
                       seed=seed, reverb_level="rms")


# --------------------------------------------------------------------------
# Per-model adapters.
#
# Each builds a dataset and knows how to pull the two things that matter out of
# an item: what the model actually consumes, and every label array.  Each is
# called once per model and the resulting dataset is re-read through Probe,
# which swaps the augmentator around it -- constructing is the expensive part,
# since KongDataset parses every MAESTRO MIDI up front.
# --------------------------------------------------------------------------

def adapter_oaf(augmentator):
    from snap2midi.models.oaf.dataset_oaf import OAFDataset
    ds = OAFDataset([f"{STORES['oaf']}/train/"], augmentator=augmentator,
                    feature_config=OAF_FEATURE_CONFIG)
    # index 0 is the feature: what forward() is given.  audio (index 5) is
    # carried in the batch but never consumed, which is the whole reason OAF
    # needs the feature recomputed rather than just the audio augmented.
    return ds, (lambda it: it[0]), (lambda it: [it[i] for i in (1, 2, 3, 4)])


def adapter_oafv2(augmentator):
    from snap2midi.models.oafv2.dataset_oafv2 import OAFV2Dataset
    cfg = dict(sequence_length=327680, hop_length=512, seed=42, sample_rate=16000)
    ds = OAFV2Dataset(cfg, [f"{STORES['oafv2']}/train/"], augmentator=augmentator)
    return ds, (lambda it: it["audio"]), (lambda it: [it[k] for k in ("label", "onset", "offset", "frame", "velocity")])


def adapter_hpp(augmentator):
    from snap2midi.models.hpp.dataset_hpp import HPPDataset
    cfg = dict(sequence_length=327680, hop_length=320, seed=42, sample_rate=16000)
    ds = HPPDataset(cfg, [f"{STORES['hpp']}/train/"], augmentator=augmentator)
    return ds, (lambda it: it["audio"]), (lambda it: [it[k] for k in ("label", "onset", "offset", "frame", "velocity")])


def adapter_kong(augmentator):
    from snap2midi.models.kong.kong_dataset import KongDataset
    ds = KongDataset(f"{STORES['kong']}/train/", extend_pedal=True, augmentator=augmentator)
    labels = lambda it: [v for k, v in sorted(it.items())
                         if k != "audio" and isinstance(v, np.ndarray)]
    return ds, (lambda it: it["audio"]), labels


def adapter_hft(augmentator):
    from snap2midi.models.hft.hft_dataset import HFTDataset
    import json
    meta = json.loads((Path(STORES["hft"]) / "meta" / "train.json").read_text()) \
        if (Path(STORES["hft"]) / "meta" / "train.json").exists() else {}
    cfg = dict(base_path=STORES["hft"], n_div_train=1, n_div_val=1, n_div_test=1,
               margin_b=32, margin_f=32, num_frame=128, n_bins=meta.get("mel_bins", 256),
               n_slice=16, feature_source="audio")
    for key in ("sample_rate", "fft_bins", "window_length", "hop_sample", "mel_bins",
                "log_offset", "pad_mode"):
        if key in meta:
            cfg[key] = meta[key]
    ds = HFTDataset(cfg, split="train", augmentator=augmentator)
    return ds, (lambda it: it[0]), (lambda it: list(it[1:]))


def adapter_transkun(augmentator):
    from snap2midi.models.transkun.transkun_dataset import TranskunDataset
    import json
    # Read conf.json directly rather than through moduleconf, which imports the
    # model itself. The dataset checks need two numbers, not a built network.
    conf = json.loads((REPO_ROOT / "snap2midi/models/transkun/conf.json").read_text())
    conf = conf["Model"]["config"]
    ds = TranskunDataset(f"{STORES['transkun']}/train/", 44100,
                         conf["segmentHopSizeInSecond"], conf["segmentSizeInSecond"],
                         augmentator=augmentator)
    ds.build_chunks(1234, epoch=0)

    # `notes` is a list of Note objects, which define no __eq__ -- comparing
    # them directly tests object identity and so reports every read as
    # different, augmented or not. Compare the fields the labels actually
    # consist of instead.
    note_fields = ("start", "end", "pitch", "velocity", "hasOnset", "hasOffset")

    def labels(it):
        notes = np.array([[float(getattr(n, f)) for f in note_fields]
                          for n in it["notes"]], dtype=np.float64)
        return [notes, np.asarray(it["begin"]), np.asarray(it["fs"])]

    return ds, (lambda it: it["audioSlice"]), labels


ADAPTERS = {"oaf": adapter_oaf, "oafv2": adapter_oafv2, "hpp": adapter_hpp,
            "kong": adapter_kong, "hft": adapter_hft, "transkun": adapter_transkun}

SAMPLE_RATES = {"oaf": 16000, "oafv2": 16000, "hpp": 16000, "kong": 16000,
                "hft": 16000, "transkun": 44100}


def as_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Probe:
    """One dataset per model, re-read under different augmentation settings.

    Constructing is the expensive part -- KongDataset parses every MAESTRO MIDI
    up front -- so the dataset is built once and the augmentator swapped in and
    out around it, which is exactly the knob the datamodule turns anyway.

    Reads at a fixed (index, epoch) are deterministic for every model, including
    the two that choose their own excerpt: oafv2 and hpp derive the crop from
    (seed, track, epoch) rather than from generator state, so re-reading does
    not silently return a different window.
    """

    def __init__(self, model: str):
        self.dataset, self.get_input, self.get_labels = ADAPTERS[model](None)

    def read(self, augmentator=None, epoch: int = 0, index: int = 0):
        self.dataset.augmentator = augmentator
        self.dataset.epoch = epoch
        item = self.dataset[index]
        return (as_array(self.get_input(item)),
                [as_array(a) for a in self.get_labels(item)])


def signature(array: np.ndarray) -> str:
    """A short stable digest of one delivered item, for comparing across epochs."""
    return hashlib.blake2b(np.ascontiguousarray(array).tobytes(),
                           digest_size=8).hexdigest()


def epoch_signatures(probe: Probe, augmentator, epochs: int, items: int,
                     workers: int = 2) -> list[list[str]]:
    """Read the first few items of each epoch through a real worker-backed loader.

    The point is to exercise the fork, not just the dataset: the epoch reaches
    a worker only because `persistent_workers=False` makes them re-fork, and a
    two-step training run never crosses an epoch boundary to find out. An
    identity collate keeps each model's own item structure intact so one
    routine serves all six.
    """
    from torch.utils.data import DataLoader

    dataset = probe.dataset
    dataset.augmentator = augmentator
    out: list[list[str]] = []
    for epoch in range(epochs):
        dataset.epoch = epoch
        if hasattr(dataset, "build_chunks"):
            # transkun carries the epoch through build_chunks, which the trainer
            # re-runs with a seed that moves each epoch.
            dataset.build_chunks(1234 + 100 * epoch, epoch=epoch)
        loader = DataLoader(dataset, batch_size=1, num_workers=workers,
                            persistent_workers=False, collate_fn=lambda b: b)
        seen = []
        for i, batch in enumerate(loader):
            if i >= items:
                break
            seen.append(signature(as_array(probe.get_input(batch[0]))))
        out.append(seen)
    return out


def check_dataset(model: str, results: Results) -> Probe:
    sample_rate = SAMPLE_RATES[model]
    aug = make_augmentator(sample_rate)
    probe = Probe(model)

    clean_input, clean_labels = probe.read(None)
    aug_input, aug_labels = probe.read(aug)

    # clean-path: no augmentator means the stored item, untouched.
    clean_again, _ = probe.read(None)
    if np.array_equal(clean_input, clean_again):
        results.add(model, "clean-path", PASS, "unaugmented item is unchanged")
    else:
        results.add(model, "clean-path", FAIL, "unaugmented item is not reproducible")

    # applied: augmentation reaches what the model consumes.
    if clean_input.shape != aug_input.shape:
        results.add(model, "applied", FAIL,
                    f"shape changed {clean_input.shape} -> {aug_input.shape}")
    elif np.array_equal(clean_input, aug_input):
        results.add(model, "applied", FAIL, "model input is identical to clean")
    else:
        rel = np.abs(aug_input - clean_input).mean() / (np.abs(clean_input).mean() or 1.0)
        results.add(model, "applied", PASS,
                    f"input {clean_input.shape} changed, mean |delta| {rel:.1%} of scale")

    # labels-intact: every label array bit-identical.
    if len(clean_labels) != len(aug_labels):
        results.add(model, "labels-intact", FAIL, "different number of label arrays")
    else:
        bad = [i for i, (a, b) in enumerate(zip(clean_labels, aug_labels))
               if not np.array_equal(a, b)]
        if bad:
            results.add(model, "labels-intact", FAIL, f"label arrays {bad} differ")
        else:
            results.add(model, "labels-intact", PASS,
                        f"{len(clean_labels)} label arrays bit-identical")

    # reproducible: same excerpt, same epoch, fresh augmentator -> same draw.
    repeat_input, _ = probe.read(make_augmentator(sample_rate), epoch=0)
    if np.array_equal(aug_input, repeat_input):
        results.add(model, "reproducible", PASS, "same excerpt+epoch gives the same draw")
    else:
        results.add(model, "reproducible", FAIL, "same excerpt+epoch drew differently")

    # redrawn: a later epoch must differ, or the run trains on one fixed copy.
    later_input, _ = probe.read(make_augmentator(sample_rate), epoch=7)
    if np.array_equal(aug_input, later_input):
        results.add(model, "redrawn", FAIL, "epoch 7 identical to epoch 0")
    else:
        results.add(model, "redrawn", PASS, "epoch 7 differs from epoch 0")

    # finite
    if np.isfinite(aug_input).all():
        peak = float(np.max(np.abs(aug_input)))
        results.add(model, "finite", PASS, f"no NaN/inf, peak |x| {peak:.3f}")
    else:
        results.add(model, "finite", FAIL, "NaN or inf in the augmented input")

    # stereo-guard: multi-channel audio must refuse the per-channel peak
    # normalization rather than silently randomize the left/right balance.
    # transkun's store is the stereo one: "peak" normalizes each channel on its
    # own and so re-rolls the left/right balance on every excerpt the reverb
    # fires on, which is why "rms" is the only safe level there and why the
    # dataset rejects "peak" outright.
    if hasattr(probe.dataset, "_check_stereo_safe") and aug_input.ndim > 1 \
            and aug_input.shape[-1] > 1:
        from snap2midi.utils.augmentator import Augmentator
        peak_aug = Augmentator(sample_rate=sample_rate, asset_root=ASSETS,
                               split="train", seed=1234, reverb_level="peak")
        probe.dataset._stereo_checked = False
        try:
            probe.read(peak_aug)
            results.add(model, "stereo-guard", FAIL,
                        "reverb_level='peak' accepted on multi-channel audio")
        except ValueError:
            results.add(model, "stereo-guard", PASS,
                        "reverb_level='peak' refused on multi-channel audio")
        finally:
            probe.dataset._stereo_checked = False

    return probe


def check_epochs(model: str, probe: Probe, results: Results) -> None:
    """Across three epochs of a worker-backed loader, items must keep changing.

    This is the check the two-step training runs cannot make. Both known ways
    for a run to quietly train on one fixed dataset show up here: an epoch that
    never reaches the workers freezes the augmentation draw, and a crop drawn
    from generator state freezes the excerpt.
    """
    aug = make_augmentator(SAMPLE_RATES[model])
    sigs = epoch_signatures(probe, aug, epochs=3, items=3)

    frozen = [i for i in range(1, len(sigs)) if sigs[i] == sigs[0]]
    if frozen:
        results.add(model, "epoch-rollover", FAIL,
                    f"epochs {frozen} deliver items identical to epoch 0")
    else:
        results.add(model, "epoch-rollover", PASS,
                    "3 epochs through a worker-backed loader all differ")


def check_coverage(model: str, probe: Probe, results: Results) -> None:
    """Over several epochs, how much of the corpus does the model actually see?

    Only meaningful where the dataset chooses an excerpt per item. The others
    enumerate every excerpt by index, so item i is the same audio each epoch by
    design and their coverage comes from the size of an epoch instead.

    Measured with augmentation off: augmented audio differs every epoch even
    when the underlying excerpt does not, which is exactly how a frozen crop
    stayed invisible.
    """
    if model not in CROPPING:
        results.add(model, "coverage", SKIP,
                    "enumerates excerpts by index; coverage comes from epoch size")
        return

    epochs, items = 4, 3
    sigs = epoch_signatures(probe, None, epochs=epochs, items=items)
    distinct = len({s for epoch in sigs for s in epoch})
    total = epochs * items

    # A frozen crop yields one distinct excerpt per item however many epochs run.
    if distinct <= items:
        results.add(model, "coverage", FAIL,
                    f"only {distinct} distinct excerpts over {total} reads -- "
                    "the crop is frozen across epochs")
    else:
        results.add(model, "coverage", PASS,
                    f"{distinct}/{total} distinct excerpts over {epochs} epochs")


# --------------------------------------------------------------------------
# Two real optimizer steps through the public Trainer API.
# --------------------------------------------------------------------------

TRAIN_CALLS = {
    "oaf": lambda t, save: t.train_oaf(
        base_path=STORES["oaf"], batch_size=2, iterations=2, num_workers=2,
        save_dir=save, logger_name="csv", augment=True, augment_asset_root=ASSETS),
    "oafv2": lambda t, save: t.train_oafv2(
        base_path=STORES["oafv2"], batch_size=2, iterations=2, num_workers=2,
        save_dir=save, logger_name="csv", augment=True, augment_asset_root=ASSETS),
    "hpp": lambda t, save: t.train_hpp(
        base_path=STORES["hpp"], batch_size=1, iterations=2, num_workers=2,
        model_type="sp", save_dir=save, logger_name="csv",
        augment=True, augment_asset_root=ASSETS),
    # epochs=1 rather than a step count: hft's trainer is epoch-driven, and the
    # store is small enough that one pass is short.
    "hft": lambda t, save: t.train_hft(
        base_path=TRAIN_STORES["hft"], batch_size=2, epochs=1, num_workers=2,
        save_dir=save, logger_name="csv",
        augment=True, augment_asset_root=ASSETS),
    "kong": lambda t, save: t.train_kong(
        base_path=STORES["kong"], batch_size=2, iterations=2, num_workers=2,
        val_steps=2, save_dir=save, logger_name="csv",
        augment=True, augment_asset_root=ASSETS),
    # reverb_level="rms" is what the trainers now default to, so passing it is
    # redundant; it stays explicit because here it is mandatory rather than a
    # preference -- the store is stereo and "peak" normalizes per channel.
    # num_workers matters too --
    # measured, the GPU eats 18.3 items/s at batch 4 while an augmented worker
    # supplies 4.2, so anything under 5 workers starves it.
    "transkun": lambda t, save: t.train_transkun(
        base_path=STORES["transkun"], batch_size=1, epochs=1, num_workers=8,
        val_steps=2, seed=1234, save_dir=save, logger_name="csv",
        augment=True, augment_asset_root=ASSETS, reverb_level="rms"),
}


def check_training(model: str, results: Results) -> None:
    if model not in TRAIN_CALLS:
        results.add(model, "trains", SKIP, "no short-run recipe for this model")
        return

    import snap2midi as s2m
    save = str(Path(os.environ.get("SCRATCH", "/tmp")) / f"verify_aug_{model}")
    try:
        TRAIN_CALLS[model](s2m.trainer.Trainer(), save)
        results.add(model, "trains", PASS, "two augmented optimizer steps completed")
    except Exception as exc:
        results.add(model, "trains", FAIL, f"{type(exc).__name__}: {str(exc)[:110]}")
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=list(ADAPTERS),
                        help="models to check (default: all)")
    parser.add_argument("--no-train", action="store_true",
                        help="skip the two-step training runs")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    results = Results()
    for model in args.models:
        if model not in ADAPTERS:
            print(f"unknown model {model!r}; known: {', '.join(ADAPTERS)}")
            return 2
        print(f"\n=== {model} ({STORES[model]}) ===")
        if not Path(STORES[model]).exists():
            results.add(model, "store", SKIP, f"{STORES[model]} not extracted")
            continue
        try:
            probe = check_dataset(model, results)
            check_epochs(model, probe, results)
            check_coverage(model, probe, results)
        except Exception as exc:
            results.add(model, "dataset", FAIL, f"{type(exc).__name__}: {str(exc)[:110]}")
            traceback.print_exc()
        if not args.no_train:
            check_training(model, results)

    print(f"\n{'='*74}\nSummary\n{'='*74}")
    counts = {s: sum(1 for r in results.rows if r[2] == s) for s in (PASS, FAIL, SKIP)}
    for model, check, status, detail in results.rows:
        if status != PASS:
            print(f"  {status:<4} {model:<10} {check:<14} {detail}")
    print(f"\n{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")
    return 1 if results.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
