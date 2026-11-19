#!/usr/bin/env python
"""Feature stores for the five Snap2MIDI models, one per (model, dataset).

OAF, OAFv2, HPP and Kong are trained and evaluated from these stores; hFT
uses them for MAESTRO pretraining and GAPS fine-tuning (it is evaluated from
the audio and MIDI files directly, like TransKun, which needs no store).
Each store is <stores>/<model>_<dataset>.

Every model has one frame grid, the one of its paper, used for every dataset;
only the sustain pedal changes between datasets (extended into note
durations on the piano sets, which carry CC64; off on the guitar sets, which
have none). Evaluation sets put every item in test/: none of these models
saw them in training, and their own splits exist for training on them.

Datasets:
  maestro         MAESTRO v3, its train/validation/test split  (pretraining, Table 1)
  studio_maestro  the 177 re-recordings of MAESTRO's test split  (Table 1)
  smd, mds        Table 1
  maps            MAPS MUS, its usual split (test: the two Disklavier pianos)
  maps_full       all 270 MAPS MUS pieces; Table 1 keeps the 60 Disklavier ones.
                  OAF extracts them directly; the other three link every piece
                  of their `maps` store (Kong pads its train-split audio with up
                  to a second of silence, so a direct extraction would differ)
  guitarset, goat zero-shot guitar (Table 2); goat holds every DI and reamped take
  gaps_ft         all 297 GAPS tracks, from the WAVs of prepare.py, in the study's
                  240/27/30 split (fine-tuning, selection; Tables 2 and 3 on test/)
  goat_di         links to the 170 DI performances of the goat store (Table 3)
  gaps_ft_val     the 27 validation tracks, for checkpoint selection: hFT keeps them
                  per track; OAF holds them whole in test/ (its gaps_ft/val is cut
                  into training windows)

A finished store gets a `.complete` marker. A store that exists without one
is from an interrupted run and is refused, not skipped: an extraction that
died half-way would otherwise be scored as if it were the whole dataset.

Usage:
    python experiments/data/extract.py --model kong --dataset gaps_ft
    python experiments/data/extract.py --model all --dataset all
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
MARKER = ".complete"
MAPS_PIECES = 270

# One grid per model, as in its paper; every dataset uses it.
COMMON = dict(sample_rate=16000, min_pitch=21, max_pitch=108)
GRID = {
    # OAF writes its log-mel at extraction time (no in-model front end), so the
    # whole mel grid is fixed here. onset/offset lengths are in milliseconds.
    "oaf":   dict(COMMON, feature="mel", frame_rate=31.25, n_mels=229, mel_n_fft=2048,
                  hop_length=512, min_frame_secs=5.0, max_frame_secs=20.0,
                  onset_length=32, offset_length=32),
    "oafv2": dict(COMMON, hop_length=512, onset_frame_length=1, offset_frame_length=1),
    "hpp":   dict(COMMON, hop_length=320, onset_frame_length=1, offset_frame_length=1),
    "kong":  dict(COMMON, hop_length=160, frame_rate=100, window_size=10.0, hop_size=1.0,
                  feature="mel", n_mels=229, mel_n_fft=2048),
    "hft":   dict(n_div_train=1, n_div_val=1, n_div_test=1),
}
STORE_MODELS = ("oaf", "oafv2", "hpp", "kong")

# dataset -> (config section, config key, reader, pedal, eval_only, models)
DATASETS = {
    "maestro":        ("datasets", "maestro", "maestro", True, False, STORE_MODELS + ("hft",)),
    "studio_maestro": ("prepared", "studio_maestro", "maestro", True, True, STORE_MODELS),
    "smd":            ("datasets", "smd", "smd", True, True, STORE_MODELS),
    "mds":            ("datasets", "mds", "mds", True, True, STORE_MODELS),
    "maps":           ("datasets", "maps", "maps", True, False, ("oafv2", "hpp", "kong")),
    "maps_full":      ("datasets", "maps", "maps", True, True, ("oaf",)),
    "guitarset":      ("datasets", "guitarset", "guitarset", False, True, STORE_MODELS),
    "goat":           ("datasets", "goat", "goat", False, True, STORE_MODELS),
    "gaps_ft":        ("prepared", "gaps", "gaps_ft", False, False, STORE_MODELS + ("hft",)),
    "gaps_ft_val":    ("prepared", "gaps", "gaps_ft", False, False, ("oaf", "hft")),
}
DERIVED = {"goat_di": STORE_MODELS, "maps_full": ("oafv2", "hpp", "kong")}
# How one GOAT item is named in each model's store (OAF numbers its windows,
# and an evaluation item is one whole-file window, _0).
GOAT_ITEM = {"oaf": "{item}_0.npz", "oafv2": "{item}.npz", "hpp": "{item}.npz", "kong": "{item}.h5"}


def extract(model, dataset, cfg, extractor):
    section, key, reader, pedal, eval_only, _ = DATASETS[dataset]
    store = cfg["stores"] / f"{model}_{dataset}"
    kwargs = dict(GRID[model], dataset_name=reader, save_name=str(store), extend_pedal=pedal)
    if model in STORE_MODELS:
        kwargs["eval_only"] = eval_only
    if dataset == "gaps_ft_val" and model == "oaf":
        # OAF cuts a training split into 5-20 s windows, so its gaps_ft/val
        # cannot be scored against the MIDI; this store holds the 27
        # validation tracks whole, in test/, as for any evaluation set.
        with tempfile.TemporaryDirectory(dir=cfg["stores"]) as tmp:
            for split in ("train", "val", "test"):
                (Path(tmp) / split).mkdir()
            for f in sorted((cfg[section][key] / "val").iterdir()):
                (Path(tmp) / "val" / f.name).symlink_to(f.resolve())
            extractor.extract_oaf(tmp, **dict(kwargs, eval_only=True))
        return
    if dataset == "gaps_ft_val":
        # The 27 validation tracks only, each kept as its own file beside the
        # collated slab, so checkpoints can be scored per track.
        kwargs.update(splits=["val"], keep_val_npz=True)
    getattr(extractor, f"extract_{model}")(str(cfg[section][key]), **kwargs)


def link_store(model, dataset, cfg, files):
    """A store of links: every file of `files` into <model>_<dataset>/test/.
    Kong reads the grid it was extracted on from a sidecar at the store root,
    so that is linked too."""
    src_root = files[0].parents[1]
    dst = cfg["stores"] / f"{model}_{dataset}"
    (dst / "test").mkdir(parents=True, exist_ok=True)
    for f in files:
        (dst / "test" / f.name).symlink_to(f.resolve())
    for sidecar in src_root.glob("*.h5"):
        (dst / sidecar.name).symlink_to(sidecar.resolve())
    print(f"{dst}: {len(files)} items")


def derive(model, dataset, cfg):
    source = {"goat_di": "goat", "maps_full": "maps"}[dataset]
    src = cfg["stores"] / f"{model}_{source}"
    if not (src / MARKER).exists():
        raise SystemExit(f"{src} is not complete: extract {source} for {model} first")
    if dataset == "goat_di":
        # The 170 DI performances of the goat store.
        files = [src / "test" / GOAT_ITEM[model].format(item=item)
                 for item in (HERE / "splits/goat_di_items.txt").read_text().split()]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} DI items missing from {src}, e.g. {missing[0].name}")
    else:
        # Every MUS piece, whichever split the pretraining store put it in.
        files = sorted(f for split in ("train", "val", "test")
                       for f in (src / split).iterdir() if f.is_file())
        if len(files) != MAPS_PIECES:
            raise SystemExit(f"{src} holds {len(files)} pieces, expected the {MAPS_PIECES} MAPS MUS pieces")
    link_store(model, dataset, cfg, files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--model", required=True, choices=sorted(GRID) + ["all"])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS) + sorted(DERIVED) + ["all"])
    args = ap.parse_args()
    cfg = config.load(args.config)

    import snap2midi as s2m
    extractor = s2m.extract.SnapExtractor()
    datasets = list(DATASETS) + list(DERIVED) if args.dataset == "all" else [args.dataset]
    for dataset in dict.fromkeys(datasets):
        extracted = DATASETS[dataset][-1] if dataset in DATASETS else ()
        derived = DERIVED.get(dataset, ())
        models = extracted + derived if args.model == "all" else (args.model,)
        for model in models:
            if model not in extracted + derived:
                raise SystemExit(f"{dataset} is not used for {model} in the study")
            store = cfg["stores"] / f"{model}_{dataset}"
            if (store / MARKER).exists():
                print(f"[done] {store}")
                continue
            if store.exists():
                raise SystemExit(f"{store} exists but has no {MARKER} marker: it is from an "
                                 "interrupted run. Delete it and run again.")
            print(f"\n=== {model} x {dataset} -> {store} ===")
            if model in derived:
                derive(model, dataset, cfg)
            else:
                extract(model, dataset, cfg, extractor)
            (store / MARKER).touch()


if __name__ == "__main__":
    main()
