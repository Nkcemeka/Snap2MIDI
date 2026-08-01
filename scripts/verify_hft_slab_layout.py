"""Check the audio slab written by extraction reproduces the old feature slab.

`verify_hft_frame_equivalence.py` proved the *maths*: a per-item mel computed
with center=False over the right sample range equals the matching slice of a
whole-track mel computed with center=True. This script checks the *layout*: that
extraction actually places each track in the audio slab where dataset_idx says
its frames are.

That is the one new risk the audio-slab rewrite introduces, and it is silent.
The frame -> sample conversion is a single `* hop_sample`; get it wrong by one
frame and every label in the dataset moves by 16 ms, which is well inside the
50 ms onset tolerance. Nothing raises, nothing fails, the model just trains on
slightly wrong targets.

So: run the real extraction over a handful of tracks, rebuild the old feature
slab the way the old collation did, and compare it against per-item mels read
out of the new audio slab at the same indices.

One difference is expected and benign. The old slab stored exactly
`num_frames` frames per track and left the rest at log(log_offset); the new one
stores samples, so frames whose analysis window straddles the end of a track see
the real decaying tail instead of hard silence. That is confined to the
fft_bins // hop_sample frames after each track ends -- a region that carries no
labels -- and it is the more faithful of the two. It is reported separately
rather than folded into the pass/fail.

Usage:
    python scripts/verify_hft_slab_layout.py
    python scripts/verify_hft_slab_layout.py --maps-path ~/Documents/maps/MAPS --n-tracks 4
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snap2midi.extract import SnapExtractor
from snap2midi.extractor.modes.hft_mode import _HFTMode

TOLERANCE = 1e-4


def _mel(audio: torch.Tensor, meta: dict, center: bool) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=meta["sr"],
        n_fft=meta["fft_bins"],
        hop_length=meta["hop_sample"],
        win_length=meta["window_length"],
        n_mels=meta["mel_bins"],
        pad_mode=meta["pad_mode"],
        norm="slaney",
        center=center,
    )
    return torch.log(transform(audio) + meta["log_offset"]).T


def item_from_slab(slab: np.ndarray, start_frame: int, meta: dict) -> torch.Tensor:
    """Rebuild one item from the audio slab -- what the dataset will do.

    A flat contiguous slice, no branching: the slab carries audio_pad samples of
    headroom at each end precisely so that frame 0 does not need a negative
    index and the final frame does not run off the end.
    """
    n_item = meta["margin_b"] + meta["num_frame"] + meta["margin_f"]
    length = (n_item - 1) * meta["hop_sample"] + meta["fft_bins"]
    begin = start_frame * meta["hop_sample"]
    buffer = torch.from_numpy(np.asarray(slab[begin: begin + length], dtype=np.float32))
    assert len(buffer) == length, f"slab too short at frame {start_frame}"
    return _mel(buffer, meta, center=False)


def track_layout(idx: np.ndarray) -> list:
    """Recover each track's (first frame, frame count) from dataset_idx.

    dataset_idx is a concatenation of arange blocks, one per track, so the
    breaks in it are the track boundaries. Reading the layout back out of the
    artefact rather than recomputing it means the check cannot accidentally
    share an arithmetic mistake with the code it is checking -- and it does not
    depend on the per-track npz files, which collation deletes for train/val.
    """
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(block[0]), len(block)) for block in np.split(idx, breaks)]


def old_style_slab(audio_files: list, layout: list, meta: dict,
                   total_num_frame: int) -> np.ndarray:
    """The feature slab the previous collation would have written."""
    slab = np.full((total_num_frame, meta["mel_bins"]),
                   np.log(meta["log_offset"]), dtype=np.float32)
    for audio_file, (loc_d, _) in zip(audio_files, layout):
        audio, sr = torchaudio.load(str(audio_file))
        audio = torchaudio.transforms.Resample(sr, meta["sr"])(torch.mean(audio, dim=0))
        feature = _mel(audio, meta, center=True)
        slab[loc_d: loc_d + feature.shape[0]] = feature.numpy()
    return slab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-path", type=Path, default=Path("~/Documents/maps/MAPS"))
    parser.add_argument("--n-tracks", type=int, default=3)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("data/_hft_layout_check"))
    parser.add_argument("--keep", action="store_true", help="do not delete work-dir")
    args = parser.parse_args()

    maps_path = args.maps_path.expanduser()
    if not maps_path.exists():
        print(f"No MAPS at {maps_path}. Pass --maps-path.")
        return 2

    work = args.work_dir
    if work.exists():
        shutil.rmtree(work)

    # Run the real extraction, restricted to a few train tracks. Everything
    # under test -- the audio slab, dataset_idx, the meta file -- is produced by
    # the same code path a full extraction uses.
    chosen: list = []

    class _Subset(_HFTMode):
        def _get_splits_hft(self):
            train, _, _ = super()._get_splits_hft()
            chosen.extend(train[: args.n_tracks])
            return chosen, [], []

    import snap2midi.extract as extract_module
    extract_module._HFTMode = _Subset
    try:
        SnapExtractor().extract_hft(str(maps_path), dataset_name="maps",
                                    save_name=str(work))
    finally:
        extract_module._HFTMode = _HFTMode

    meta = json.loads((work / "meta/train/dataset_meta000.json").read_text())
    slab = np.load(work / "audio/train/dataset_audio000.npy", mmap_mode="r")
    idx = np.load(work / "idx/train/dataset_idx000.npy")
    hop, sr = meta["hop_sample"], meta["sr"]

    print(f"\n{meta['num_tracks']} tracks, {meta['total_num_frame']} frames, "
          f"slab {len(slab)} samples ({len(slab) / sr / 60:.1f} min), "
          f"audio_pad {meta['audio_pad']}\n")

    expected_len = meta["total_num_frame"] * hop + 2 * meta["audio_pad"]
    if len(slab) != expected_len:
        print(f"FAIL: slab is {len(slab)} samples, expected {expected_len}")
        return 1

    # Collation globs the per-track npz files in sorted order, and each is named
    # after its audio file's stem, so this is the order the tracks were laid
    # down in.
    audio_files = [c[0] for c in sorted(chosen, key=lambda c: c[0].stem)]
    layout = track_layout(idx)
    if len(layout) != len(audio_files):
        print(f"FAIL: dataset_idx describes {len(layout)} tracks, "
              f"{len(audio_files)} were extracted")
        return 1
    reference = old_style_slab(audio_files, layout, meta, meta["total_num_frame"])

    n_item = meta["margin_b"] + meta["num_frame"] + meta["margin_f"]
    track_end = [loc_d + n for loc_d, n in layout]

    # The two slabs may differ in exactly one place, and it is worth stating
    # precisely rather than waving at "the tails". The old slab held real mel for
    # a track's own frames and log(log_offset) after; the new one holds samples,
    # so a frame whose 2048-sample window straddles the end of a track sees the
    # real decaying tail instead. A window reaching frame e + fft_bins // hop is
    # clear of the audio entirely and is silence again. So the difference is
    # confined to frames [e, e + fft_bins // hop) for each track end e -- and
    # every other frame, including the last frame of the track itself, must
    # match exactly. That is checked per frame, not per item.
    tail = meta["fft_bins"] // hop

    # Probe across the division so a per-track offset cannot hide, and probe each
    # track's first and last frames directly so the boundary behaviour above is
    # actually exercised rather than assumed.
    probes = set(np.linspace(0, len(idx) - 1, 40).astype(int).tolist())
    for loc_d, n in layout:
        end = loc_d + n
        for frame in (loc_d, loc_d + 1, end - 1, end - 2, end - 40,
                      end - (n_item - meta["margin_b"] - 1)):
            if loc_d <= frame < end:
                probes.add(int(np.searchsorted(idx, frame)))
    probes = sorted(p for p in probes if 0 <= p < len(idx))

    worst_clean, worst_tail, n_boundary, failures = 0.0, 0.0, 0, []

    for p in probes:
        start = int(idx[p]) - meta["margin_b"]
        actual = item_from_slab(slab, start, meta)
        expect = torch.from_numpy(reference[start: start + n_item])
        diff = (actual - expect).abs().numpy()

        frames = np.arange(start, start + n_item)
        excused = np.zeros(n_item, dtype=bool)
        for e in track_end:
            excused |= (frames >= e) & (frames < e + tail)

        if excused.any():
            n_boundary += 1
            worst_tail = max(worst_tail, float(diff[excused].max()))
        if (~excused).any():
            here = float(diff[~excused].max())
            worst_clean = max(worst_clean, here)
            if here > TOLERANCE:
                failures.append((p, start, here))

    print(f"  probes: {len(probes)}  ({n_boundary} include frames just past a "
          f"track end)")
    print(f"  worst |diff|, every frame that must match : {worst_clean:.3e}  "
          f"(tolerance {TOLERANCE:.0e})")
    print(f"  worst |diff|, the {tail} frames after a track end : {worst_tail:.3e}  "
          f"(may differ: new slab keeps the real tail)")

    # A frame offset is the failure this exists to catch, so name it directly
    # rather than leaving a bare tolerance breach to be interpreted.
    for p, start, diff in failures[:5]:
        print(f"    idx[{p}] start frame {start}: max|diff| = {diff:.3e}")
        actual = item_from_slab(slab, start, meta)
        for shift in (-2, -1, 1, 2):
            lo = start + shift
            if lo < 0 or lo + n_item > meta["total_num_frame"]:
                continue
            e = torch.from_numpy(reference[lo: lo + n_item])
            if float((actual - e).abs().max()) <= TOLERANCE:
                print(f"      >>> matches at a shift of {shift} frames "
                      f"({shift * hop / sr * 1000:+.0f} ms of label drift)")

    # A layout check that cannot fail proves nothing, and the error it exists to
    # catch -- a frame-sized slip in the * hop_sample conversion -- is invisible
    # downstream. So read the slab deliberately wrong and require it to show up.
    print("\n  sensitivity (these must all be detected):")
    probe = int(idx[len(idx) // 2]) - meta["margin_b"]
    truth = torch.from_numpy(reference[probe: probe + n_item])
    sensitive = True
    for shift in (-2, -1, 1, 2):
        diff = float((item_from_slab(slab, probe + shift, meta) - truth).abs().max())
        caught = diff > TOLERANCE
        sensitive &= caught
        print(f"    {shift:+d} frame ({shift * hop / sr * 1000:+.0f} ms): "
              f"max|diff| = {diff:.3e}  {'detected' if caught else '*** MISSED ***'}")

    # Sub-frame: a half-hop slip survives frame arithmetic entirely and is what a
    # check counting only frames would wave through.
    length = (n_item - 1) * hop + meta["fft_bins"]
    begin = probe * hop + hop // 2
    slipped = _mel(torch.from_numpy(
        np.array(slab[begin: begin + length], dtype=np.float32)), meta, center=False)
    diff = float((slipped - truth).abs().max())
    caught = diff > TOLERANCE
    sensitive &= caught
    print(f"    half-hop ({hop // 2} samples, {hop / 2 / sr * 1000:.0f} ms): "
          f"max|diff| = {diff:.3e}  {'detected' if caught else '*** MISSED ***'}")

    # dataset_idx must still address only real frames, and the labels must line
    # up with the same layout the audio was written into.
    labels = np.load(work / "label_frames/train/dataset_label_frames000.npy",
                     mmap_mode="r")
    ok_shape = labels.shape[0] == meta["total_num_frame"]
    ok_idx = int(idx.max()) + n_item - meta["margin_b"] <= meta["total_num_frame"]
    print(f"\n  labels have {labels.shape[0]} rows, slab has "
          f"{meta['total_num_frame']} frames: {'ok' if ok_shape else 'MISMATCH'}")
    print(f"  furthest item ends at frame "
          f"{int(idx.max()) + n_item - meta['margin_b']} of "
          f"{meta['total_num_frame']}: {'in bounds' if ok_idx else 'OUT OF BOUNDS'}")

    if not args.keep:
        shutil.rmtree(work)

    if failures or not ok_shape or not ok_idx:
        print("\nFAIL: the audio slab does not reproduce the old feature slab.")
        return 1
    if not sensitive:
        print("\nINCONCLUSIVE: the slab matched, but the check failed to notice a "
              "deliberate misalignment, so the match proves nothing.")
        return 1
    print("\nPASS: per-item mels from the audio slab reproduce the old feature "
          "slab at every dataset_idx probed, and the check detects both frame "
          "and sub-frame misalignment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
