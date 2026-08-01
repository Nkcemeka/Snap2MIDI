"""Prove hFT items can be built from audio instead of from a stored spectrogram.

hFT currently stores one log-mel per track and slices it by frame index at
training time, so by then the waveform is gone and there is nothing left to
augment. Moving augmentation to hFT means storing audio instead and computing
the mel per item -- but only if a per-item mel is *identical* to the
corresponding slice of the whole-track mel it replaces.

That equality is the whole risk of the change, and it is a silent one. One
frame of drift is 16 ms at hop 256 / 16 kHz, comfortably inside the 50 ms
onset tolerance, so a mistake here never raises and never fails an evaluation.
It just quietly moves every label in the dataset. This script exists to turn
that into a loud, checkable fact before any extraction or dataset code is
touched.

What has to line up
-------------------
The stored feature is computed with center=True, which pads the signal by
n_fft // 2 and centres frame t on sample t * hop -- so frame t spans

    [t * hop - n_fft // 2,  t * hop + n_fft // 2)

An item is `margin_b + num_frame + margin_f` = 192 consecutive frames starting
at some frame s. Reproducing those frames from audio alone means taking

    [s * hop - n_fft // 2,  (s + 191) * hop + n_fft // 2)

which is (192 - 1) * hop + n_fft samples, and running the same transform with
center=False so it does not pad again. Frame 0 of that buffer then covers
exactly the same samples as frame s of the whole-track feature.

The padding between tracks needs no special handling. hft_mode.py fills the
gaps with log(log_offset), and since the feature is log(mel + log_offset),
that is exactly what silence produces -- so an audio slab zero-padded in the
same layout reproduces the current padding for free. The first and last items
of a track exercise that path and are tested explicitly below.

Usage:
    python scripts/verify_hft_frame_equivalence.py
    python scripts/verify_hft_frame_equivalence.py --audio-root ~/Documents/maps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Defaults of Trainer.extract_hft / HFTDataset. Kept literal rather than
# imported so a config change cannot make this check silently vacuous.
SR = 16000
FFT_BINS = 2048
WINDOW_LENGTH = 2048
HOP_SAMPLE = 256
MEL_BINS = 256
LOG_OFFSET = 1e-8
PAD_MODE = "constant"

MARGIN_B = 32
NUM_FRAME = 128
MARGIN_F = 32
ITEM_FRAMES = MARGIN_B + NUM_FRAME + MARGIN_F  # 192

# log-mel values run to about -18.4 (silence). Whole-track and per-item FFTs
# are the same arithmetic in a different batching, so only float noise should
# separate them; anything near a real frame offset is orders of magnitude
# larger than this.
TOLERANCE = 1e-4


def _mel_transform(center: bool) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=FFT_BINS,
        hop_length=HOP_SAMPLE,
        win_length=WINDOW_LENGTH,
        n_mels=MEL_BINS,
        pad_mode=PAD_MODE,
        norm="slaney",
        center=center,
    )


def load_audio(path: Path) -> torch.Tensor:
    """Mono 16 kHz waveform, exactly as hft_mode._get_audio_hft loads it."""
    audio, sr = torchaudio.load(str(path))
    audio = torch.mean(audio, dim=0)
    return torchaudio.transforms.Resample(sr, SR)(audio)


def whole_track_feature(audio: torch.Tensor) -> torch.Tensor:
    """The current path: one mel over the whole track, shape (frames, mel)."""
    feature = _mel_transform(center=True)(audio)
    return torch.log(feature + LOG_OFFSET).T


def item_from_audio(audio: torch.Tensor, start_frame: int) -> torch.Tensor:
    """The proposed path: ITEM_FRAMES frames rebuilt from the waveform alone.

    Reads past either end of the track as silence, which is what center=True
    does at the track edges and what the inter-track gaps hold anyway.
    """
    pad = FFT_BINS // 2
    begin = start_frame * HOP_SAMPLE - pad
    length = (ITEM_FRAMES - 1) * HOP_SAMPLE + FFT_BINS

    buffer = torch.zeros(length, dtype=audio.dtype)
    src_s, src_e = max(begin, 0), min(begin + length, audio.shape[0])
    if src_e > src_s:
        buffer[src_s - begin: src_e - begin] = audio[src_s:src_e]

    feature = _mel_transform(center=False)(buffer)
    return torch.log(feature + LOG_OFFSET).T


def check_track(path: Path) -> bool:
    audio = load_audio(path)
    full = whole_track_feature(audio)
    n_frames = full.shape[0]

    # First and last valid items exercise the zero-padded edges; the rest are
    # spread through the track so a position-dependent error cannot hide.
    last = n_frames - ITEM_FRAMES
    if last < 0:
        print(f"  SKIP {path.name}: shorter than one item")
        return True
    starts = sorted({0, 1, last // 4, last // 2, (3 * last) // 4, last - 1, last})
    starts = [s for s in starts if 0 <= s <= last]

    print(f"  {path.name}  ({n_frames} frames, {n_frames * HOP_SAMPLE / SR / 60:.1f} min)")
    ok = True
    worst = 0.0
    for s in starts:
        expected = full[s: s + ITEM_FRAMES]
        actual = item_from_audio(audio, s)
        if actual.shape != expected.shape:
            print(f"    frame {s:>7}: SHAPE {tuple(actual.shape)} != {tuple(expected.shape)}")
            ok = False
            continue
        diff = float((actual - expected).abs().max())
        worst = max(worst, diff)
        flag = "ok " if diff <= TOLERANCE else "FAIL"
        print(f"    frame {s:>7}: max|diff| = {diff:.3e}  {flag}")
        ok &= diff <= TOLERANCE

    # A frame offset is the failure this exists to catch, so name it directly
    # rather than leaving a bare tolerance breach to be interpreted.
    if not ok:
        expected = full[starts[len(starts) // 2]: starts[len(starts) // 2] + ITEM_FRAMES]
        actual = item_from_audio(audio, starts[len(starts) // 2])
        for shift in (-2, -1, 1, 2):
            a = actual[max(0, shift): ITEM_FRAMES + min(0, shift)]
            e = expected[max(0, -shift): ITEM_FRAMES + min(0, -shift)]
            if float((a - e).abs().max()) <= TOLERANCE:
                print(f"    >>> matches at a shift of {shift} frames "
                      f"({shift * HOP_SAMPLE / SR * 1000:+.0f} ms of label drift)")
    print(f"    worst: {worst:.3e} (tolerance {TOLERANCE:.0e})")
    return ok


def check_sensitivity(path: Path) -> bool:
    """Confirm the comparison would actually notice a misalignment.

    An equivalence check that cannot fail proves nothing, and the failure it
    has to catch -- a whole-frame or sub-frame offset -- is invisible
    downstream. So deliberately introduce both and require them to be flagged.
    """
    audio = load_audio(path)
    full = whole_track_feature(audio)
    start = min(5000, full.shape[0] - ITEM_FRAMES - 4)
    item = item_from_audio(audio, start)

    ok = True
    for shift in (-2, -1, 1, 2):
        expected = full[start + shift: start + shift + ITEM_FRAMES]
        diff = float((item - expected).abs().max())
        caught = diff > TOLERANCE
        ok &= caught
        print(f"    {shift:+d} frame ({shift * HOP_SAMPLE / SR * 1000:+.0f} ms): "
              f"max|diff| = {diff:.3e}  {'detected' if caught else '*** MISSED ***'}")

    # Sub-frame: a half-hop slip survives frame arithmetic and is the error a
    # frame-counting check would let through.
    pad = FFT_BINS // 2
    length = (ITEM_FRAMES - 1) * HOP_SAMPLE + FFT_BINS
    begin = start * HOP_SAMPLE - pad + HOP_SAMPLE // 2
    buffer = audio[begin: begin + length].clone()
    slipped = torch.log(_mel_transform(center=False)(buffer) + LOG_OFFSET).T
    diff = float((slipped - full[start: start + ITEM_FRAMES]).abs().max())
    caught = diff > TOLERANCE
    ok &= caught
    print(f"    half-hop ({HOP_SAMPLE // 2} samples, "
          f"{HOP_SAMPLE / 2 / SR * 1000:.0f} ms): max|diff| = {diff:.3e}  "
          f"{'detected' if caught else '*** MISSED ***'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, default=Path("~/Documents/maps"))
    parser.add_argument("--n-tracks", type=int, default=3)
    args = parser.parse_args()

    root = args.audio_root.expanduser()
    tracks = sorted(root.rglob("MAPS_MUS-*.wav"))[: args.n_tracks]
    if not tracks:
        print(f"No MAPS audio under {root}. Pass --audio-root.")
        return 2

    print(f"item = {MARGIN_B} + {NUM_FRAME} + {MARGIN_F} = {ITEM_FRAMES} frames "
          f"= {(ITEM_FRAMES - 1) * HOP_SAMPLE + FFT_BINS} samples "
          f"({((ITEM_FRAMES - 1) * HOP_SAMPLE + FFT_BINS) / SR:.2f} s of audio)\n")

    ok = True
    for track in tracks:
        ok &= check_track(track)
        print()

    print("  sensitivity (these must all be detected):")
    sensitive = check_sensitivity(tracks[0])
    print()

    if ok and sensitive:
        print("PASS: per-item mel reproduces the stored feature, and the check "
              "detects both frame and sub-frame misalignment.")
        return 0
    if ok and not sensitive:
        print("INCONCLUSIVE: features matched, but the check failed to notice a "
              "deliberate misalignment, so the match proves nothing.")
        return 1
    print("FAIL: per-item mel does not reproduce the stored feature.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
