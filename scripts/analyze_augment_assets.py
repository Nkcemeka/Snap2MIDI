"""
Analyze augmentation assets (room IRs and background noise).

Computes the acoustic statistics used to filter the augmentation pools:

  * Room IRs   -> RT60, early/late energy ratio, duration.
                  Long tails smear note onsets; near-anechoic IRs are no-ops.
  * Bg noise   -> crest factor, short-term energy variance, and the fraction of
                  energy inside the transcription band. Impulsive noise adds
                  spurious onset evidence; noise that is almost entirely rumble
                  is inaudible to the model at the SNRs used here.

Writes a JSON record per pool; thresholds are applied later by
build_augment_manifests.py so the analysis is only run once.

Usage:
    python scripts/analyze_augment_assets.py --root ~/Desktop/mtg_ir_datasets
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

EARLY_TIME_MS = 50  # early reflection window, matches the room IR literature
FRAME_MS = 50       # short-term energy frame for the stationarity measure

# Band used to decide whether a noise recording is audible to the transcription
# models at all. The upper edge is C8 = 4186 Hz, the highest piano fundamental.
# The lower edge excludes the sub-100 Hz region, where traffic and ventilation
# rumble concentrate and where the models carry little note evidence.
BAND_LOW_HZ = 100.0
BAND_HIGH_HZ = 4200.0
SPECTRUM_N_FFT = 16384  # ~1 s at 16 kHz; resolves the 100 Hz edge cleanly


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file and downmix to mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def estimate_rt60(ir: np.ndarray, sr: int) -> float:
    """Estimate RT60 via Schroeder backward integration.

    Returns the IR duration if the decay never reaches -60 dB, which happens
    when the IR has been truncated before the tail died out.
    """
    cum_energy = np.cumsum(ir[::-1] ** 2)[::-1]
    db = 10 * np.log10(cum_energy / (cum_energy[0] + 1e-20) + 1e-20)
    below = np.flatnonzero(db < -60)
    return float(below[0] / sr) if below.size else float(len(ir) / sr)


def analyze_ir(path: Path) -> dict:
    """Acoustic descriptors for a single impulse response."""
    ir, sr = load_mono(path)

    # Align to the direct sound so the early window is measured from arrival,
    # not from whatever pre-delay the recording happens to contain.
    onset = int(np.argmax(np.abs(ir)))
    ir = ir[onset:]

    split = int(EARLY_TIME_MS * sr / 1000)
    e_early = float(np.sum(ir[:split] ** 2))
    e_late = float(np.sum(ir[split:] ** 2))

    ratio_db = 10 * np.log10(e_early / e_late) if e_late > 0 else 100.0

    return {
        "rt60_s": estimate_rt60(ir, sr),
        "early_late_ratio_db": float(ratio_db),
        "duration_s": float(len(ir) / sr),
        "sample_rate": int(sr),
    }


def band_energy_fraction(audio: np.ndarray, sr: int) -> float:
    """Fraction of the total energy that falls inside the transcription band.

    Averaged over non-overlapping windows so a stationary recording is measured
    the same way regardless of its length. Recordings shorter than one window
    are measured from a single zero-padded window.
    """
    n_fft = min(SPECTRUM_N_FFT, len(audio))
    n_windows = max(len(audio) // n_fft, 1)
    windows = np.resize(audio, n_windows * n_fft).reshape(n_windows, n_fft)

    spectrum = np.abs(np.fft.rfft(windows * np.hanning(n_fft), axis=1)) ** 2
    psd = spectrum.mean(axis=0)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)

    band = (freq >= BAND_LOW_HZ) & (freq < BAND_HIGH_HZ)
    return float(psd[band].sum() / (psd.sum() + 1e-20))


def analyze_noise(path: Path) -> dict:
    """Descriptors capturing how impulsive a noise recording is."""
    audio, sr = load_mono(path)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    crest_db = 20 * np.log10(peak / rms) if rms > 0 else 100.0

    # Short-term RMS spread: stationary ambience stays flat, impulsive
    # material (horns, door chimes, speech) swings wildly frame to frame.
    frame = int(FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10
    frame_db = 20 * np.log10(frame_rms)

    return {
        "rms": rms,
        "crest_factor_db": float(crest_db),
        "band_energy_fraction": band_energy_fraction(audio, sr),
        "frame_db_std": float(np.std(frame_db)),
        "frame_db_range": float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5)),
        "duration_s": float(len(audio) / sr),
        "sample_rate": int(sr),
    }


def scan(pool_dir: Path, analyze_fn) -> list[dict]:
    """Analyze every wav under pool_dir/{train,test}, tagging each with its split."""
    records = []
    for split in ("train", "test"):
        files = sorted((pool_dir / split).rglob("*.wav"))
        for path in tqdm(files, desc=f"{pool_dir.name}/{split}"):
            try:
                record = analyze_fn(path)
            except Exception as exc:  # keep going; report at the end
                print(f"  skipped {path.name}: {exc}")
                continue
            record["path"] = str(path.relative_to(pool_dir))
            record["split"] = split
            record["source"] = path.relative_to(pool_dir / split).parts[0]
            records.append(record)
    return records


def summarize(records: list[dict], keys: list[str]) -> None:
    """Print percentiles so thresholds can be chosen from the data."""
    for key in keys:
        values = np.array([r[key] for r in records])
        pcts = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
        print(f"  {key:24s} " + "  ".join(f"{p:7.2f}" for p in pcts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="mtg_ir_datasets directory containing room_ir/ and bg_noise/")
    parser.add_argument("--out", type=Path, default=Path("scripts/augment_asset_stats.json"))
    args = parser.parse_args()

    root = args.root.expanduser()

    print("Analyzing room impulse responses...")
    irs = scan(root / "room_ir", analyze_ir)
    print("Analyzing background noise...")
    noise = scan(root / "bg_noise", analyze_noise)

    header = "  " + " " * 24 + "  ".join(f"{p:>7s}" for p in
                                         ["min", "p5", "p25", "p50", "p75", "p95", "max"])
    print(f"\nroom_ir  (n={len(irs)})")
    print(header)
    summarize(irs, ["rt60_s", "early_late_ratio_db", "duration_s"])
    print(f"\nbg_noise (n={len(noise)})")
    print(header)
    summarize(noise, ["crest_factor_db", "frame_db_std", "frame_db_range",
                      "band_energy_fraction"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"room_ir": irs, "bg_noise": noise}, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
