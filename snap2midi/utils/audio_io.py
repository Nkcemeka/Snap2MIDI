"""Waveform loading that does not depend on torchcodec.

torchaudio >= 2.9 routes `torchaudio.load` through torchcodec, which is a
separate package with its own FFmpeg requirement. The cluster's snap2midi
environment has the new torchaudio and no torchcodec, so every extraction
there died with "TorchCodec is required for load_with_torchcodec" -- on the
laptop the same code passed because torchcodec happened to be installed.

`load_audio` tries torchaudio first, exactly as before, and falls back to
soundfile, which reads PCM WAV/FLAC without FFmpeg. For 16-bit and 24-bit
PCM the two decoders return the same float32 samples (verified bit-exact on
GAPS WAVs), so the store does not depend on which path ran.
"""

from __future__ import annotations

import numpy as np
import torch


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Return (waveform[channels, samples] float32, sample_rate)."""
    try:
        import torchaudio
        return torchaudio.load(path)
    except ImportError:
        # torchaudio present but its decoder backend (torchcodec) is not.
        pass
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)
