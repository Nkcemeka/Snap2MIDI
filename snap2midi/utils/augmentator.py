"""Data augmentation for source-domain (piano) pretraining.

This is the pipeline of Edwards et al., "A Data-Driven Analysis of Robust
Automatic Piano Transcription" (arXiv:2402.01424), Fig. 1, applied unchanged:
five stages, each applied independently with probability 0.5, in a fixed order.

    1. random seven-band parametric EQ    gain per band in [-10, +5] dB
    2. additive background noise          SNR in [17.5, 25] dB
    3. pitch shift                        +/- 0.1 semitone (+/- 10 cents)
    4. random seven-band parametric EQ    independent draw, same range
    5. reverb                             convolution with one room IR

The pipeline is reproduced rather than redesigned, deliberately. Augmentation
is a controlled nuisance variable in this project, not a contribution: it is
held identical across every architecture under comparison, so it cannot explain
any difference between them. Reproducing a published, already-ablated pipeline
means the justification for the stage list and the parameter ranges is Edwards'
Tables III and IV rather than an experiment we would otherwise owe. Every
deviation below is therefore either forced by our assets or is a property of
how the assets are sampled, never a change to what the pipeline does.

Two things Edwards' ablation is worth remembering for:

  * Pitch shift and reverb carry the pipeline (+3.1 and +2.8 F1 alone on MAPS;
    -3.5 and -3.6 when removed). Background noise and EQ are close to neutral.
    That bounds how much our substituted asset pools can matter: a useless
    noise pool costs about one F1 point, a useless IR pool about three.
  * The +/- 10 cent shift is tuning jitter, not transposition. It is
    label-preserving, so no MIDI has to be edited and no note can be pushed
    outside the 88-key range. That is the property that keeps this pipeline
    free of the silent label-corruption failure mode.

Deviations from the paper, and why:

  * Reverb pool: Edwards' own source. He draws from 14 impulse responses from
    echothief.com, a library of deliberately dramatic spaces -- caves, tunnels,
    stairwells, underpasses, fortresses. We use the same library. The paper does
    not say which 14, so all 115 are used rather than inventing a selection:
    same source, same character, a superset of whatever he drew. No RT60 band or
    space-type filter is applied, because he applied none. See
    scripts/build_edwards_reverb_manifests.py.

    An earlier version substituted a filtered pool from MIT, AachenIR and
    OPENAIR, excluding outdoor spaces and capping RT60 at 2.0 s on the grounds
    that they are not plausible piano recording environments. That produced a
    training pool of domestic rooms -- bedrooms and kitchens, median RT60
    0.57 s -- against Edwards' median 1.09 s reaching 3.47 s: close to the
    opposite of his distribution rather than a variation on it. That pool is now
    the held-out robustness set, which is a cleaner split anyway -- training on
    the paper's distribution, evaluating on realistic rooms never seen.
  * Noise pool. Edwards uses restaurant08.wav from the Audio Degradation
    Toolbox plus four freesound recordings, cut into 65 ten-second clips: pub
    and cafe babble, a narrow distribution of human speech. That is not
    redistributed, so we substitute TUT-acoustic-scenes-2016. It is a different
    *kind* of sound, not a wider sample of the same kind -- 81% of the selected
    clips carry more than half their energy below 100 Hz, engine and traffic
    rumble rather than babble, because the stationarity filter selects for it.
    Edwards' ablation puts background noise at +0.3 F1 alone and -1.0 when
    removed, which bounds what this can cost.
  * Reverb is sampled uniformly over physical spaces rather than over files.
    EchoThief holds one measurement per space so the two coincide there, but the
    held-out pool has 226 impulse responses over 166 spaces, because the
    AachenIR rooms are each measured from many microphone positions. Sampling
    over files would spend about a quarter of those draws on six rooms.
  * Impulse responses are aligned to their direct sound before convolution.
    Ours carry leading silence that would otherwise delay the audio while the
    labels stayed put; see SpaceUniformImpulseResponse. This is a correctness
    requirement, not a stylistic choice -- without it the reverb stage injects
    onset misalignment of up to 90 ms against a 50 ms tolerance.
  * audiomentations is pinned to 0.34.1. It is contemporaneous with the paper,
    and later versions require numpy >= 2, which conflicts with nnaudio2.

Applied during MAESTRO pretraining only. Guitar fine-tuning runs on clean
audio: the impulse response pool is filtered on the premise that MAESTRO is a
concert-hall recording, which does not transfer to guitar, and leaving
adaptation unaugmented keeps the target-domain experiments a test of the
adaptation strategy alone.

Usage:
    aug = Augmentator(sample_rate=16000, asset_root="~/Desktop/mtg_ir_datasets")
    audio = aug(audio, track_id=track, excerpt_start=start, epoch=epoch)
"""

import functools
import hashlib
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import convolve
from audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
    Compose,
    PitchShift,
    SevenBandParametricEQ,
)
from audiomentations.core.audio_loading_utils import load_sound_file

# Edwards et al., Fig. 1. These are the paper and should not be tuned.
EQ_MIN_GAIN_DB = -10.0
EQ_MAX_GAIN_DB = 5.0
NOISE_MIN_SNR_DB = 17.5
NOISE_MAX_SNR_DB = 25.0
PITCH_SHIFT_SEMITONES = 0.1
STAGE_PROBABILITY = 0.5

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "augment" / "manifests"

# Impulse responses are short, so the whole pool is cached. Noise clips are 30 s
# each -- 1.9 MB at 16 kHz, 5.3 MB at 44.1 kHz -- and the cache is per dataloader
# worker, so caching all 520 would cost ~1 GB per worker at 16 kHz. 64 keeps the
# resident set near 120 MB at 16 kHz while still absorbing most reloads.
DEFAULT_NOISE_CACHE_SIZE = 64


def _read_lines(path: Path) -> list[str]:
    """Read a manifest of bare pool-relative paths, one per line."""
    return [line for line in path.read_text().splitlines() if line.strip()]


def _read_grouped_lines(path: Path) -> dict[str, list[str]]:
    """Read a manifest of `space<TAB>path` pairs into space -> paths.

    Paths stay pool-relative; the caller resolves them against the pool
    directory.
    """
    spaces: dict[str, list[str]] = {}
    for line in _read_lines(path):
        space, _, relative = line.partition("\t")
        if not relative:
            raise ValueError(
                f"{path} line {line!r} is missing the space column. Regenerate the "
                "manifests with scripts/build_augment_manifests.py."
            )
        spaces.setdefault(space, []).append(relative)
    return spaces


class SpaceUniformImpulseResponse(ApplyImpulseResponse):
    """ApplyImpulseResponse that draws a physical space, then a measurement,
    and that aligns each impulse response to its direct sound before convolving.

    Two changes to the parent, both about our assets rather than about what the
    reverb stage does.

    Sampling. The parent picks uniformly over files, which over-weights rooms
    measured from many microphone positions. Two measurements of the same
    stairway are not two acoustics, so weighting them as two would quietly
    narrow the reverb distribution.

    Pre-delay. Our impulse responses contain leading silence before the direct
    sound -- a median of 5.9 ms across the training pool, over 10 ms for 21% of
    them and over 50 ms for 2.8%. Convolving with such a file delays the whole
    excerpt by that amount while the MIDI labels stay where they were, so the
    reverb stage would silently misalign onsets against a 50 ms tolerance, and
    for the worst files would push every onset in the excerpt out of tolerance
    on its own. Trimming to the direct sound removes the delay and leaves the
    reverberation itself untouched. This mirrors what `analyze_ir` in
    scripts/analyze_augment_assets.py already does before measuring RT60.

    Output level, via `level`:

        "peak"  the parent's own behaviour, and therefore what Edwards ran:
                peak-normalize the convolution to 0.5 regardless of the input.
                This discards the excerpt's level. Measured on real hFT items it
                raises the mean log-mel by 3.5 nats (+15.2 dB of power) and makes
                "did reverb fire?" predictable from loudness alone at AUC 0.93.
                About 61% of what the stage does to the spectrogram is that level
                jump rather than room character.
        "rms"   restore the excerpt's own energy, which is Kaldi's
                wav-reverberate --normalize-output and its default: "scale so
                that the signal energy is the same as the original input
                signal". torchaudio's augmentation tutorial reaches the same
                place from the other side, normalizing the impulse response to
                unit L2 norm and not touching the output. On the same items this
                takes the AUC to 0.42 -- chance -- with no clipping.

    Default is "peak", so the reproduction is faithful unless asked otherwise.
    Both mainstream reverb-augmentation references preserve level and
    audiomentations is the outlier, so "rms" is the better-supported choice on
    the merits; it is not the default only because the published numbers this
    project checks itself against were produced with "peak".
    """

    def __init__(self, spaces: dict[str, list[str]], level: str = "peak", **kwargs):
        if level not in ("peak", "rms"):
            raise ValueError(f"level must be 'peak' or 'rms', got {level!r}")
        self.level = level
        self._spaces = [sorted(files) for _, files in sorted(spaces.items())]
        flat = [f for files in self._spaces for f in files]
        super().__init__(ir_path=flat, **kwargs)
        cache_size = kwargs.get("lru_cache_size", len(flat))
        # Shadows the staticmethod of the same name; see __getstate__. mypy
        # objects to the shadowing, which is the whole point of the pattern.
        self._load_aligned_ir = functools.lru_cache(maxsize=cache_size)(  # type: ignore[method-assign]
            SpaceUniformImpulseResponse._load_aligned_ir
        )

    @staticmethod
    def _load_aligned_ir(file_path: str, sample_rate: int) -> np.ndarray:
        """Load an impulse response and drop everything before its direct sound."""
        ir, _ = load_sound_file(file_path, sample_rate)
        return ir[int(np.argmax(np.abs(ir))):]

    def __getstate__(self) -> dict:
        """Drop the decode cache so the transform can be pickled.

        Both audiomentations parents do exactly this, and the reason the
        pattern works is easy to miss: the deleted name also exists as a
        staticmethod on the class, so attribute lookup falls through to an
        uncached loader and the unpickled transform still works, just slower
        until the cache refills. Doing the same here rather than inventing a
        __setstate__ keeps this class readable next to the ones it inherits
        from.

        Without this an lru_cache wrapper reaches pickle and raises, which
        makes the whole Augmentator unpicklable -- fine under fork, fatal
        under any spawn-based dataloader or ddp_spawn.
        """
        state = super().__getstate__()
        del state["_load_aligned_ir"]
        return state

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int) -> None:
        # The parent draws should_apply and then a uniform-over-files impulse
        # response; the latter is discarded and replaced by the two-stage draw.
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(random.choice(self._spaces))

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convolve with the aligned impulse response, then set the level.

        `level="peak"` mirrors the parent exactly -- peak normalization to 0.5,
        then truncation -- differing only in that the impulse response has been
        aligned to its direct sound. `level="rms"` restores the excerpt's own
        energy instead; see the class docstring for why that is an option.

        The realized gain is recorded in `parameters` so `last_applied()`
        surfaces it: under "peak" this stage moves the level by a factor of
        roughly 8 and the effect is otherwise invisible. Mono only, which is all
        this project feeds it.
        """
        ir = self._load_aligned_ir(self.parameters["ir_file_path"], sample_rate)
        reverberated = convolve(samples, ir)

        if self.level == "peak":
            peak: float = max(np.amax(reverberated), -np.amin(reverberated))
            if peak > 0.0:
                reverberated *= 0.5 / peak
            # Truncation is not optional here: the labels are not shifted by the
            # convolution, so the excerpt has to keep its original length.
            out = reverberated[: samples.shape[-1]]
        else:
            # Kaldi's wav-reverberate --normalize-output, which is its default:
            # "scale so that the signal energy is the same as the original
            # input signal". Measured on the delivered excerpt rather than the
            # full convolution, since the tail past the excerpt is discarded.
            out = reverberated[: samples.shape[-1]]
            rms_in = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
            rms_out = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))
            if rms_in > 0.0 and rms_out > 0.0:
                out = out * (rms_in / rms_out)

        rms_in = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        rms_final = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))
        self.parameters["level"] = self.level
        self.parameters["rms_gain"] = rms_final / rms_in if rms_in > 0.0 else 0.0
        return out.astype(np.float32)


class Augmentator:
    """Edwards et al.'s piano pretraining augmentation, applied on the fly.

    Args
    -----
        sample_rate (int): rate of the audio passed to `__call__`. Assets are
            16 kHz and are resampled to match on first use; every model front
            end in this project caps at or below 8 kHz, so the pools cover the
            full modelled band even for the 44.1 kHz model.
        asset_root (str | Path): directory holding `room_ir/` and `bg_noise/`.
        split (str): which manifest pair to draw from. Training uses `train`;
            `test` exists so robustness evaluation measures generalization to
            unseen rooms rather than recall of the training pool.
        manifest_dir (str | Path | None): overrides the packaged manifests.
        seed (int): salts the per-excerpt seed derivation, so a run can be
            repeated exactly or varied wholesale.
        noise_cache_size (int): LRU entries for decoded noise clips, per worker.
        ir_cache_size (int | None): LRU entries for decoded impulse responses.
            Defaults to the whole pool.
        reverb_level (str): how the reverb stage sets its output level.
            "peak" (default) reproduces audiomentations, and therefore Edwards.
            "rms" preserves the excerpt's energy, as Kaldi's wav-reverberate
            does by default. See SpaceUniformImpulseResponse.
    """

    def __init__(
        self,
        sample_rate: int,
        asset_root: str | Path,
        split: str = "train",
        manifest_dir: str | Path | None = None,
        seed: int = 0,
        noise_cache_size: int = DEFAULT_NOISE_CACHE_SIZE,
        ir_cache_size: int | None = None,
        reverb_level: str = "peak",
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.sample_rate = sample_rate
        self.split = split
        self.seed = seed
        self.reverb_level = reverb_level

        root = Path(asset_root).expanduser()
        manifests = Path(manifest_dir) if manifest_dir else DEFAULT_MANIFEST_DIR

        ir_relative = _read_grouped_lines(manifests / f"room_ir_{split}.txt")
        noise_relative = _read_lines(manifests / f"bg_noise_{split}.txt")

        # Absolute paths: audiomentations keys its decode cache on the string it
        # is given, and resolving here keeps those keys stable across workers.
        ir_spaces = {
            space: [str((root / "room_ir" / rel).resolve()) for rel in files]
            for space, files in ir_relative.items()
        }
        noise_files = [str((root / "bg_noise" / rel).resolve()) for rel in noise_relative]
        self._check_assets_exist(ir_spaces, noise_files, root)

        self.n_spaces = len(ir_spaces)
        self.n_impulse_responses = sum(len(f) for f in ir_spaces.values())
        self.n_noise_files = len(noise_files)

        # The decode cache makes each asset a one-off resample, so the warning
        # audiomentations raises on every resample is noise about work we have
        # already amortized.
        warnings.filterwarnings("ignore", message=".*had to be resampled.*")

        self.transform = Compose(
            [
                SevenBandParametricEQ(
                    min_gain_db=EQ_MIN_GAIN_DB,
                    max_gain_db=EQ_MAX_GAIN_DB,
                    p=STAGE_PROBABILITY,
                ),
                AddBackgroundNoise(
                    sounds_path=noise_files,
                    min_snr_db=NOISE_MIN_SNR_DB,
                    max_snr_db=NOISE_MAX_SNR_DB,
                    p=STAGE_PROBABILITY,
                    lru_cache_size=noise_cache_size,
                ),
                PitchShift(
                    min_semitones=-PITCH_SHIFT_SEMITONES,
                    max_semitones=PITCH_SHIFT_SEMITONES,
                    p=STAGE_PROBABILITY,
                ),
                SevenBandParametricEQ(
                    min_gain_db=EQ_MIN_GAIN_DB,
                    max_gain_db=EQ_MAX_GAIN_DB,
                    p=STAGE_PROBABILITY,
                ),
                SpaceUniformImpulseResponse(
                    spaces=ir_spaces,
                    level=reverb_level,
                    p=STAGE_PROBABILITY,
                    lru_cache_size=ir_cache_size or self.n_impulse_responses,
                ),
            ]
        )

    @staticmethod
    def _check_assets_exist(
        ir_spaces: dict[str, list[str]], noise_files: list[str], root: Path
    ) -> None:
        """Fail at construction rather than partway through the first epoch."""
        missing = [
            path
            for path in [f for files in ir_spaces.values() for f in files] + noise_files
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} augmentation assets listed in the manifests are not "
                f"under {root} (first: {missing[0]}). Check --asset_root."
            )

    def _derive_seed(self, track_id: str, excerpt_start: float, epoch: int) -> int:
        """Seed an excerpt's augmentation from its identity, not from call order.

        Two properties matter. Across architectures, the same excerpt in the
        same epoch draws the same augmentation, so a comparison between models
        is not partly a comparison of the augmentation noise they happened to
        receive -- their differing batch sizes and step counts would otherwise
        desynchronise a step-seeded stream. Across epochs, the seed changes, so
        augmentation stays resampled every epoch, which is where the
        regularization comes from.

        blake2b rather than hash(): the built-in is salted per process, so it
        would not reproduce across workers or runs.
        """
        key = f"{self.seed}|{track_id}|{excerpt_start:.6f}|{epoch}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    def __call__(
        self,
        audio: np.ndarray,
        track_id: str | None = None,
        excerpt_start: float = 0.0,
        epoch: int = 0,
    ) -> np.ndarray:
        """Augment one mono excerpt.

        Args
        -----
            audio (np.ndarray): mono waveform at `self.sample_rate`.
            track_id (str | None): stable identifier of the source recording.
                When given, augmentation is seeded from it and is reproducible;
                when omitted, the ambient RNG is used.
            excerpt_start (float): excerpt offset in seconds, part of the seed.
            epoch (int): training epoch, part of the seed.

        Returns
        --------
            augmented (np.ndarray): float32 waveform, same length as the input.
        """
        if audio.ndim != 1:
            raise ValueError(f"expected a mono waveform, got shape {audio.shape}")
        audio = np.asarray(audio, dtype=np.float32)

        if track_id is None:
            return self.transform(audio, sample_rate=self.sample_rate)

        # audiomentations draws from the global `random` module, so seeding has
        # to be global too. The surrounding state is restored afterwards to keep
        # this from perturbing shuffling or any other stochastic step sharing
        # the worker.
        derived = self._derive_seed(track_id, excerpt_start, epoch)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(derived)
            np.random.seed(derived % (2**32))
            return self.transform(audio, sample_rate=self.sample_rate)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def last_applied(self) -> dict:
        """Which stages fired on the most recent call, and with which values.

        Intended for verifying the pipeline -- confirming the realised stage
        rate is near 0.5, or that two architectures received identical draws.
        """
        applied = {}
        for stage, transform in enumerate(self.transform.transforms, start=1):
            parameters = dict(transform.parameters)
            if parameters.pop("should_apply", False):
                # Stage-numbered: the two equalizers are the same class and
                # would otherwise collide on the class name alone.
                applied[f"{stage}_{type(transform).__name__}"] = parameters
        return applied

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sample_rate={self.sample_rate}, split={self.split!r}, "
            f"spaces={self.n_spaces}, impulse_responses={self.n_impulse_responses}, "
            f"noise_files={self.n_noise_files}, "
            f"reverb_level={self.reverb_level!r})"
        )
