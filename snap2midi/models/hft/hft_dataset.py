# Import the necessary libraries
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from .inference import mel_from_audio

# The pre-rewrite store, kept readable so an unaugmented run can be reproduced
# against the exact files it originally used. (directory, file stem, npz key).
LEGACY_ARRAYS = {
    "feature": ("feature", "dataset_feature"),
    "onset": ("label_onset", "dataset_label_onset"),
    "offset": ("label_offset", "dataset_label_offset"),
    "frames": ("label_frames", "dataset_label_frames"),
    "velocity": ("label_velocity", "dataset_label_velocity"),
}


class HFTDataset(Dataset):
    """
        A PyTorch Dataset class for the hFT-Transformer dataset.

        Two storage backends, selected by `config["feature_source"]`:

          "audio" (default)
              Reads the waveform slab written by the current extractor and
              computes the log-mel per item. This is what makes on-the-fly
              augmentation possible at all: the previous design had already
              thrown the waveform away by training time.

          "legacy_feature"
              Reads the pre-rewrite `.npz` spectrogram store and slices it by
              frame index, exactly as this class did before the rewrite. There
              is no waveform, so augmentation is rejected rather than silently
              ignored. Kept so an unaugmented baseline can be reproduced against
              the original files; it is not the maintained path.

        The two backends produce the same *spectrogram* -- verified to 1.9e-6,
        float32 noise, by scripts/verify_hft_dataset_equivalence.py. Their
        labels can differ, because each reads the labels stored alongside its
        own features: commit 7724d29 replaced _midi2note's pretty_midi
        implementation with Sony's mido one, so a store written before it holds
        labels roughly 1.8% different from what the current extractor produces.
        That is a property of the two stores, not of this class.

        This is a map-style Dataset -- __len__ plus __getitem__(i) -- rather than
        the IterableDataset it replaces, and that applies to both backends.
        The old __iter__ had two live bugs:

          * It split work across dataloader workers *by division*. n_div_train
            is 1, so worker 0 did everything and workers 1-3 exited immediately.
          * It shuffled with a generator seeded in the parent process while
            randperm ran in the worker's copy. Workers re-fork from identical
            state every epoch, so the order was byte-identical for the whole
            run. Combined with n_slice's fixed stride, training saw the same
            1/16 of excerpts, in the same order, every epoch.

        Handing indexing to PyTorch's sampler removes both, and they cannot come
        back, because the code that caused them is gone.
    """

    def __init__(self, config: dict, split: str = Optional[None],
                 augmentator=None):
        """
            Initializes the HFTDataset class.

            Args
            -----
                config (dict): Config dictionary. `feature_source` selects the
                    backend and defaults to "audio".
                split (str): Split: train, val or test
                augmentator (Augmentator | None): Applied to the waveform before
                    the mel when set. Leave None to get the unaugmented item,
                    which is bit-identical to what the old dataset returned.
                    Not available under "legacy_feature".
        """
        super().__init__()

        if split is None:
            raise ValueError("Split must be specified. Use 'train', 'val', or 'test'.")

        self.config = config
        self.split = split
        self.augmentator = augmentator
        self.base_path = Path(config["base_path"])
        self.ndivs = config[f"n_div_{split}"]
        self.source = config.get("feature_source", "audio")

        if self.source not in ("audio", "legacy_feature"):
            raise ValueError(
                f"feature_source must be 'audio' or 'legacy_feature', "
                f"got {self.source!r}")
        if self.source == "legacy_feature" and augmentator is not None:
            raise ValueError(
                "feature_source='legacy_feature' stores spectrograms, not audio, "
                "so there is nothing to augment. Re-extract and use "
                "feature_source='audio'.")

        # Set by the epoch callback so augmentation is redrawn each epoch.
        # Without it the augmentator would be told epoch=0 forever and apply
        # identical damage to identical excerpts for the whole run.
        self.epoch = 0

        self.margin_b = config["margin_b"]
        self.num_frame = config["num_frame"]
        self.n_item_frames = self.margin_b + self.num_frame + config["margin_f"]

        if self.source == "audio":
            # Feature parameters come from the meta file written at extraction,
            # not from the training config, so there is exactly one source of
            # truth for how the audio was laid out. The input parameters *are*
            # in the training config, and are checked against the meta rather
            # than assumed -- an off-by-one in the frame -> sample conversion is
            # 16 ms of label drift, comfortably inside the 50 ms onset
            # tolerance, so it would never raise on its own.
            self.meta = self._load_meta(0)
            for div in range(1, self.ndivs):
                if self._load_meta(div) != self.meta:
                    raise ValueError(
                        f"division {div} of {split} was extracted with different "
                        f"parameters than division 0")

            for key in ("margin_b", "margin_f", "num_frame"):
                if config[key] != self.meta[key]:
                    raise ValueError(
                        f"config {key}={config[key]} but {split} was extracted "
                        f"with {key}={self.meta[key]}")
            if config["n_bins"] != self.meta["mel_bins"]:
                raise ValueError(
                    f"config n_bins={config['n_bins']} but {split} was extracted "
                    f"with mel_bins={self.meta['mel_bins']}")

            self.hop = self.meta["hop_sample"]

            # Samples an item needs. The slab carries audio_pad = fft_bins // 2
            # of headroom at each end precisely so this is one flat contiguous
            # slice -- frame f starts at f * hop, with no negative index at
            # f = 0 and no overrun at the last frame.
            self.item_samples = ((self.n_item_frames - 1) * self.hop
                                 + self.meta["fft_bins"])
        else:
            self.meta = None
            self.hop = None
            self.item_samples = None

        # Flat index over every division: one entry per excerpt, holding which
        # division it lives in and which frame it starts at. Sharding across
        # workers is then just PyTorch handing out different i values.
        frames, divs = [], []
        n_slice = config["n_slice"]
        for div in range(self.ndivs):
            idx = self._load_idx(div)
            if n_slice > 1:
                # take every n_slice-th index, as before
                idx = idx[: (len(idx) // n_slice) * n_slice: n_slice]
            frames.append(idx.astype(np.int64))
            divs.append(np.full(len(idx), div, dtype=np.int16))
        self.frames = np.concatenate(frames)
        self.divs = np.concatenate(divs)

        # Opened lazily in the worker that first touches them -- see _slabs.
        self._open: dict = {}

    def _path(self, kind: str, name: str) -> Path:
        return self.base_path / kind / self.split / name

    def _legacy_path(self, kind: str, stem: str, div: int) -> Path:
        """Locate a legacy array, tolerating both division naming schemes.

        Commit bdb0e10 renamed these from `dataset_idx.npz` to
        `dataset_idx000.npz`. Stores written either side of that are both still
        around, so try the current name and fall back to the older one rather
        than making the caller know which vintage they have.
        """
        numbered = self._path(kind, f"{stem}{div:03d}.npz")
        if numbered.exists():
            return numbered
        legacy = self._path(kind, f"{stem}.npz")
        if legacy.exists():
            return legacy
        raise FileNotFoundError(
            f"no legacy array at {numbered} or {legacy}. If this store was "
            f"written by the current extractor, use feature_source='audio'.")

    def _load_meta(self, div: int) -> dict:
        with open(self._path("meta", f"dataset_meta{div:03d}.json")) as f:
            return json.load(f)

    def _load_idx(self, div: int) -> np.ndarray:
        if self.source == "audio":
            return np.load(self._path("idx", f"dataset_idx{div:03d}.npy"))
        return np.load(self._legacy_path("idx", "dataset_idx", div))["dataset_idx"]

    def _slabs(self, div: int) -> dict:
        """
            Per-division arrays, opened on first use.

            Opened here rather than in __init__ so each worker opens them
            itself. For "audio" that means each worker maps the files and the
            OS shares the pages; mapping in the parent would still work under
            fork, but under spawn the handles would have to survive pickling,
            and a np.memmap pickles by serialising its contents -- which would
            ship gigabytes to every worker. __getstate__ drops them for the same
            reason.

            For "legacy_feature" there is nothing to map: a .npz is a zip
            archive, so each worker reads its own full copy into RAM. That is
            the old memory behaviour -- roughly 3.5 GB per worker for MAPS train
            -- and is a large part of why the audio backend exists.
        """
        if div not in self._open:
            if self.source == "audio":
                self._open[div] = {
                    "audio": np.load(self._path("audio", f"dataset_audio{div:03d}.npy"),
                                     mmap_mode="r"),
                    "onset": np.load(self._path("label_onset", f"dataset_label_onset{div:03d}.npy"),
                                     mmap_mode="r"),
                    "offset": np.load(self._path("label_offset", f"dataset_label_offset{div:03d}.npy"),
                                      mmap_mode="r"),
                    "frames": np.load(self._path("label_frames", f"dataset_label_frames{div:03d}.npy"),
                                      mmap_mode="r"),
                    "velocity": np.load(self._path("label_velocity", f"dataset_label_velocity{div:03d}.npy"),
                                        mmap_mode="r"),
                }
            else:
                self._open[div] = {
                    name: np.load(self._legacy_path(kind, stem, div))[stem]
                    for name, (kind, stem) in LEGACY_ARRAYS.items()
                }
        return self._open[div]

    def __getstate__(self) -> dict:
        """Drop the open arrays so pickling the dataset stays cheap."""
        state = self.__dict__.copy()
        state["_open"] = {}
        return state

    def __len__(self):
        """
            Returns the length of the dataset.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
            Returns the spectrogram and labels at a given index (idx).

            Args
            ----
                idx (int): Index value

            Returns
            -------
                spec : spectrogram segment
                label_onset: Onset labels
                label_offset: Offset labels
                label_frames: Frame labels
                label_velocity: Velocity labels
        """
        div = int(self.divs[idx])
        frame = int(self.frames[idx])
        slabs = self._slabs(div)

        # for idx_feature_s or idx_feature_start, we need to subtract the margin_b
        # after getting our starting index
        idx_feature_s = frame - self.margin_b

        # idx_label_s and idx_label_e are the starting and ending indices for the labels
        idx_label_s = frame
        idx_label_e = frame + self.num_frame

        if self.source == "audio":
            # Copy out of the map: the mel needs a writable, contiguous buffer,
            # and the copy is 204 KB against a page cache hit.
            begin = idx_feature_s * self.hop
            audio = np.array(slabs["audio"][begin: begin + self.item_samples],
                             dtype=np.float32)

            if self.augmentator is not None:
                # The slab concatenates tracks, so there is no track name here --
                # but (split, div, frame) is just as stable an excerpt identity,
                # and excerpt_start is genuinely this excerpt's offset in seconds.
                audio = self.augmentator(
                    audio,
                    track_id=f"hft|{self.split}|{div}",
                    excerpt_start=idx_feature_s * self.hop / self.meta["sr"],
                    epoch=self.epoch,
                )

            # center=False because the buffer already carries the padding a
            # centred transform would add; this reproduces the matching slice of
            # the whole-track feature exactly. Proven in
            # scripts/verify_hft_frame_equivalence.py and, on the real slab, in
            # scripts/verify_hft_slab_layout.py.
            # mel_from_audio gives [frames, mel_bins]; transpose to match the old
            # a_feature -> spec convention of [n_feature, margin+num_frame+margin].
            spec = mel_from_audio(torch.from_numpy(audio), self.meta,
                                  center=False).T
        else:
            # a_feature: [margin+num_frame+margin, n_feature] -(transpose)->
            # spec: [n_feature, margin+num_frame+margin]
            idx_feature_e = frame + self.num_frame + self.config["margin_f"]
            spec = torch.from_numpy(
                np.array(slabs["feature"][idx_feature_s:idx_feature_e])).T

        # label_onset: [num_frame, n_note]
        label_onset = torch.from_numpy(np.array(slabs["onset"][idx_label_s:idx_label_e]))

        # label_offset: [num_frame, n_note]
        label_offset = torch.from_numpy(np.array(slabs["offset"][idx_label_s:idx_label_e]))

        # label_frames: [num_frame, n_note]
        # bool -> float
        label_frames = torch.from_numpy(
            np.array(slabs["frames"][idx_label_s:idx_label_e])).float()

        # label_velocity: [num_frame, n_note]
        # int8 -> long
        label_velocity = torch.from_numpy(
            np.array(slabs["velocity"][idx_label_s:idx_label_e])).long()

        return spec, label_onset, label_offset, label_frames, label_velocity
