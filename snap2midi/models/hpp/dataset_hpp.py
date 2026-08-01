# Imports
import hashlib
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Sequence, Optional

class HPPDataset(Dataset):
    def __init__(self, config: dict, emb_paths: Optional[list[str]]=None,
                 augmentator=None) -> None:
        """
            Dataset class to load training segments for
            the Onsets and Frames model V2.

        Args
        ----
            emb_paths (list):
                List of paths to npz files containing data for training.
            augmentator (Augmentator | None):
                Applied to the excerpt's waveform before it is returned. The
                model computes its own spectrogram from that waveform, so
                nothing else has to change; the labels are read from the store
                and are untouched. Leave None for a clean baseline.
        """
        super().__init__()
        if emb_paths is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        self.sequence_length = config["sequence_length"]
        self.crop_seed = config.get("seed", 42)
        self.hop_length = config["hop_length"]
        self.augmentator = augmentator

        # Only consulted when augmenting, so a config without it stays valid for
        # the clean path rather than failing on a key it never needed.
        self.sample_rate = config.get("sample_rate")
        if augmentator is not None and self.sample_rate is None:
            raise ValueError(
                "augmentation needs config['sample_rate'] to convert the "
                "excerpt offset into seconds for the augmentator's seed.")

        # Set by the epoch callback so augmentation is redrawn each epoch. Left
        # at 0 every pass applies the identical draw to the identical excerpt --
        # a fixed pre-corrupted dataset rather than augmentation, and it fails
        # silently. See EpochUpdateCallback in train_hpp.py.
        self.epoch = 0
        self.data = [] # path to files
        for emb_path in emb_paths:
            assert Path(emb_path).exists(), f"{emb_path} does not exist."
            self.emb_path = emb_path
            self.data.extend(sorted(Path(emb_path).glob("*.npz")))


    def _crop_start(self, item_path: str, audio_length: int) -> int:
        """Choose this item's excerpt for this epoch.

        Derived from (seed, track, epoch) rather than drawn from a RandomState
        held on the dataset. That state lives in the main process and is only
        ever advanced inside the worker copies, so with num_workers > 0 -- the
        default -- every epoch re-forked from an un-advanced generator and
        replayed the identical crop: one frozen window per track for the whole
        run. At the default 20.48 s that is 5.5 h of MAESTRO's ~159 h train
        split, and the model never saw the other 96.6%. It failed silently, since a frozen crop is
        indistinguishable from a working one at any single step.

        Deriving the draw instead makes it vary per epoch, reproduce exactly
        from the same (track, epoch), and not depend on how many workers happen
        to be running -- the same three properties the augmentator's own seed
        derivation buys, for the same reasons.
        """
        span = audio_length - self.sequence_length
        if span <= 0:
            # Track shorter than one excerpt: take it from the start.
            return 0
        key = f"{self.crop_seed}|{item_path}|{self.epoch}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % span

    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        """ 
            Gets item from dataset based on index.

            Args
            ----
                index (int): Index 
            
            Returns
            -------
                None
        """
        item_path = str(self.data[index])
        item = np.load(item_path)
        result = {}

        if self.sequence_length is not None:
            audio_length = len(item["audio"])
            frame_start = self._crop_start(item_path, audio_length) // self.hop_length
            n_frames = self.sequence_length // self.hop_length
            frame_end = frame_start + n_frames

            start_samples = frame_start * self.hop_length
            end_samples = start_samples + self.sequence_length

            result["audio"] = item["audio"][start_samples:end_samples].astype(np.float32)
            result["label"] = item["label"][frame_start:frame_end, :]
            result["velocity"] = item["velocity"][frame_start:frame_end, :]
        else:
            start_samples = 0
            result["audio"] = item["audio"].astype(np.float32)
            result["label"] = item["label"]
            result["velocity"] = item["velocity"]

        if self.augmentator is not None:
            # Applied to the excerpt rather than the whole track: cheaper, and
            # it matches what kong does. Seeded from the excerpt's identity
            # rather than call order, so the same excerpt in the same epoch
            # draws the same augmentation whatever the batch size or worker
            # count. The labels above are read from the store and are unaffected
            # -- every stage of the pipeline is label-preserving.
            result["audio"] = self.augmentator(
                result["audio"],
                track_id=Path(item_path).stem,
                excerpt_start=start_samples / self.sample_rate,
                epoch=self.epoch,
            )

        result['onset'] = (result['label'] == 3).astype(np.float32)
        result['offset'] = (result['label'] == 1).astype(np.float32)
        result['frame'] = (result['label'] > 1).astype(np.float32)
        result['velocity'] - (result["velocity"]).astype(np.float32)
        return result

    def __len__(self) -> int:
        """ 
            Length of dataset.

            Args
            ----
                length (int): Length of data
        """
        return len(self.data)

if __name__ == "__main__":
    dataset = HPPDataset(["./extractors/maestro_events_segments/train/"])
    print(dataset[0][0].shape, dataset[0][1].shape)
