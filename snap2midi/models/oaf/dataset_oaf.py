# Imports
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Sequence, Optional

from snap2midi.extractor.utils.handcrafted_features import HandcraftedFeatures

# What the extractor called compute_mel/compute_cqt with, so that recomputing a
# feature here reproduces the stored one exactly. See _compute_feature.
FEATURE_KEYS = ("feature", "sample_rate", "frame_rate", "max_frame_secs",
                "n_mels", "mel_n_fft", "hop_length", "cqt_bins_oct",
                "cqt_num_octaves")


class OAFDataset(Dataset):
    def __init__(self, emb_paths: Optional[list[str]]=None, augmentator=None,
                 feature_config: Optional[dict]=None) -> None:
        """
            Dataset class to load training segments for
            the Onsets and Frames model.

        Args
        ----
            emb_paths (list):
                List of paths to npz files containing audio and feature data.
            augmentator (Augmentator | None):
                Applied to the excerpt's waveform, after which the feature is
                recomputed from the augmented audio. OAF is the one model here
                that trains on a feature stored at extraction time rather than
                on the waveform, so augmenting the audio alone would be a silent
                no-op. Leave None to serve the stored feature untouched.
            feature_config (dict | None):
                The feature parameters the store was extracted with. Required
                when augmentating, because the OAF extractor writes no
                extraction config for the dataset to read back.
        """
        super().__init__()
        if emb_paths is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        self.augmentator = augmentator
        self.feature_config = feature_config or {}

        # Set by the epoch callback so augmentation is redrawn each epoch. Left
        # at 0 every pass applies the identical draw to the identical excerpt --
        # a fixed pre-corrupted dataset rather than augmentation, and it fails
        # silently. See EpochUpdateCallback in train_oaf.py.
        self.epoch = 0

        self.data = [] # path to files
        for emb_path in emb_paths:
            assert Path(emb_path).exists(), f"{emb_path} does not exist."
            self.emb_path = emb_path
            self.data.extend(sorted(Path(emb_path).glob("*.npz")))

        if self.augmentator is not None:
            self._check_feature_params()

    def _compute_feature(self, audio: np.ndarray) -> np.ndarray:
        """Recompute the stored feature from a waveform.

        Mirrors OAFMode._get_feature exactly, including the transpose to
        (time, embedding). Any drift between the two makes the augmented run
        train on a different representation than the clean one, which is why
        _check_feature_params measures the agreement rather than trusting it.
        """
        config = self.feature_config
        kind = config["feature"]
        hf = HandcraftedFeatures(sample_rate=config["sample_rate"],
                                 window_size=config["max_frame_secs"],
                                 frame_rate=config["frame_rate"])
        if kind == "mel":
            feature = hf.compute_mel(audio, n_mels=config["n_mels"],
                                     n_fft=config["mel_n_fft"],
                                     hop_length=config.get("hop_length"))
        elif kind == "cqt":
            feature = hf.compute_cqt(audio, bins_per_octave=config["cqt_bins_oct"],
                                     num_octaves=config["cqt_num_octaves"],
                                     hop_length=config.get("hop_length"))
        else:
            raise ValueError(f"unsupported feature {kind!r} for recomputation")
        return feature.T

    def _check_feature_params(self) -> None:
        """Confirm feature_config actually describes how this store was made.

        The OAF extractor saves no extraction config, so the parameters have to
        be supplied by the caller and nothing else would catch them being wrong.
        A mismatch is not a crash -- it silently trains the augmented arm on a
        different representation than the baseline, which reads later as
        "augmentation hurt". Recomputing one item from its own clean audio and
        comparing against the stored feature turns that into an error here.
        """
        missing = [k for k in ("feature", "sample_rate", "frame_rate", "max_frame_secs")
                   if k not in self.feature_config]
        if missing:
            raise ValueError(
                f"augmenting OAF needs feature_config entries {missing}; the "
                "extractor writes no config for the dataset to read back, so "
                "they have to match what extract_oaf was called with.")
        if not self.data:
            return

        item = np.load(str(self.data[0]))
        stored = item["feature"].astype(np.float32)
        rebuilt = self._compute_feature(item["audio"].astype(np.float32))

        if stored.shape != rebuilt.shape:
            raise ValueError(
                f"recomputed feature {rebuilt.shape} does not match the stored "
                f"{stored.shape}. feature_config does not describe this store; "
                "check sample_rate, frame_rate, max_frame_secs and hop_length "
                "against the extract_oaf call that wrote it.")

        # float32 melspectrogram over ~600 frames: agreement is to roughly
        # 1e-5 relative, so the tolerance catches a wrong parameter without
        # tripping on librosa's own arithmetic.
        deviation = float(np.max(np.abs(stored - rebuilt)))
        scale = float(np.max(np.abs(stored))) or 1.0
        if deviation / scale > 1e-3:
            raise ValueError(
                f"recomputed feature differs from the stored one by "
                f"{deviation:.4g} ({deviation/scale:.2%} of full scale). "
                "feature_config does not match how this store was extracted, "
                "so augmented and clean runs would train on different "
                "representations.")

    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        """ 
            Gets item from dataset based on index.

            Args
            ----
                index (int): Index 
            
            Returns
            -------
                feature (np.ndarray): Feature for training
                label_frame (np.ndarray): Frame labels
                label_onset (np.ndarray): Onset labels
                label_velocity (np.ndarray): Label for velocity
                label_weights (np.ndarray): Label weights
                audio (np.ndarray): audio waveform
        """
        item_path = str(self.data[index])
        item = np.load(item_path)

        # Get the audio representation
        audio = item["audio"].astype(np.float32)

        if self.augmentator is not None:
            # Unlike the other models here, OAF trains on the stored feature, so
            # the feature has to be rebuilt from the augmented waveform --
            # augmenting the audio alone would change nothing the model sees.
            # Seeded from the excerpt's identity rather than call order, so the
            # same excerpt in the same epoch draws the same augmentation
            # whatever the batch size or worker count. Each npz is one excerpt,
            # so its offset within the excerpt is 0. The labels below come from
            # the store and are unaffected.
            audio = self.augmentator(audio, track_id=Path(item_path).stem,
                                     excerpt_start=0.0, epoch=self.epoch)
            feature = self._compute_feature(audio).astype(np.float32)
        else:
            # Get the latent space representation
            feature = item["feature"].astype(np.float32)

        audio = torch.from_numpy(audio)
        feature = torch.from_numpy(feature)

        # Get the roll representation
        label_frame = item["label_frames"].astype(np.float32)
        label_onset = item["label_onsets"].astype(np.float32)
        label_velocity = item["label_velocities"].astype(np.float32)
        label_weights = item["label_weights"].astype(np.float32)

        return (feature, label_frame, label_onset,\
                label_velocity, label_weights, audio)

    def __len__(self) -> int:
        """ 
            Length of dataset.

            Args
            ----
                length (int): Length of data
        """
        return len(self.data)

if __name__ == "__main__":
    dataset = OAFDataset(["./extractors/maestro_events_segments/train/"])
    print(dataset[0][0].shape, dataset[0][1].shape)
