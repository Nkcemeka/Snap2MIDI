"""
    File: dataset_events.py
    Author: Chukwuemeka L. Nkama
    Date: 4/3/2025
    Description: A dataset class to load
                 audio segments for training
                 the Onsets and Frames model.
"""

# Imports
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Sequence, Optional

class OAFDataset(Dataset):
    def __init__(self, emb_paths: Optional[list[str]]=None, dataset_type: str="train") -> None:
        """
        Args:
            emb_paths (list): List of paths to npz files containing audio and feature data.
            dataset_type (str): Type of dataset to load. Options are "train", "val", "test".
        """
        super().__init__()
        if emb_paths is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        self.data = [] # path to npz files
        for emb_path in emb_paths:
            assert Path(emb_path).exists(), f"{emb_path} does not exist."
            self.emb_path = emb_path

            # Open dataset_type.txt file
            dataset_type_path = Path(emb_path)/ f"{dataset_type}.txt"
            with open(dataset_type_path, "r") as f:
                for line in f:
                    line = line.strip()
                    self.data.append(Path(emb_path + "/" + line + ".npz"))

    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        item_path = str(self.data[index])
        item = np.load(item_path)

        # Get the audio representation
        audio = item["audio"].astype(np.float32)
        audio = torch.from_numpy(audio)

        # Get the latent space representation
        feature = item["feature"].astype(np.float32)
        feature = torch.from_numpy(feature)

        # Get the roll representation
        label_frame = item["label_frames"].astype(np.float32)
        label_onset = item["label_onsets"].astype(np.float32)
        label_offset = item["label_offsets"].astype(np.float32)
        label_velocity = item["label_velocities"].astype(np.float32)

        return (feature, label_frame, label_onset, label_offset,\
                label_velocity, audio)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    dataset = OAFDataset(["./extractors/maestro_events_segments/train/"])
    print(dataset[0][0].shape, dataset[0][1].shape)
