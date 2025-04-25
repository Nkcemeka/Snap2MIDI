"""
    File: dataset_events.py
    Author: Chukwuemeka L. Nkama
    Date: 4/3/2025
    Description: A dataset class to load
                 audio segments for training
                 an architecture such as Onsets
                 and frames that requires onsets,
                 offsets, etc...!
"""

# Imports
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class SnapEventsDataset(Dataset):
    def __init__(self, emb_path=None):
        super().__init__()
        if emb_path is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        assert Path(emb_path).exists(), f"{emb_path} does not exist."
        self.emb_path = emb_path
        self.data = sorted(Path(self.emb_path).glob("*.npz"))

    def __getitem__(self, index):
        item_path = str(self.data[index])
        item = np.load(item_path)

        # Get the audio representation
        audio = item["audio"].astype(np.float32)
        audio = torch.from_numpy(audio)

        # Get the latent space representation
        feature = item["feature"].astype(np.float32)
        feature = torch.from_numpy(feature)

        # Get the roll representation
        label_frame = item["roll_frame"].astype(np.float32)
        label_onset = item["roll_onset"].astype(np.float32)
        label_offset = item["roll_offset"].astype(np.float32)
        label_velocity = item["roll_velocity"].astype(np.float32)

        return (feature, label_frame, label_onset, label_offset,\
                label_velocity, audio)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = SnapEventsDataset("./extractors/maestro/maestro_events_segments/train/")
    print(dataset[0][0].shape, dataset[0][1].shape)
