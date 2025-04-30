"""
    File: dataset_shallow.py
    Author: Chukwuemeka L. Nkama
    Date: 4/3/2025
    Description: A dataset class to load
                 audio segments for training
                 a shallow transcriber!
"""

# Imports
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ShallowDataset(Dataset):
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
        label = item["roll"].astype(np.float32)
        label = torch.from_numpy(label)
        return (feature, label, audio)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = ShallowDataset("./extractors/maestro/maestro_segments/train/")
    print(dataset[0][0].shape, dataset[0][1].shape)