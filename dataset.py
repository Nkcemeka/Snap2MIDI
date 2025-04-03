"""
    File: dataset.py
    Author: Chukwuemeka L. Nkama
    Date: 4/3/2025
    Description: A dataset class to load
                 audio segments for training!
"""

# Imports
import pretty_midi
from pathlib import Path
import torch
from torch.utils.data import Dataset


class SnapDataset(Dataset):
    def __init__(self, emb_path=None):
        super().__init__()
        if emb_path is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")
        self.emb_path = emb_path
        self.data = sorted(Path(self.emb_path).glob("*.npz"))

    def __getitem__(self, index):
        item_path = str(self.data[index])
        item = np.load(item_path)
        return item 

    def __len__(self):
        return len(self.data)
