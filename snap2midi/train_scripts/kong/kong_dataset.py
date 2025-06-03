"""
    File: kong_dataset.py
    Author: Chukwuemeka L. Nkama
    Date: 6/2/2025
    Description: A dataset class to load
                 audio segments for training
                 the Kong model.
"""

# Imports
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Sequence, Optional

class KongDataset(Dataset):
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

    def __getitem__(self, index: int) -> dict:
        item_path = str(self.data[index])
        item = np.load(item_path)
        item_dict: dict = {}

        for key in item.keys():
            item_dict[key] = item[key]
        return item_dict

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    dataset = KongDataset(["./extractors/maestro/maestro_events_segments/"])
    print(dataset[0][0].shape, dataset[0][1].shape)
