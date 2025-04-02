"""
File: maestro.py
Author: Chukwuemeka L. Nkama
Date: 4/2/2025

Description: Contains Dataset class to load
             the MAESTRO dataset!
"""

# Imports
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import argbind 
from tqdm import tqdm

@argbind.bind()
class MaestroDataset(Dataset):
    def __init__(self, path: str=None):
        super().__init__()

        if path is None:
            raise ValueError(f"Path is None! Cannot Instantiate {self.__class__.__name__}!")
        self.base_path = Path(path)
        self.years = [d.name for d in self.base_path.iterdir() \
                if d.is_dir()]
        self.extract()

    def extract(self):
        print(f"Extracting features and labels...")
        for year in self.years:
            pass

    
    def __getitem__(self):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroDataset()
