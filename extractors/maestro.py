"""
File: maestro.py
Author: Chukwuemeka L. Nkama
Date: 4/2/2025

Description: Extracts Audio Segments and the 
             corresponding labels!
"""

# Imports
import torch
import numpy as np
from pathlib import Path
import argbind 
from tqdm import tqdm
from framed_signal import FramedAudio

@argbind.bind()
class MaestroExtractor:
    def __init__(self, path: str=None, window_size: float=1.0,
                 sample_rate: float=None, duration: float=6.0,
                 train_ratio: float=0.75, valid_ratio: float=0.15,
                 test_ratio: float=0.10):
        """
            Args:
                path (str): Base path to the MAESTRO dataset directory
                window_size (float): window size in seconds to consider
                sample_rate (float): sample rate to use
                duration (float): Total duration of the audio segments
                              generated in hours!
                train_ratio (float): Training ratio
                valid_ratio (float): Validation ratio
                test_ratio (float): Testing ratio

            Returns:
                None
        """
        super().__init__()

        if path is None:
            raise ValueError(f"Path is None! Cannot Instantiate {self.__class__.__name__}!")
        self.base_path = Path(path)
        self.years = [d.name for d in self.base_path.iterdir() \
                if d.is_dir()]
        self.duration = duration
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.extract()

    def extract(self):
        """
            Extract the features and the corresponding
            labels!

            Args:
                None 

            Returns
                None
        """
        splits = ['train', 'valid', 'test']

        # Get files for each split
        train = self.years[:int(self.train_ratio*len(self.years))]
        rem = self.years[int(self.train_ratio*len(self.years)):]
        valid = rem[:int((self.valid_ratio)/(self.valid_ratio + test_ratio))]
        test = rem[int((self.valid_ratio)/(self.valid_ratio + test_ratio)):]

        # Create the needed directories
        Path("maestro_data/train/features").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/train/labels").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/valid/features").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/valid/labels").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/test/features").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/test/labels").mkdir(parents=True, \
                exist_ok=True)
        

        print(f"Extracting features and labels...")
        for split in splits:
            print(f"Processing {split} split...")
            self.extract_split(train, split)

        for year in tqdm(self.years, total=len(self.years)):
            pass

    def extract_split(self, years, split):
        """
            Get the duration split which denotes 
            the hours for the split! The number of
            segments extracted per year is the 
        """
        if split == 'train':
            duration_split = self.train_ratio * self.duration
        elif split == 'valid':
            duration_split = self.valid_ratio * self.duration
        else:
            duration_split = self.test_ratio * self.duration

        for year in tqdm(years, total=len(files)): 
            pass
        pass

    
    def __getitem__(self):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroExtractor()
