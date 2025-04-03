"""
File: maestro_extract.py
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
                 test_ratio: float=0.10, ext_audio: str="wav",
                 ext_midi: str="midi", hop_size: float=0.8):
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
        if path is None:
            raise ValueError(f"Path is None! Cannot Instantiate {self.__class__.__name__}!")
        self.base_path = Path(path)
        self.years = [d.name for d in self.base_path.iterdir() \
                if d.is_dir()]
        self.duration = duration
        self.window_size = window_size
        self.sample_rate = sample_rate if sample_rate is not None else 22050
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.hop_size = hop_size
        self.ext_audio = ext_audio
        self.ext_midi = ext_midi
        self.extract()

    def extract(self):
        """
            Extract the audio segments and the corresponding
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
        valid = rem[:int((self.valid_ratio)/(self.valid_ratio + self.test_ratio))]
        test = rem[int((self.valid_ratio)/(self.valid_ratio + self.test_ratio)):]

        # Create the needed directories
        Path("maestro_data/train/segments").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/valid/segments").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_data/test/segments").mkdir(parents=True, \
                exist_ok=True)
        

        print(f"Extracting Audio segments and labels...")
        for split in splits:
            print(f"Processing {split} split...")
            self.extract_split(train, split)

    def extract_split(self, years, split):
        """
            Get the duration split which denotes 
            the hours for the split! The duration_split_year
            helps us get the number of segments to be extracted
            for each year!
        """
        if split == 'train':
            duration_split = self.train_ratio * self.duration * 3600
        elif split == 'valid':
            duration_split = self.valid_ratio * self.duration * 3600
        else:
            duration_split = self.test_ratio * self.duration * 3600

        # Get the duration split per year
        duration_split_year = duration_split/len(years)

        for year in tqdm(years, total=len(years)): 
            files_year = sorted((self.base_path / f"{year}").glob(f"*.{self.ext_audio}"))
            labels_year = sorted((self.base_path / f"{year}").glob(f"*.{self.ext_midi}"))

            num_files_year = len(files_year)
            duration_file = duration_split_year / num_files_year
            duration_file_samples = duration_file * self.sample_rate

            for file in files_year: 
                assert file.exists(), f"{file} does not exist!"
                audio_chunks = FramedAudio(str(file), self.hop_size, \
                        self.window_size, sample_rate=self.sample_rate)
                len_chunks = len(audio_chunks) # length of audio chunks

                # we need duration_file_samples from this file
                # and each chunk is self.window_size * self.sample_rate
                num_chunks = duration_file_samples // (self.window_size * self.sample_rate)

                # chunk indices to select
                idxs = np.random.choice(len_chunks, int(num_chunks))
                for idx in idxs:
                    # select the correpsonding chunks and 
                    # save them in a dict alongside the label
                    # for the chunk
                    store_dict = {'audio': None, 'roll': None}
                    chunk = audio_chunks[idx]
                    store_path = f"./maestro_data/{split}/segments/{str(file.stem)}_{idx}.npz" 
                    np.savez(store_path, **store_dict) 


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroExtractor()
