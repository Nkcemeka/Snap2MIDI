"""
File: maestro_extract.py
Author: Chukwuemeka L. Nkama
Date: 4/2/2025

Description: Extracts Audio Segments!
"""

# Imports
import torch
import numpy as np
import pretty_midi
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
                 ext_midi: str="midi", hop_size: float=0.8,
                 pr_rate: int=None):
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
                ext_midi (str): Midi extenstion
                ext_audio (str): Audio extension
                pr_rate (int): Number of frames per second for piano roll

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
        if pr_rate is None:
            self.pr_rate = sample_rate
        else:
            self.pr_rate = pr_rate
        self.extract()

    def get_label(self, midi, duration, hop_size, idx):
        num_frames = int(duration * self.pr_rate)
        start = idx * hop_size
        end = start + duration
        labels = np.zeros((num_frames, 128))

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue
                    pitch = note.pitch
                    start_frame = max(0, int(self.pr_rate * (note.start-start)))
                    end_frame = min(num_frames, int(self.pr_rate * (note.end - start)))
                    labels[start_frame:end_frame, pitch] = 1
        return labels

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

        # I round the below for valid and test as using int leads to
        # test being 2 and valid 1 given the default ratios and if 
        # len(years) is 10
        valid = rem[:round((self.valid_ratio)*len(rem)/(self.valid_ratio + self.test_ratio))]
        test = rem[round((self.valid_ratio)*len(rem)/(self.valid_ratio + self.test_ratio)):]

        # Create the needed directories
        Path("maestro_segments/train/").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_segments/valid/").mkdir(parents=True, \
                exist_ok=True)
        Path("maestro_segments/test/").mkdir(parents=True, \
                exist_ok=True)
        

        print(f"Extracting Audio segments...")
        for i, split in enumerate(splits):
            print(f"Processing {split} split...")
            if split == "train":
                dataset = train
            elif split == "valid":
                dataset = valid
            else:
                dataset = test

            self.extract_split(dataset, split)

    def extract_split(self, years, split):
        """
            Extract all the audio segments for a given split!
            Get the duration split which denotes the hours for 
            the split! The duration_split_year helps us get the 
            number of segments to be extracted for each year!

            Args:
                years (List(str)): List of years for a split
                split (str): Training, Testing or Validation split
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

            for file in tqdm(files_year, total=len(files_year)): 
                label_path = file.parent / f"{file.stem}.{self.ext_midi}"
                assert file.exists(), f"{file} does not exist!"
                assert label_path.exists(), f"{label_path} does not exist!"
                midi = pretty_midi.PrettyMIDI(str(label_path))

                # Get audio chunks for file
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
                    # save them in a dict alongside the info to 
                    # extract the label for the chunk.
                    # For the roll, we need starting index (idx), 
                    # sample_rate to get length of audio segment 
                    # and hop_size to know the start
                    # time using idx
                    store_dict = {'audio': None, 
                             'roll': None}

                    chunk = audio_chunks[idx]
                    label = self.get_label(midi, self.window_size, \
                            self.hop_size, idx)
                    store_path = f"./maestro_segments/{split}/{str(file.stem)}_{idx}.npz" 

                    store_dict['audio'] = chunk
                    store_dict['roll'] = label
                    np.savez(store_path, **store_dict) 


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroExtractor()
