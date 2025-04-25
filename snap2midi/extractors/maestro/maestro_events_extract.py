"""
File: maestro_extract.py
Author: Chukwuemeka L. Nkama
Date: 4/2/2025

Description: Extracts Audio Segments, features and the corresponding labels
             for the MAESTRO dataset! The labels consider all events: velocity,
             onsets, offsets and frames.
"""

# Imports
import numpy as np
import pretty_midi
from pathlib import Path
import argbind 
from tqdm import tqdm
from .. import FramedAudio
from .. import HandcraftedFeatures

@argbind.bind()
class MaestroEventsExtractor:
    def __init__(self, path: str=None, window_size: float=1.0,
                 sample_rate: float=None, duration: float=6.0,
                 train_ratio: float=0.75, valid_ratio: float=0.15,
                 test_ratio: float=0.10, ext_audio: str="wav",
                 ext_midi: str="midi", hop_size: float=0.8,
                 pr_rate: int=None, feature: str="mel", save_name: str="maestro_events_segments"):
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
                feature (str): Feature to extract
                hop_size (float): Hop size in seconds

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
        self.feature = feature
        self.save_name = save_name
        if pr_rate is None:
            self.pr_rate = sample_rate
        else:
            self.pr_rate = pr_rate
        self.extract()

    def get_label(self, midi, duration, hop_size, idx):
        """
            Get the labels for a given audio segment
            Args:
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                duration (float): Duration of the audio segment
                hop_size (float): Hop size in seconds
                idx (int): Index of the audio segment

            Returns:
                np.ndarray: Labels for the audio segment
        """
        num_frames = int(duration * self.pr_rate)
        start = idx * hop_size
        end = start + duration
        label_frames = np.zeros((num_frames, 128))
        label_onsets = np.zeros((num_frames, 128))
        label_offsets = np.zeros((num_frames, 128))
        label_velocities = np.zeros((num_frames, 128))

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue
                    pitch = note.pitch
                    start_frame = max(0, int(self.pr_rate * (note.start-start)))
                    end_frame = min(num_frames, int(self.pr_rate * (note.end - start)))

                    onset_end = min(num_frames, start_frame + 1)
                    offset_end = min(num_frames, end_frame + 1)
                    # label onsets
                    label_onsets[start_frame:onset_end, pitch] = 1

                    # label frames
                    label_frames[onset_end:end_frame, pitch] = 1

                    # label offsets
                    label_offsets[end_frame:offset_end, pitch] = 1

                    # label velocities
                    label_velocities[start_frame:end_frame, pitch] = note.velocity
        return label_frames, label_onsets, label_offsets, label_velocities

    def extract(self):
        """
            Extract the audio segments, features and the corresponding
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
        Path(f"{self.save_name}/train/").mkdir(parents=True, \
                exist_ok=True)
        Path(f"{self.save_name}/valid/").mkdir(parents=True, \
                exist_ok=True)
        Path(f"{self.save_name}/test/").mkdir(parents=True, \
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

                assert int(num_chunks) < len_chunks, f"Num chunks {int(num_chunks)} is greater than \
                        len chunks {len_chunks} for file {file}!"

                # chunk indices to select
                idxs = np.random.choice(len_chunks, int(num_chunks), replace=False)
                for idx in idxs:
                    # select the correpsonding chunks and 
                    # save them in a dict alongside the info to 
                    # extract the label for the chunk.
                    # For the roll, we need starting index (idx), 
                    # sample_rate to get length of audio segment 
                    # and hop_size to know the start
                    # time using idx
                    store_dict = {'audio': None}

                    chunk = audio_chunks[idx]
                    label_frames, label_onsets, label_offsets, \
                         label_velocities = self.get_label(midi, self.window_size, \
                            self.hop_size, idx)
                    feature = self.get_feature(chunk, feature=self.feature)
                    feature = feature.T # (time, embedding)

                    if feature.shape[0] < label_frames.shape[0]:
                        # raise error to show that the num of frames of 
                        # the feature is less than the num of frames
                        # of the label
                        raise RuntimeError(f"Feature shape {feature.shape} \
                                is less than label shape {label_frames.shape}!")
                    elif feature.shape[0] > label_frames.shape[0]:
                        # truncate the feature
                        feature = feature[:label_frames.shape[0], :]

                    
                    # Save the audio segment and the labels
                    # in a npz file
                    store_path = f"./{self.save_name}/{split}/{str(file.stem)}_{idx}.npz" 
                    store_dict['audio'] = chunk
                    store_dict['roll_frame'] = label_frames
                    store_dict['roll_onset'] = label_onsets
                    store_dict['roll_offset'] = label_offsets
                    store_dict['roll_velocity'] = label_velocities
                    store_dict['feature'] = feature
                    np.savez(store_path, **store_dict) 
    
    def get_feature(self, audio, feature):
        """
            Get the feature for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
                feature (str): Feature to extract

            Returns:
                np.ndarray: Feature of the audio segment
        """
        if feature not in ["mel", "cqt"]:
            raise ValueError(f"Feature {feature} not supported!")
        
        # Create the HandcraftedFeatures object
        hf = HandcraftedFeatures(sample_rate=self.sample_rate, \
                window_size=self.window_size, pr_rate=self.pr_rate)
        if feature == "mel":
            return hf.compute_mel(audio)
        elif feature == "cqt":
            return hf.compute_cqt(audio)


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroEventsExtractor()
