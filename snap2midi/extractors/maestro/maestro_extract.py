"""
File: maestro_extract.py
Author: Chukwuemeka L. Nkama
Date: 4/2/2025

Description: Extracts Audio Segments, features and the corresponding labels
             for MAESTRO daaset!
"""

# Imports
import numpy as np
import pretty_midi
from pathlib import Path
import argbind 
from tqdm import tqdm
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures

@argbind.bind()
class MaestroExtractor:
    def __init__(self, path: str=None, window_size: float=1.0,
                 sample_rate: float=None, duration: float=6.0,
                 train_ratio: float=0.75, valid_ratio: float=0.15,
                 test_ratio: float=0.10, ext_audio: str="wav",
                 ext_midi: str="midi", hop_size: float=0.8,
                 cqt_num_octaves: int=None, cqt_bins_oct: int=None,
                 n_mels: int=None, mel_n_fft: int=None,
                 pr_rate: int=None, feature: str="mel", save_name: str="maestro_segments"):
        """
            Default constructor for the MaestroExtractor class.

            Args:
                path (str): Path to the MAESTRO dataset
                window_size (float): Window size for the audio segments
                sample_rate (float): Sample rate for the audio segments
                duration (float): Duration of all the audio segments together
                train_ratio (float): Ratio of training data
                valid_ratio (float): Ratio of validation data
                test_ratio (float): Ratio of testing data
                ext_audio (str): Extension of the audio files
                ext_midi (str): Extension of the midi files
                hop_size (float): Hop size for the audio segments
                cqt_num_octaves (int): Number of octaves for CQT
                cqt_bins_oct (int): Number of bins per octave for CQT
                n_mels (int): Number of mel bands
                mel_n_fft (int): Size of FFT window for mel spectrogram
                pr_rate (int): Rate of the audio segments
                feature (str): Feature to extract from the audio segments
                save_name (str): Name of the directory to save the segments

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
        self.cqt_num_octaves = cqt_num_octaves if cqt_num_octaves is not None else 6
        self.cqt_bins_oct = cqt_bins_oct if cqt_bins_oct is not None else 24
        self.n_mels = n_mels if n_mels is not None else 229
        self.mel_n_fft = mel_n_fft if mel_n_fft is not None else 2048
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
                labels (np.ndarray): Labels for the audio segment
        """
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
                    feature = self.get_feature(chunk, feature=self.feature)
                    feature = feature.T # (time, embedding)

                    if feature.shape[0] < label.shape[0]:
                        # raise error to show that the num of frames of 
                        # the feature is less than the num of frames
                        # of the label
                        raise RuntimeError(f"Feature shape {feature.shape} \
                                is less than label shape {label.shape}!")
                    elif feature.shape[0] > label.shape[0]:
                        # truncate the feature
                        feature = feature[:label.shape[0], :]
                        
                    store_path = f"./{self.save_name}/{split}/{str(file.stem)}_{idx}.npz" 
                    store_dict['audio'] = chunk
                    store_dict['roll'] = label
                    store_dict['feature'] = feature
                    np.savez(store_path, **store_dict) 
    
    def get_feature(self, audio, feature):
        """
            Get the feature for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
                feature (str): Feature to extract

            Returns:
                feature (np.ndarray): Feature for the audio segment
        """
        if feature not in ["mel", "cqt"]:
            raise ValueError(f"Feature {feature} not supported!")
        
        # Create the HandcraftedFeatures object
        hf = HandcraftedFeatures(sample_rate=self.sample_rate, \
                window_size=self.window_size, pr_rate=self.pr_rate)
        if feature == "mel":
            return hf.compute_mel(audio, \
                                  n_mels=self.n_mels, n_fft=self.mel_n_fft)
        elif feature == "cqt":
            return hf.compute_cqt(audio, bins_per_octave=self.cqt_bins_oct, \
                    num_octaves=self.cqt_num_octaves)
    


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        MaestroExtractor()
