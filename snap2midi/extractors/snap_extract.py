"""
File: snap_extract.py
Author: Chukwuemeka L. Nkama
Date: 2025-02-05

Description: This file extracts audio segments, features and labels for a collection 
             of audio and MIDI files.
"""

# Imports
import numpy as np
import pretty_midi
import yaml
from pathlib import Path
import argbind 
import jams
import random
from tqdm import tqdm
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.extractors.utils.conv_jams_midi import jams_to_midi
from typing import Sequence


@argbind.bind()
class SnapExtractor:
    """
        Class to extract audio segments, features and labels 
         from any given dataset. This assumes that we have
        two lists of files, one for audio and the other for MIDI.
    """

    def __init__(self, path: str=None, window_size: float=1.0, sample_rate: int | float=None, 
                 duration: int | float=6.0, train_split: float=0.8, val_split: float=0.1, 
                 test_split: float=0.1, ext_audio: str="wav", ext_midi: str="midi",
                 hop_size: float=0.8, pr_rate: int | None=None, feature: str="mel", 
                 feature_params: dict | None =None, dataset_name: str="maestro", \
                 save_name: str="gset_segments") -> None:
        """
            Args:
                path (str): Path to the dataset
                window_size (float): Window size for the audio segments
                sample_rate (float): Sample rate for the audio segments
                duration (float): Duration of the audio segments to extract
                ext_audio (str): Extension of the audio files
                ext_midi (str): Extension of the midi files
                hop_size (float): Hop size for the audio segments
                pr_rate (int): Rate at which to extract features
                feature (str): Feature to extract from the audio segments
                feature_params (dict): Parameters for the feature extraction
                dataset_name (str): Name of the dataset
                train_split (float): Split for the training set
                val_split (float): Split for the validation set
                test_split (float): Split for the test set
                save_name (str): Name of the folder to save the extracted files

            Returns:
                None
        """
        # Check if the path is None
        if path is None:
            raise ValueError(f"Path is None! Cannot Instantiate {self.__class__.__name__}!")
        
        # Check if the path exists
        if not Path(path).exists():
            raise ValueError(f"Path {path} does not exist!")
        
        self.hop_size = hop_size
        self.window_size = window_size
        self.ext_audio = ext_audio
        self.ext_midi = ext_midi
        self.duration = duration
        self.sample_rate = sample_rate
        self.dataset_name = dataset_name
        self.save_name = save_name
        self.feature = feature
        self.feature_params = feature_params
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Check if the splits are valid
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError(f"Splits do not add up to 1.0! \
                    Train split: {self.train_split}, Val split: {self.val_split}, \
                    Test split: {self.test_split}")

        if pr_rate is None:
            self.pr_rate = sample_rate
        else:
            self.pr_rate = pr_rate
        
        if feature_params is None:
            self.feature = "mel"
            self.feature_params = {"n_mels": 229, "mel_n_fft": 2048}

        if self.dataset_name == "maestro":
            self.data = self.get_files_maestro(path)
        elif self.dataset_name == "guitarset":
            self.data = self.get_files_guitarset(path)
        elif self.dataset_name == "musicnet":
            self.data = self.get_files_musicnet(path)
        elif self.dataset_name == "slakh":
            self.data = self.get_files_slakh(path)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Create the save directory if it does not exist
        Path(self.save_name).mkdir(parents=True, exist_ok=True)

        # Perform the extraction
        self.extract()

        # After extraction, perform training, validation and test split
        # store the names of the files in a text file
        # for train, val and test
        self.split()
    
    def split(self):
        """
            Split the data into train, val and test sets
            and save the names of the files in a text file
            for each set.

            Args:
                None

            Returns:
                None
        """
        emb_path = self.save_name
        self.embeddings = sorted(Path(emb_path).glob("*.npz"))

        # Randomly shuffle the embeddings
        random.shuffle(self.embeddings)

        # Get the number of files
        num_files = len(self.embeddings)

        # Get the number of files for each set
        num_train = int(num_files * self.train_split)
        num_val = int(num_files * self.val_split)
        num_test = int(num_files * self.test_split)
        
        # Get the train, val and test sets
        train_set = self.embeddings[:num_train]
        val_set = self.embeddings[num_train:num_train+num_val]
        test_set = self.embeddings[num_train+num_val:num_train+num_val+num_test]

        # Save the train, val and test sets to a text file
        with open(f"{emb_path}/train.txt", "w") as f:
            for file in train_set:
                f.write(str(file.stem) + "\n")
        
        with open(f"{emb_path}/val.txt", "w") as f:
            for file in val_set:
                f.write(str(file.stem) + "\n")
        
        with open(f"{emb_path}/test.txt", "w") as f:
            for file in test_set:
                f.write(str(file.stem) + "\n")


    def extract(self):
        """
            Extract the audio segments, features and labels from the audio and midi files

            Args:
                None

            Returns:
                None
        """
        audio_files = self.data[0]
        midi_files = self.data[1]
        total_duration = 0 # Track the accumulated duration of the audio segments
        for audio_file, midi_file in zip(audio_files, midi_files):
            assert audio_file.exists(), f"{audio_file} does not exist"
            assert midi_file.exists(), f"{midi_file} does not exist"

            # Load the MIDI file
            midi = pretty_midi.PrettyMIDI(str(midi_file))

            # Get audio chunks for file
            audio_chunks = FramedAudio(str(audio_file), self.hop_size, \
                    self.window_size, sample_rate=self.sample_rate)
            len_chunks = len(audio_chunks) # length of audio chunks

            for idx, chunk in tqdm(enumerate(audio_chunks), desc="Extracting chunks", total=len_chunks):
                # Get the correpsonding chunks and 
                # save them in a dict alongside the info to 
                # extract the label for the chunk.
                # For the roll, we need starting index (idx), 
                # sample_rate to get length of audio segment 
                # and hop_size to know the start
                # time using idx

                # Get the duration of the chunk and add it to the total duration
                # to check if we have exceeded the total duration needed for all
                # audio segments
                total_duration += (len(chunk) / self.sample_rate)

                if total_duration > (self.duration*3600):
                    print(f"Quitting....")
                    print(f"Extraction complete! Total duration: {total_duration - (\
                        len(chunk) / self.sample_rate)} seconds")
                    return

                store_dict = {'audio': None, 'roll': None, 'roll_frame': None, 
                'roll_onset': None, 'roll_offset': None, 'roll_velocity': None,
                'feature': None}  

                label_frames, label_onsets, label_offsets, \
                label_velocities, label_roll = self.get_label_events(midi, self.window_size, \
                self.hop_size, idx)

                feature = self.get_feature(chunk, feature=self.feature)
                feature = feature.T # (time, embedding)

                feature = self.trunc_feature(feature, label_frames)
                    
                if self.dataset_name != "slakh":
                    store_path = f"./{self.save_name}/{str(audio_file.stem)}_{idx}.npz" 
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"./{self.save_name}/{track_name}_{str(audio_file.stem)}_{idx}.npz"

                store_dict['audio'] = chunk
                store_dict['feature'] = feature
                store_dict['roll'] = label_roll
                store_dict['roll_frame'] = label_frames
                store_dict['roll_onset'] = label_onsets
                store_dict['roll_offset'] = label_offsets
                store_dict['roll_velocity'] = label_velocities
                np.savez(store_path, **store_dict)
    
        print(f"Extraction complete! Total duration: {total_duration} seconds")
    
    def get_label(self, midi: pretty_midi.PrettyMIDI, \
                  duration: int | float, hop_size: float, idx: int) -> np.ndarray:
        """
            Get the labels for a given audio segment
            Args:
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                duration (float): Duration of the audio segment
                hop_size (float): Hop size in seconds
                idx (int): Index of the audio segment

            Returns:
                label (np.ndarray): Label for the audio segment
        """
        num_frames = int(duration * self.pr_rate)
        start = idx * hop_size
        end = start + duration
        label = np.zeros((num_frames, 128))

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue
                    pitch = note.pitch
                    start_frame = max(0, int(self.pr_rate * (note.start-start)))
                    end_frame = min(num_frames, int(self.pr_rate * (note.end - start)))
                    label[start_frame:end_frame, pitch] = 1
        return label

    def get_label_events(self, midi: pretty_midi.PrettyMIDI, \
                         duration: int | float, hop_size: float, idx: int) -> \
                            Sequence[np.ndarray]:
        """
            Get the label events for a given audio segment
            Args:
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                duration (float): Duration of the audio segment
                hop_size (float): Hop size in seconds
                idx (int): Index of the audio segment

            Returns:
                label_frames (np.ndarray): Label frames for the audio segment
                label_onsets (np.ndarray): Label onsets for the audio segment
                label_offsets (np.ndarray): Label offsets for the audio segment
                label_velocities (np.ndarray): Label velocities for the audio segment
        """
        num_frames = int(duration * self.pr_rate)
        start = idx * hop_size
        end = start + duration
        label_frames = np.zeros((num_frames, 128))
        label_onsets = np.zeros((num_frames, 128))
        label_offsets = np.zeros((num_frames, 128))
        label_velocities = np.zeros((num_frames, 128))
        label_roll = np.zeros((num_frames, 128))

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue
                    pitch = note.pitch
                    start_frame = max(0, int(self.pr_rate * (note.start-start)))
                    end_frame = min(num_frames, int(self.pr_rate * (note.end - start)))

                    # label roll
                    label_roll[start_frame:end_frame, pitch] = 1

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
        return label_frames, label_onsets, label_offsets, label_velocities, label_roll
    
    def trunc_feature(self, feature: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
            Truncate the feature to the length of the label
            Args:
                feature (np.ndarray): Feature to truncate
                label (np.ndarray): Label to truncate to

            Returns:
                feature (np.ndarray): Truncated feature
        """       
        if feature.shape[0] < label.shape[0]:
            # raise error to show that the num of frames of 
            # the feature is less than the num of frames
            # of the label
            raise RuntimeError(f"Feature shape {feature.shape} \
                    is less than label shape {label.shape}!")
        elif feature.shape[0] > label.shape[0]:
            # truncate the feature
            feature = feature[:label.shape[0], :]
        
        return feature

    
    def get_feature(self, audio : np.ndarray, feature: str):
        """
            Get the feature for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
                feature (str): Feature_type to extract
                feature_name (str): Name of the feature to extract
                feature_params (dict): Parameters for the feature extraction

            Returns:
                feature (np.ndarray): Feature for the audio segment
        """
        supported_features = ["mel", "cqt"]
        handcrafted_features = ["mel", "cqt"]

        # Check if the feature is supported
        if feature not in supported_features:
            raise ValueError(f"Feature {feature} not supported! \
                    Supported features are: {supported_features}")
        
        # Create the HandcraftedFeatures object
        if feature in handcrafted_features:
            hf = HandcraftedFeatures(sample_rate=self.sample_rate, \
                    window_size=self.window_size, pr_rate=self.pr_rate)
            
            if feature == "mel":
                n_mels = self.feature_params["n_mels"]
                n_fft = self.feature_params["mel_n_fft"]
                return hf.compute_mel(audio, \
                                    n_mels=n_mels, n_fft=n_fft)
            elif feature == "cqt":
                bins_per_octave = self.feature_params["cqt_bins_oct"]
                num_octaves = self.feature_params["cqt_num_octaves"]
                return hf.compute_cqt(audio, bins_per_octave=bins_per_octave, \
                        num_octaves=num_octaves)

    def checker(self, audio_files: list[Path], midi_files: list[Path]) -> None:
        """
            Check if the audio and midi files are the same.
            This function should be used only on datasets
            with audio and midi files having the same name.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"
        
        # Generate a random number from 0 to len(audio_files)
        idx = np.random.randint(0, len(audio_files))
        assert audio_files[idx].stem == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"
    
    def checker_guitarset_slakh(self, audio_files: list[Path], midi_files: list[Path], \
                                dataset: str="guitarset") -> None:
        """
            Check if the audio and midi files are the same
            for the GuitarSet dataset (assuming the audio is
            from the audio_mono-mic folder) or for Slakh.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"
        
        # Generate a random number from 0 to len(audio_files)
        idx = np.random.randint(0, len(audio_files))
        if dataset == "guitarset":
            assert audio_files[idx].stem[:-4] == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"
        else:
            # We assume its Slakh
            audio_track_name = audio_files[idx].parent.parent.stem
            midi_track_name = midi_files[idx].parent.parent.stem

            # Check the track names
            assert audio_track_name == midi_track_name, \
                f"Audio Track name: {audio_track_name} not the same as midi Track name: {midi_track_name}"
            
            # Check the file names
            assert audio_files[idx].stem == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"

    def get_files_maestro(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the MAESTRO dataset.

            Args:
                path (str): Path to the MAESTRO dataset
            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        audio_files = sorted(Path(path).rglob(f"*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*.{self.ext_midi}"))

        # Since MAESTRO's audio and midi files have the same name,
        # we can use checker
        self.checker(audio_files, midi_files)
        return audio_files, midi_files

    def get_files_guitarset(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the GuitarSet dataset.

            Args:
                path (str): Path to the GuitarSet dataset
            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        # check if the annotations-midi folder exists
        if not (Path(path)/"annotations-midi").exists():
            # Get all the jams in this path
            path_annot = Path(path)/"annotation"
            all_jams = sorted(path_annot.glob("*.jams")) 

            for _, jamPath in tqdm(enumerate(all_jams), total=len(all_jams)):
                jam_path = str(jamPath)
                jam = jams.load(jam_path)
                midi = jams_to_midi(jam, q=1)
                save_path = path_annot.parent / f"annotations-midi/{Path(jam_path).stem}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                midi.write(str(save_path) + f".{self.ext_midi}")
        
        # Get the list of audio and midi 
        audio_files = sorted((Path(path)/"audio_mono-mic").rglob(f"*.{self.ext_audio}"))
        midi_files = sorted((Path(path)/"annotations-midi").rglob(f"*.{self.ext_midi}"))

        # Use the GuitarSet checker to check if the audio and midi files are okay
        self.checker_guitarset_slakh(audio_files, midi_files)

        return audio_files, midi_files
    
    def get_files_musicnet(self, path: str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the MusicNet dataset. This function assumes that you are using 
        the MusicNet dataset alongside the musicnet_em labels. Also, note 
        that this function assumes that the musicnet_em labels are in a 
        folder called musicnet_em which should be in the same directory 
        with the test_data/, train_data/, etc. folders.

        Args:
            path (str): Path to the MusicNet dataset
        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        # Get the list of audio and midi files
        audio_files = []
        midi_files = sorted((Path(path)/"musicnet_em/").glob(f"*.{self.ext_midi}"))

        for i in range(len(midi_files)):
            midi_files[i] = Path(midi_files[i])
            temp = sorted(Path(path).rglob(f"{midi_files[i].stem}.{self.ext_audio}"))
            audio_files.append(temp[0])

        # Since MusicNet's audio and midi files have the same name,
        # we can use checker
        self.checker(audio_files, midi_files)

        return audio_files, midi_files

    def get_files_slakh(self, path: str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the Slakh dataset.

        Args:
            path (str): Path to the Slakh dataset
        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        unwanted = ["Drums", "Percussive", "Sound Effects", "Sound effects", \
                    "Chromatic Percussion"]
        audio_files = []
        midi_files = []

        for each in ['train', 'validation', 'test']:
            base_path = Path(path)/f"{each}/"
            tracks = [folder for folder in base_path.iterdir() if folder.is_dir()]
            for track in tqdm(tracks):
                try:
                    metadata = track / "metadata.yaml"
                    with open(metadata, "r") as f:
                        yaml_data = yaml.safe_load(f)
        
                    for key, value in yaml_data["stems"].items():
                        if value["inst_class"] not in unwanted:
                            audio_file = track / "stems" / f"{key}.{self.ext_audio}"
                            midi_file = track / "MIDI" / f"{key}.{self.ext_midi}"

                            try:
                                assert audio_file.exists(), f"{audio_file} does not exist"
                                assert midi_file.exists(), f"{midi_file} does not exist"
                            except AssertionError as e:
                                continue
                            audio_files.append(audio_file)
                            midi_files.append(midi_file)
                except:
                    print(f"Error in {track}")
                    continue
        
        # Check if the audio and midi files are the same
        # for the Slakh dataset
        self.checker_guitarset_slakh(audio_files, midi_files, dataset="slakh")

        return audio_files, midi_files


if __name__ == '__main__':
    args = argbind.parse_args()
    with argbind.scope(args):
        SnapExtractor()