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
import argparse
import jams
import json
import random
from tqdm import tqdm
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.extractors.utils.conv_jams_midi import jams_to_midi
from typing import Optional
import torchaudio
import torch


class SnapExtractor:
    """
        Class to extract audio segments, features and labels 
         from any given dataset. This assumes that we have
        two lists of files, one for audio and the other for MIDI.
    """

    def __init__(self, config: dict, mode: str="general") -> None:
        """
            Args:
                config (dict): Configuration dictionary containing the parameters
                mode (str): Mode of extraction. Default is "general".

            Returns:
                None
        """
        # init the main core attributes
        self.dataset_name = config["dataset_name"]
        self.ext_audio = config["ext_audio"]
        self.ext_midi = config["ext_midi"]
        self.path = config["path"]
        self.save_name = config["save_name"]

        # Extract datasets
        if self.dataset_name == "maestro":
            self.data = self.get_files_maestro(self.path)
        elif self.dataset_name == "guitarset":
            self.data = self.get_files_guitarset(self.path)
        elif self.dataset_name == "musicnet":
            self.data = self.get_files_musicnet(self.path)
        elif self.dataset_name == "slakh":
            self.data = self.get_files_slakh(self.path)
        elif self.dataset_name == "maps":
            self.data = self.get_files_maps(self.path)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Create the save directory if it does not exist
        Path(self.save_name).mkdir(parents=True, exist_ok=True)

        if self.dataset_name == "maps":
            # To prevent tick overflow in pretty_midi,
            # for the MAPS dataset
            pretty_midi.pretty_midi.MAX_TICK = 1e10

        if mode == "general":
            self.general(config)
        elif mode == "hft":
            self.extract_hft(config)
            self.collate_hft(config)
        else:
            raise ValueError(f"Mode {mode} not supported! \
                    Supported modes are: general, hft")
        
    def general(self, config: dict):
        # Check if the path is None
        if self.path is None:
            raise ValueError(f"Path is None! Cannot Instantiate {self.__class__.__name__}!")
        
        # Check if the path exists
        if not Path(self.path).exists():
            raise ValueError(f"Path {self.path} does not exist!")
        
        # Init defaults
        self.hop_size = config["hop_size"]
        self.window_size = config["window_size"]
        self.duration = config["duration"]
        self.sample_rate = config["sample_rate"] if config["sample_rate"] is not None else 22050
        self.feature = config["feature"]
        self.feature_params = config["feature_params"]
        self.train_split = config["train_split"]
        self.val_split = config["val_split"]
        self.test_split = config["test_split"]
        self.pr_rate = config["pr_rate"]
        self.feature = config["feature"]
        self.feature_params = config["feature_params"]

        # Check if the splits are valid
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError(f"Splits do not add up to 1.0! \
                    Train split: {self.train_split}, Val split: {self.val_split}, \
                    Test split: {self.test_split}")

        # Set pr_rate
        if self.pr_rate is None:
            self.pr_rate = 32
        
        # Set feature params.
        if not self.feature_params:
            self.feature = "mel"
            self.feature_params = {"n_mels": 229, "mel_n_fft": 2048}

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

                store_dict = {'audio': None, 'feature': None}  

                label_dict = self.get_label_roll(midi, self.window_size, \
                    self.hop_size, idx)

                feature = self.get_feature(chunk, feature=self.feature)
                feature = feature.T # (time, embedding)

                feature = self.trunc_feature(feature, label_dict['label_frames'])

                if self.dataset_name != "slakh":
                    store_path = f"./{self.save_name}/{str(audio_file.stem)}_{idx}.npz" 
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"./{self.save_name}/{track_name}_{str(audio_file.stem)}_{idx}.npz"

                store_dict['audio'] = chunk
                store_dict['feature'] = feature

                for key in label_dict.keys():
                    store_dict[key] = label_dict[key]
                    
                np.savez(store_path, **store_dict)
    
        print(f"Extraction complete! Total duration: {total_duration} seconds")
    
    def collate_hft(self, config: dict):
        """
            Collate the features and labels for the hFT-Transformer model by Sony.
            This method processes the features and labels extracted from the audio and MIDI files
            and stores them in a single npz file for each split (train, val, test).

            Args:
                config (dict): Configuration dictionary containing the parameters
            Returns:
                None
        """
        # Process the train, val and test splits
        self.collate_hft_split(config, "train")
        self.collate_hft_split(config, "val")
        self.collate_hft_split(config, "test")
    
    def collate_hft_split(self, config: dict, split: str):
        num_frame_list = [] # stores the number of frames for each file

        # the total number of frames read including the margins
        # we are trying to get a window of Nctx from L frames from the actual feature
        # with a hop size of 1. Total number of frames becomes L - Nctx + 1. Since this does
        # not add up to L, we add Nctx - 1 to the original and so we will process get L + Nctx - 1.
        # We also add the margin at the beginning and end of the feature
        # see explanation below. total_num_frame computes this total including the margins
        # The reason for this is explained due to the __getitem__ method we will make in the 
        # dataset class.
        total_num_frame = config["input"]["margin_b"]

        # the total or actual number of frames read from the features
        total_num_frame_idx = 0 

        # load the list of the filenames in the feature directory
        files_all = sorted(Path(f"{self.save_name}/feature/{split}").rglob("*.npz"))

        for i, each in enumerate(files_all):
            # load the npz file
            npz_file = np.load(each)
            num_frame_feature = npz_file['feature'].shape[0] # number of frames for the feature
            num_frame_label = len(npz_file['label_frames']) # number of frames for MPE
            del npz_file # delete npz file to free memory

            # Get the number of frames based on the maximum for the feature and label
            num_frame = max(num_frame_feature, num_frame_label)
            num_frame_list.append(num_frame)

            # So, here we go: we have num_frame (L) for the number of features
            # But we want config['input']['num_frame'] (Nctx) instead. 
            # the expec. no of frames = (L - Nctx)/1 + 1
            # which is L - Nctx + 1. To get exactly L. no of frames, we add Nctx - 1
            # This gives us L + Nctx - 1; now we need some backward and forward margin
            # This leads to L + Nctx - 1 + Mf. Note that Mf for file i is Mb for file
            # i - 1;
            # Hence, Mb (start pad)| L1 + Nctx - 1 | Mf | L2 + Nctx - 1 | Mf etc.... 

            # We are interested in getting L frames because our idx in the __getitem__ can give
            # us the starting point of our window as any frame of L frames...
            total_num_frame += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
            total_num_frame_idx += num_frame
        
        # dataset_idx (helps us keep track of the location of the actual features)
        print(f"Processing dataset_idx for {split}...")
        dataset_idx = np.zeros(total_num_frame_idx, dtype=np.int32)
        loc_i = 0
        loc_d = config['input']['margin_b'] 
        for i, each in enumerate(files_all):
            # Tells us where our segemnt L ended up in.
            # so for each raw frame in the unpadded, where did it
            # end in the padded...
            num_frame = num_frame_list[i]
            dataset_idx[loc_i:loc_i + num_frame] = np.arange(loc_d, loc_d + num_frame)
            loc_i += num_frame
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
        
        # store dataset_idx in the save_name directory as a npz file
        np.savez(f"{self.save_name}/idx/{split}/dataset_idx.npz", dataset_idx=dataset_idx)
        del dataset_idx # delete to free memory

        ## Process the features
        print(f"Processing features for {split}...")
        if config['feature']['log_offset'] > 0.0:
            zero_value = np.log(config['feature']['log_offset'])
        else:
            zero_value = config['feature']['log_offset']
        
        dataset_feature = np.full((total_num_frame, config['feature']['mel_bins']), zero_value, dtype=np.float32)
        loc_d = config['input']['margin_b'] 

        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the features at the right location
            npz_file = np.load(each)
            num_frame_feature = npz_file['feature'].shape[0] # number of frames for the feature
            dataset_feature[loc_d:loc_d + num_frame_feature, :] = npz_file['feature']
            loc_d += num_frame_feature + config['input']['margin_f'] + config['input']['num_frame'] - 1
            del npz_file # delete npz file to free memory

        # store the dataset_feature in the save_name directory as a npz file
        np.savez(f"{self.save_name}/feature/{split}/dataset_feature.npz", dataset_feature=dataset_feature)
        del dataset_feature # delete to free memory

        ## Process the labels
        print(f"Processing labels for {split}...")
        dataset_label_frames = np.zeros((total_num_frame, config['midi']['num_note']), dtype=np.bool)
        loc_d = config['input']['margin_b']

        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the labels at the right location
            npz_file = np.load(each)
            num_frame_label = len(npz_file['label_frames'])
            dataset_label_frames[loc_d:loc_d + num_frame_label, :] = npz_file['label_frames'][:]
            del npz_file # delete npz file to free memory
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
        
        # store the dataset_label_frames in the save_name directory as a npz file
        np.savez(f"{self.save_name}/label_frames/{split}/dataset_label_frames.npz", dataset_label_frames=dataset_label_frames)
        del dataset_label_frames # delete to free memory

        ## process the onsets
        print(f"Processing onsets for {split}...")
        dataset_label_onset = np.zeros((total_num_frame, config['midi']['num_note']), dtype=np.float32)
        loc_d = config['input']['margin_b']
        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the labels at the right location
            npz_file = np.load(each)
            num_frame_label = len(npz_file['label_onset'])
            dataset_label_onset[loc_d:loc_d + num_frame_label, :] = npz_file['label_onset'][:]
            del npz_file
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
        
        # store the dataset_label_onset in the save_name directory as a npz file
        np.savez(f"{self.save_name}/label_onset/{split}/dataset_label_onset.npz", dataset_label_onset=dataset_label_onset)
        del dataset_label_onset # delete to free memory

        ## process the offsets
        print(f"Processing offsets for {split}...")
        dataset_label_offset = np.zeros((total_num_frame, config['midi']['num_note']), dtype=np.float32)
        loc_d = config['input']['margin_b']
        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the labels at the right location
            npz_file = np.load(each)
            num_frame_label = len(npz_file['label_offset'])
            dataset_label_offset[loc_d:loc_d + num_frame_label, :] = npz_file['label_offset'][:]
            del npz_file
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
        
        # store the dataset_label_offset in the save_name directory as a npz file
        np.savez(f"{self.save_name}/label_offset/{split}/dataset_label_offset.npz", dataset_label_offset=dataset_label_offset)
        del dataset_label_offset # delete to free memory

        ## process the velocities
        print(f"Processing velocities for {split}...")
        dataset_label_velocity = np.zeros((total_num_frame, config['midi']['num_note']), dtype=np.int8)
        loc_d = config['input']['margin_b']
        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the labels at the right location
            npz_file = np.load(each)
            num_frame_label = len(npz_file['label_velocity'])
            dataset_label_velocity[loc_d:loc_d + num_frame_label, :] = npz_file['label_velocity'][:]
            del npz_file
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        # store the dataset_label_velocity in the save_name directory as a npz file
        np.savez(f"{self.save_name}/label_velocity/{split}/dataset_label_velocity.npz", dataset_label_velocity=dataset_label_velocity)
        del dataset_label_velocity # delete to free memory
    
    def extract_hft(self, config: dict):
        """
            Extract the features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.

            Args:
                config (dict): Configuration dictionary containing the parameters   
            Returns:
                None
        """
        split_files = self.get_splits_hft() # train, val, test
        split_str = ["train", "val", "test"]

        print(f"------ Dataset Split Statistics (hFT-Transformer) ------")
        print(f"Number of train files: {len(split_files[0])}")
        print(f"Number of val files: {len(split_files[1])}")
        print(f"Number of test files: {len(split_files[2])}")

        # make a feature directory if it does not exist
        Path(f"{self.save_name}/feature/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/feature/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/feature/test").mkdir(parents=True, exist_ok=True)

        # Also create paths for the labels, we will use them later on
        Path(f"{self.save_name}/label_frames/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_frames/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_frames/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_onset/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_onset/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_onset/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_offset/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_offset/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_offset/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_velocity/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_velocity/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_velocity/test").mkdir(parents=True, exist_ok=True)

        # make a directory for the idx
        Path(f"{self.save_name}/idx/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/idx/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/idx/test").mkdir(parents=True, exist_ok=True)
            

        print(f"Extracting features and labels for the hFT-Transformer model...")
        for i, split in tqdm(enumerate(split_files)):
            split_name = split_str[i]
            for (audio_file, midi_file) in tqdm(split, total=len(split), desc="Extracting files"):
                filename = audio_file.stem
                feature = self.get_feature_hft(str(audio_file), config)
                label = self.get_label_hft(str(midi_file), config)
                feature_dict = {'feature': feature}
                feature_dict.update(label)
                
                # Save the feature and label to a npz file
                if self.dataset_name != "slakh":
                    store_path = f"{self.save_name}/feature/{split_name}/{filename}.npz"
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"{self.save_name}/feature/{split_name}/{track_name}_{filename}.npz"

                np.savez(store_path, **feature_dict)
    
    def get_feature_hft(self, audio_file: str, config: dict) -> torch.Tensor:
        """
            Get the feature for the audio file for the hFT-Transformer model by Sony.

            Args:
                audio_file (str): Path to the audio file
                config (dict): Configuration dictionary containing the parameters
            Returns:
                feature (torch.Tensor): Feature for the audio file
        """
        # Get the feature for the audio file
        # we use torchaudio to speed this up; librosa is too slow
        audio, sr = torchaudio.load(audio_file)
        audio = torch.mean(audio, dim=0)
        resample = torchaudio.transforms.Resample(sr, config["feature"]["sr"])
        audio = resample(audio)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["feature"]["sr"],
            n_fft=config["feature"]["fft_bins"],
            hop_length=config["feature"]["hop_sample"],
            win_length=config["feature"]["window_length"],
            n_mels=config["feature"]["mel_bins"],
            pad_mode=config["feature"]["pad_mode"],
            norm="slaney"
        )
        feature = mel_transform(audio)
        feature = (torch.log(feature + config['feature']['log_offset'])).T
        return feature

    def get_label_hft(self, midi_file: str, config: dict) -> dict:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        instrument = midi_data.instruments[0]
        notes = []
        max_offset = 0

        # Extract the notes from the MIDI file
        for note in instrument.notes:
            pitch = note.pitch
            start = note.start
            end = note.end  
            velocity = note.velocity
            notes.append({
                'pitch': pitch,
                'onset': start,
                'offset': end,
                'velocity': velocity
            })

            if max_offset < end:
                max_offset = end

        # Sort notes by onset and pitch
        notes = sorted(sorted(notes, key=lambda x: x['pitch']), key=lambda x: x['onset'])

        # The hop is the distance between two consecutive frames
        hop_ms = 1000*config["feature"]["hop_sample"] / config["feature"]["sr"]

        # We use a 50ms window as a tolerance as to how many frames to the left or right
        # of the true onset we should consider as being close enough
        # we convert the tolerance to frames below
        onset_tolerance = int(50.0 / hop_ms + 0.5)
        offset_tolerance = int(50.0 / hop_ms + 0.5)

        # number of frames per second
        nframe_in_sec = config['feature']['sr'] / config['feature']['hop_sample']

        nframe = int(max_offset * nframe_in_sec + 0.5) + 1
        label_frames = np.zeros((nframe, config['midi']['num_note']), dtype=np.bool)
        label_onset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
        label_offset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
        label_velocity = np.zeros((nframe, config['midi']['num_note']), dtype=np.int8)

        for i in range(len(notes)):
            pitch = notes[i]['pitch'] - config['midi']['note_min']

            # Get onset time in frames
            onset_frame = int(notes[i]['onset'] * nframe_in_sec + 0.5)
            onset_ms = notes[i]['onset']*1000.0 # onset time in ms
            onset_sharpness = onset_tolerance

            # offset time in frames
            offset_frame = int(notes[i]['offset'] * nframe_in_sec + 0.5)
            offset_ms = notes[i]['offset']*1000.0 # offset time in ms
            offset_sharpness = offset_tolerance

            # velocity
            velocity = notes[i]['velocity']

            # onset
            for j in range(0, onset_sharpness+1):
                # Create a traingular soft label centred at the note's actual onset time
                # This is the idea for the Kong Model
                onset_ms_q = (onset_frame + j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame+j < nframe:
                    # There may be multiple notes whose tolerance windows overlap.
                    # We always keep the highest label value at each (frame, pitch).
                    label_onset[onset_frame+j][pitch] = max(label_onset[onset_frame+j][pitch], onset_val)
                    if (label_onset[onset_frame+j][pitch] >= 0.5):
                        #  If this frame is “close enough,” record velocity
                        # We only record the velocity if the corresponding onset value is greater than 0.5
                        # By doing this, we ensure the velocity is most-dependent on the onset responsible
                        label_velocity[onset_frame+j][pitch] = velocity
                
            for j in range(1, onset_sharpness+1):
                onset_ms_q = (onset_frame - j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame-j >= 0:
                    label_onset[onset_frame-j][pitch] = max(label_onset[onset_frame-j][pitch], onset_val)
                    if (label_onset[onset_frame-j][pitch] >= 0.5) and (label_velocity[onset_frame-j][pitch] == 0):
                        # Think about it, if the velocity is already set, we don't need to set it again
                        # This is because we have found a frame above closest to it.
                        # Drawing this out on paper might help
                        label_velocity[onset_frame-j][pitch] = velocity
            
            # mpe or frames label
            for j in range(onset_frame, offset_frame+1):
                label_frames[j][pitch] = 1
            
            # offset
            offset_flag = True
            for j in range(len(notes)):
                if notes[i]['pitch'] != notes[j]['pitch']:
                    continue
                if notes[i]['offset'] == notes[j]['onset']:
                    offset_flag = False
                    break
            
            if offset_flag is True:
                for j in range(0, offset_sharpness+1):
                    offset_ms_q = (offset_frame + j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame+j < nframe:
                        label_offset[offset_frame+j][pitch] = max(label_offset[offset_frame+j][pitch], offset_val)

                for j in range(1, offset_sharpness+1):
                    offset_ms_q = (offset_frame - j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame-j >= 0:
                        label_offset[offset_frame-j][pitch] = max(label_offset[offset_frame-j][pitch],  offset_val)
        
        label = {
            'label_frames': label_frames.tolist(),
            'label_onset': label_onset.tolist(),
            'label_offset': label_offset.tolist(),
            'label_velocity': label_velocity.tolist()
        }
        return label

    def get_splits_hft(self):
        """
            Extract the audio segments, features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.
        """
        train_files = []
        val_files = []
        test_files = []

        if self.dataset_name == "maps":
            # collect all tunes for test first from the ENSTDkAm and ENSTDkCl
            # After that, collect the rest of the tunes for train and val
            tunes = []
            for i, each in enumerate(self.data[0]):
                tmp = str(each).replace(f"{self.path}", "").replace(\
                    f".{self.ext_audio}", "").rstrip('\n').split('/')
                code = tmp[1] # folder name
                content = tmp[2] # category name (MUS, ISOL, etc.)
                tune = tmp[-1].rstrip(code).lstrip('MAPS_'+content+'-') # tune name

                if (code == 'ENSTDkAm' or code == 'ENSTDkCl'):
                    # append tune name to the tunes list
                    test_files.append((each, self.data[1][i]))
                    if tune not in tunes:
                        tunes.append(tune)
            
            for i, each in enumerate(self.data[0]):
                tmp = str(each).replace(f"{self.path}", "").replace(\
                    f".{self.ext_audio}", "").rstrip('\n').split('/')
                code = tmp[1] # folder name
                content = tmp[2] # category name (MUS, ISOL, etc.)
                tune = tmp[-1].rstrip(code).lstrip('MAPS_'+content+'-') # tune name

                if (code != 'ENSTDkAm' and code != 'ENSTDkCl'):
                    if tune not in tunes:
                        train_files.append((each, self.data[1][i]))
                    else:
                        val_files.append((each, self.data[1][i]))
        
        return train_files, val_files, test_files

    def get_label_roll(self, midi: pretty_midi.PrettyMIDI, \
                  duration: int | float, hop_size: float, idx: int) -> dict:
        """
            Get the label and pedal rolls for a given audio segment.
            The regressed rolls generated follow Kong's model!

            Args:
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                duration (float): Duration of the audio segment
                hop_size (float): Hop size in seconds
                idx (int): Index of the audio segment

            Returns:
                target_dict: {
                    label_frames (np.ndarray): Frames label
                    label_onsets (np.ndarray): Onsets label
                    label_offsets (np.ndarray): Offsets label
                    label_reg_onsets (np.ndarray): Label regression onsets for the audio segment
                    label_reg_offsets (np.ndarray): Label regression offsets for the audio segment
                    label_velocities (np.ndarray): Label velocities for the audio segment
                    pedal_onset (np.ndarray): Pedal onset label
                    pedal_frames (np.ndarray): Pedal frames label
                    pedal_offset (np.ndarray): Pedal offset label
                    pedal_reg_onset (np.ndarray): Regressed pedal onset label
                    pedal_reg_offset (np.ndarray): Regressed pedal offset label
                    mask_roll (np.ndarray): Mask roll label to remove all events that do not occur
                                            in the audio segment
                    note_events (np.ndarray): Array of note events for the audio segment
                    pedal_events (np.ndarray): Array of pedal events for the audio segment
                } 
        """
        # init target_dict and set the keys to None
        target_dict: dict[str, Optional[np.ndarray]] = {
            'label_frames': None,
            'label_onsets': None,
            'label_offsets': None,
            'label_reg_onsets': None,
            'label_reg_offsets': None,
            'label_velocities': None,
            'pedal_onset': None,
            'pedal_frames': None,
            'pedal_offset': None,
            'pedal_reg_onset': None,
            'pedal_reg_offset': None,
            'mask_roll': None,
            'note_events': None
        }

        num_frames = int(round(duration * self.pr_rate)) + 1
        start = idx * hop_size
        end = start + duration

        # initialize the labels
        label_frames = np.zeros((num_frames, 128))
        label_onsets = np.zeros((num_frames, 128))
        label_offsets = np.zeros((num_frames, 128))
        label_velocities = np.zeros((num_frames, 128))
        label_reg_onsets = np.ones((num_frames, 128))
        label_reg_offsets = np.ones((num_frames, 128))
        pedal_onset = np.zeros((num_frames,))
        pedal_frames = np.zeros((num_frames,))
        pedal_offset = np.zeros((num_frames,))
        pedal_reg_onset = np.ones((num_frames,))
        pedal_reg_offset = np.ones((num_frames,))
        mask_roll = np.ones((num_frames, 128))

        note_events = []  # contains (onset, offset, pitch, velocity)
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 

                    pitch = note.pitch

                    # Need to deal with the case where the note starts before
                    # the start of the segment and ends after the end of the segment
                    # Not sure if what I have done below is the right thing to do
                    note_events.append([max(0, note.start - start), min(duration, note.end - start), 
                                        pitch, note.velocity])
                    
                    assert np.min(note_events) >= 0, "Note events contain negative values!"
                    
                    start_frame = int(round(self.pr_rate * (note.start - start)))

                    # clamp end frame like was done in Kong's paper
                    end_frame = int(round(self.pr_rate * (note.end - start)))

                    # prepare labels (note that end_frame is >= 0 else it would
                    # have been skipped)
                    if end_frame < num_frames:
                        label_frames[max(0, start_frame):end_frame + 1, pitch] = 1
                        label_offsets[end_frame, pitch] = 1
                        label_velocities[max(0, start_frame):end_frame + 1, pitch] = note.velocity

                        label_reg_offsets[end_frame, pitch] = \
                        (note.end - start) - (end_frame / self.pr_rate)

                        if start_frame >= 0:
                            label_onsets[start_frame, pitch] = 1
                            label_reg_onsets[start_frame, pitch] = \
                            (note.start - start) - (start_frame / self.pr_rate)
                        else:
                            # We will never get here
                            mask_roll[:end_frame + 1, pitch] = 0
                    else:
                        if start_frame >= 0:
                            # This section wasn't here before
                            # ------------------------------
                            label_frames[start_frame:, pitch] = 1
                            label_onsets[start_frame, pitch] = 1
                            label_velocities[start_frame:, pitch] = note.velocity
                            label_reg_onsets[start_frame, pitch] = \
                            (note.start - start) - (start_frame / self.pr_rate)
                            # ------------------------------
                            mask_roll[start_frame:, pitch] = 0
                        else:
                            # we won't get here too
                            mask_roll[:, pitch] = 0
        
        for pitch in range(128):
            label_reg_onsets[:, pitch] = self.get_reg(label_reg_onsets[:, pitch])
            label_reg_offsets[:, pitch] = self.get_reg(label_reg_offsets[:, pitch])
        
        # Get the pedal events
        # Credits: https://github.com/craffel/pretty-midi/blob/main/pretty_midi/instrument.py#L69
        CC_SUSTAIN_PEDAL = 64
        frame_pedal_on = 0
        is_pedal_on = False
        pedal_events: list[list] = [] # contains [onset_time, offset_time]

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for cc in [_e for _e in instrument.control_changes
                           if _e.number == CC_SUSTAIN_PEDAL]:
                    
                    if cc.time >= end or cc.time <= start:
                        continue
                   
                    # Kong's implementation used round before int & so I did...why?
                    # round minimizes the average timing error to half a frame...
                    # int is okay but leads to more errors on average..
                    frame_now = int(round((cc.time - start) * self.pr_rate))
                    time_now = cc.time - start
                    is_current_pedal_on = (cc.value >= CC_SUSTAIN_PEDAL)

                    if not is_pedal_on and is_current_pedal_on:
                        frame_pedal_on = frame_now
                        is_pedal_on = True
                        onset_time = time_now
                    elif is_pedal_on and not is_current_pedal_on:
                        # store the pedal information
                        # We add +1 due to python's indexing. Also, see num_frames above;
                        # + 1 was added to catch events that fall exactly at the right
                        # edge of the last frame (which can happen)
                        pedal_frames[frame_pedal_on:frame_now + 1] = 1
                        pedal_offset[frame_now] = 1
                        pedal_reg_offset[frame_now] = \
                        (time_now) - (frame_now / self.pr_rate)

                        if frame_pedal_on >= 0:
                            pedal_onset[frame_pedal_on] = 1
                            pedal_reg_onset[frame_pedal_on] = \
                            (onset_time) - (frame_pedal_on / self.pr_rate)
                        
                        is_pedal_on = False
                        pedal_events.append([onset_time, time_now])
        
        # Get the pedal regressed values
        pedal_reg_onset = self.get_reg(pedal_reg_onset)
        pedal_reg_offset = self.get_reg(pedal_reg_offset)

        # update the target_dict
        target_dict['label_frames'] = label_frames
        target_dict['label_onsets'] = label_onsets
        target_dict['label_offsets'] = label_offsets
        target_dict['label_reg_onsets'] = label_reg_onsets
        target_dict['label_reg_offsets'] = label_reg_offsets
        target_dict['label_velocities'] = label_velocities
        target_dict['pedal_onset'] = pedal_onset
        target_dict['pedal_frames'] = pedal_frames
        target_dict['pedal_offset'] = pedal_offset
        target_dict['pedal_reg_onset'] = pedal_reg_onset
        target_dict['pedal_reg_offset'] = pedal_reg_offset
        target_dict['mask_roll'] = mask_roll
        target_dict['note_events'] = np.array(note_events)
        target_dict['pedal_events'] = np.array(pedal_events)
        return target_dict
        
    
    def get_reg(self, input: np.ndarray) -> np.ndarray:
        """
            Get the regression for a given roll using
            Kong's approach! The code is slightly different
            from the original.

            Args:
                input (np.ndarray): Roll to regress
            Returns:
                output (np.ndarray): Regressed roll
        """
        step = 1. / self.pr_rate
        output = np.ones_like(input)
        
        # Get the locations where the events (onsets/offsets) occur
        # since the initial regression matrix is 1 everwhere and an
        # onset/offset can occur whithin a frame resolution of 1/self.pr_rate
        # then 0.5 is enough to determine the locations of the events
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    # should be input[locts[i + 1]]
                    output[t] = step * (t - locts[i + 1]) - input[locts[i + 1]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output

    
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
            raise RuntimeError(f"Feature frames {feature.shape} \
                    is less than label frames {label.shape}!")
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
    
    def get_files_maps(self, path:str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the MAPS dataset. Note that MAPS is organized into
        several categories: ISOL, RAND, UCHO and MUS.

        ISOL: Isolated notes and monophonic sounds
        RAND: Randomly chords
        UCHO: Usual chords
        MUS: Pieces of music

        This function extracts the audio and midi files
        for the MUS category only.

        Args:
            path (str): Path to the MAPS dataset

        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        # Get the list of audio and midi files
        audio_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_midi}"))
        self.checker(audio_files, midi_files)
        return audio_files, midi_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, \
                        help='Path to the config file')
    parser.add_argument('--mode', type=str, default='general', choices=['general', 'hft', '...'], \
                        help='Mode of operation.')
    args = parser.parse_args()
    mode = args.mode
    config = args.config_path

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()

    # parse JSON file
    config = json.loads(content)
    SnapExtractor(config, mode=mode)