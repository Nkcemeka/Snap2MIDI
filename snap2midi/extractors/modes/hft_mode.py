"""
    Author: Chukwuemeka Livinus Nkama
    Date: 7/12/2025
    Description: HFT mode extractor for Snap2MIDI.
"""
from .base_mode import *
import torch
import torchaudio

class HFTMode(BaseMode):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.hft(self.config)
    
    def hft(self, config: dict):
        """
            Extract the features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.

            Args:
                config (dict): Configuration dictionary containing the parameters   
            Returns:
                None
        """
        self.extract_hft(config)
        self.collate_hft(config)
    
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
