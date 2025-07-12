"""
    Author: Chukwuemeka Livinus Nkama
    Date: 7/12/2025
    Description: General mode extractor for Snap2MIDI.
"""
from .base_mode import *
import random
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures

class GeneralMode(BaseMode):
    """
        General mode extractor for Snap2MIDI.
        This class inherits from BaseMode and implements methods to extract audio and MIDI files
        from various datasets such as MusicNet, Slakh, and MAPS.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.general(self.config)
    
    def general(self, config: dict):
        # Check if the path is None
        if self.path is None:
            raise ValueError(f"Path is None!")

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
                    self.hop_size, idx, self.config['pr_rate'])

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
