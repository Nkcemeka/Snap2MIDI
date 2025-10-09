from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
import h5py

class _KongMode(_BaseMode):
    """
        KongMode is a class that extracts audio and MIDI files from the dataset
        for training the Kong model.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self._kong(config)

    def _kong(self, config):
        # Check if the path is None
        if self.path is None or not Path(self.path).exists():
            raise ValueError(f"Path {self.path} does not exist!")
        
        if config["dataset_name"] not in ["maps", "maestro"]:
            raise ValueError(f"Dataset {config['dataset_name']} not supported for Kong mode!")
        
        if config["frame_rate"] is None:
            raise ValueError("frame_rate must be set!")
        
        if config["sample_rate"] is None:
            raise ValueError("sample_rate must be set!")
        
        self.sample_rate = config["sample_rate"]
        self.hop_size = config["hop_size"]
        self.window_size = config["window_size"]
        self.frame_rate = config["frame_rate"]
        self.extraction_config = config
        
        # Perform the extraction
        self._extract()

    def _extract(self):
        """
            Extract audio and MIDI files from the dataset.
            We use a 10 second split for the audio files.
        """

        # Implement the extraction logic here
        if self.dataset_name == "maps":
            train_files, val_files, test_files = self._get_maps_train_val_test()
        elif self.dataset_name == "maestro":
            train_files, val_files, test_files = self._get_maestro_train_val_test()
        
        splits = ["train", "val", "test"]
        splits_data = [train_files, val_files, test_files]
        print(f"Total train files found: {len(train_files)}, total val files found: {\
            len(val_files)}, total test files found: {len(test_files)}")
        
        # create the directory for the splits
        for split in splits:
            split_path = Path(f"./{self.save_name}/{split}")
            split_path.mkdir(parents=True, exist_ok=True)

        for idx, split in enumerate(splits):
            split_files = splits_data[idx]

            for audio_file, midi_file in tqdm(split_files, \
                        desc=f"Processing audio and MIDI files for {split} split..."):
                assert audio_file.exists(), f"{audio_file} does not exist"
                assert midi_file.exists(), f"{midi_file} does not exist"

                if self.dataset_name != "slakh":
                    store_path = f"./{self.save_name}/{split}/{str(audio_file.stem)}.h5" 
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"./{self.save_name}/{split}/{track_name}_{str(audio_file.stem)}.h5"
                
                # Load the audio file
                audio = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)[0]
                if split == "train" or split == "val":
                    hop_samples = int(self.hop_size * self.sample_rate)
                    window_samples = int(self.window_size * self.sample_rate)

                    # pad audio so that it fits when we take segments of window_size
                    # at a hop_size interval
                    if (audio.shape[-1] - window_samples) % hop_samples != 0:
                        padding = hop_samples - ((audio.shape[-1] - window_samples) % hop_samples)
                        audio = np.pad(audio, (0, padding))

                    # how to get the start indices for each segment
                    # start_list = np.arange(0, audio.shape[-1] - window_samples + 1, hop_samples)
                    # convert audio to int16
                    assert np.max(np.abs(audio)) <= 1.
                    audio =  (audio * 32767.).astype(np.int16)

                    with h5py.File(store_path, 'w') as hf:
                        hf.attrs.create('midi_path', str(midi_file))
                        hf.attrs.create('hop_samples', hop_samples)
                        hf.attrs.create('window_samples', window_samples)
                        hf.attrs.create('sample_rate', self.sample_rate)
                        hf.create_dataset('audio', data=audio, dtype=np.int16)
                else:
                    # For the test set, we extract the whole file
                    # convert audio to int16
                    assert np.max(np.abs(audio)) <= 1.
                    audio =  (audio * 32767.).astype(np.int16)

                    with h5py.File(store_path, 'w') as hf:
                        hf.attrs.create('midi_path', str(midi_file))
                        hf.attrs.create('sample_rate', self.sample_rate)
                        hf.create_dataset('audio', data=audio, dtype=np.int16)

        # Store the extraction config
        config_path = Path(f"./{self.save_name}/extraction_config.h5")
        with h5py.File(config_path, 'w') as hf:
            for key, value in self.extraction_config.items():
                if key == "feature_params":
                    for fkey, fvalue in value.items():
                        hf.attrs.create(fkey, fvalue)
                else:
                    if isinstance(value, str):
                        hf.attrs.create(key, value, dtype='S100')
                    else:
                        hf.attrs.create(key, value)

        print(f"Extraction complete! Total files extracted: {len(train_files) + len(val_files) + len(test_files)}")