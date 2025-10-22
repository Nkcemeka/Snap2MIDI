"""
    Extracts data for training the datasets for each state of the art model.
    The extractor is currently configured to extract data for the datasets used in
    the original paper for each architecture. Future versions might make this more flexible!
"""
from .extractor.modes.hft_mode import _HFTMode
from .extractor.modes.oaf_mode import _OAFMode
from .extractor.modes.kong_mode import _KongMode

class SnapExtractor:
    """
        Class to extract audio segments, features and labels 
        for training.
    """

    def __init__(self) -> None:
        pass
        
    def _build_config_from_kwargs(self, **kwargs):
        """
            Build configuration dictionary from keyword arguments.

            Parameters:
            -----------
                **kwargs: 
                    Keyword arguments to build the configuration dictionary.

            Returns
            -------
                config (dict): 
                    Configuration dictionary.
        """
        config = {}
        for key, value in kwargs.items():
            config[key] = value
        return config

    def extract_hft(self, path: str, dataset_name: str="maps", margin_b: int = 32, margin_f: int = 32, 
            sample_rate: int = 16000, hop_sample: int = 256, num_frame: int = 128, note_min: int = 21,
            note_max: int = 108, num_velocity: int = 128, mel_bins: int = 256, n_bins: int = 256, 
            fft_bins: int = 2048, window_length: int = 2048, log_offset: float = 1e-8,
            window: str = "hann", pad_mode: str = "constant"):
        """
            Extract audio and MIDI files from the HFT dataset.
            
            Parameters:
            -----------
                path (str): 
                    Path to the MAPS dataset. Other datasets not supported yet.
                dataset_name (str): 
                    Name of the dataset. Default is "maps".
                margin_b (int): 
                    Backward margin in frames. Default is 32.
                margin_f (int): 
                    Forward margin in frames. Default is 32.
                sample_rate (int): 
                    Sample rate of the audio. Default is 16000.
                hop_sample (int): 
                    Hop size in samples. Default is 256.
                num_frame (int): 
                    Number of frames in each segment. Default is 128.
                note_min (int): 
                    Minimum MIDI note number. Default is 21.
                note_max (int): 
                    Maximum MIDI note number. Default is 108.
                num_velocity (int): 
                    Number of velocity levels. Default is 128.
                mel_bins (int): 
                    Number of mel bins. Default is 256.
                n_bins (int): 
                    Number of frequency bins. Default is 256.
                fft_bins (int): 
                    Number of FFT bins. Default is 2048.
                window_length (int): 
                    Window length in samples. Default is 2048.
                log_offset (float): 
                    Log offset for numerical stability. Default is 1e-8.
                window (str): 
                    Type of window to use. Default is "hann".
                pad_mode (str): 
                    Padding mode for the window. Default is "constant".

            Returns
            -------
                None
        """
        feature_config = self._build_config_from_kwargs(sr=sample_rate, hop_sample=hop_sample, mel_bins=mel_bins,
                n_bins=n_bins, fft_bins=fft_bins, window_length=window_length, log_offset=log_offset,
                window=window, pad_mode=pad_mode)
        input_config = self._build_config_from_kwargs(margin_b=margin_b, margin_f=margin_f, num_frame=num_frame)
        midi_config = self._build_config_from_kwargs(note_min=note_min, note_max=note_max, num_note=note_max-note_min+1,
                num_velocity=num_velocity)
        extra_config = self._build_config_from_kwargs(save_name="data/hft", path=path, dataset_name=dataset_name,
                        ext_audio="wav", ext_midi="mid")
        
        # merge all configs
        config = {**extra_config}
        config["feature"] = feature_config
        config["input"] = input_config
        config["midi"] = midi_config

        _HFTMode(config)

    def extract_oaf(self, path: str, dataset_name: str="maps", sample_rate: int = 16000, feature: str = "mel",
                min_frame_secs: float = 5.0, max_frame_secs: float = 20.0, min_pitch: int = 21, max_pitch: int = 108,
                onset_length: int = 32, offset_length: int = 32, frame_rate: float = 31.25,
                n_mels: int = 229, mel_n_fft: int = 2048, hop_length: int = 512):
        """
            Extract audio and MIDI files from the OAF dataset.

            Parameters
            -----------
                path (str): 
                    Path to the MAPS dataset. Other datasets not supported yet.
                dataset_name (str): 
                    Name of the dataset. Default is "maps".
                sample_rate (int): 
                    Sample rate of the audio. Default is 16000.
                feature (str): 
                    Type of feature to extract. Default is "mel".
                min_frame_secs (float): 
                    Minimum frame length in seconds. Default is 5.0.
                max_frame_secs (float): 
                    Maximum frame length in seconds. Default is 20.0.
                min_pitch (int): 
                    Minimum MIDI note number. Default is 21.
                max_pitch (int): 
                    Maximum MIDI note number. Default is 108.
                onset_length (int): 
                    Length of onset in frames. Default is 32.
                offset_length (int): 
                    Length of offset in frames. Default is 32.
                frame_rate (float): 
                    Frame rate of the feature. Default is 31.25.
                n_mels (int): 
                    Number of mel bins. Default is 229.
                mel_n_fft (int): 
                    Number of FFT bins for mel spectrogram. Default is 2048.
                hop_length (int): 
                    Hop length in samples. Default is 512.

            Returns
            -------
                None
        """
        feature_params = self._build_config_from_kwargs(n_mels=n_mels, mel_n_fft=mel_n_fft, hop_length=hop_length)
        extra_config = self._build_config_from_kwargs(dataset_name=dataset_name, ext_audio="wav", ext_midi="mid",
                path=path, sample_rate=sample_rate, feature=feature, min_frame_secs=min_frame_secs,
                max_frame_secs=max_frame_secs, min_pitch=min_pitch, max_pitch=max_pitch,
                onset_length=onset_length, offset_length=offset_length, frame_rate=frame_rate,
                save_name="data/oaf/")
        config = {**extra_config}
        config["feature_params"] = feature_params
        _OAFMode(config)

    def extract_kong(self, path: str, dataset_name: str="maps", sample_rate: int = 16000, window_size: float = 10.0,
            feature: str = "mel", hop_size: float = 1.0, min_pitch: int = 21, max_pitch: int = 108,
            frame_rate: int = 100, n_mels: int = 229, mel_n_fft: int = 2048, hop_length: int = 160,
            extend_pedal: bool = True):
        """
            Extract audio and MIDI files from the Kong dataset.

            Parameters
            -----------
                path (str): 
                    Path to the MAESTRO dataset. Other datasets not supported yet.
                dataset_name (str): 
                    Name of the dataset. Default is "maestro".
                sample_rate (int): 
                    Sample rate of the audio. Default is 16000.
                window_size (float): 
                    Window size in seconds. Default is 10.0.
                feature (str): 
                    Type of feature to extract. Default is "mel".
                hop_size (float): 
                    Hop size in seconds. Default is 1.0.
                min_pitch (int): 
                    Minimum MIDI note number. Default is 21.
                max_pitch (int): 
                    Maximum MIDI note number. Default is 108.
                frame_rate (int): 
                    Frame rate of the feature. Default is 100.
                n_mels (int): 
                    Number of mel bins. Default is 229.
                mel_n_fft (int): 
                    Number of FFT bins for mel spectrogram. Default is 2048.
                hop_length (int): 
                    Hop length in samples. Default is 160.
                extend_pedal (bool): 
                    Whether to extend pedal information. Default is True.
            
            Returns
            -------
                None
        """
        feature_params = self._build_config_from_kwargs(n_mels=n_mels, mel_n_fft=mel_n_fft,
                hop_length=hop_length)
        extra_config = self._build_config_from_kwargs(dataset_name=dataset_name, ext_audio="wav", ext_midi="midi",
                path=path, sample_rate=sample_rate, window_size=window_size, feature=feature,  hop_size=hop_size,
                min_pitch=min_pitch, max_pitch=max_pitch, frame_rate=frame_rate, extend_pedal=extend_pedal,
                save_name="data/kong/")
        config = {**extra_config}
        config["feature_params"] = feature_params
        _KongMode(config)
