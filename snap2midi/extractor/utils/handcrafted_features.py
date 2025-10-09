# Imports
import numpy as np
import librosa
from typing import List, Optional, Tuple

# Define a list of supported features
SUPPORTED_FEATURES = ["mel", "cqt"]

# Create class for handcrafted features
class HandcraftedFeatures:
    """
        HandcraftedFeatures is a class that implements DSP-like
        features for audio processing. These features are not learned
        by a neural network but are computed using traditional
        signal processing techniques.

        Modes supported:
            - mel: Mel spectrogram
            - cqt: Constant-Q Transform
    """
    def __init__(self, \
            sample_rate: int = 22050, \
            window_size: float = 20.0, \
            frame_rate: int = 4) -> None:
        """
            Default constructor for HandcraftedFeatures

            Args:
                sample_rate (int): Sample rate of audio
                window_size (float): Window size of audio if you don't want to calculate the 
                                     hop_length
                frame_rate (int): Frame rate of audio

            Returns:
                None
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.frame_rate = frame_rate

    def compute_cqt(self, audio: np.ndarray, bins_per_octave: int=24,\
                     num_octaves: int=6, hop_length: int | None = None) -> np.ndarray:
        """
            Compute the CQT for a given audio segment.
            This returns the log of the squared magnitude of the CQT.

            Args:
                audio (np.ndarray): Audio segment
                bins_per_octave (int): Number of bins per octave
                num_octaves (int): Number of octaves
                hop_length (int | None): Hop length for CQT computation

            Returns:
                np.ndarray: CQT of the audio segment
        """
        # compute hop length to get CQT
        if hop_length is None:
            numerator = self.window_size * self.sample_rate
            denominator = (self.frame_rate * self.window_size) - 1
            hop_length = int(numerator / denominator)

        cqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=int(bins_per_octave*num_octaves), \
                          bins_per_octave=bins_per_octave, hop_length=hop_length)

        # convert to squared magnitude
        cqt = np.abs(cqt)**2 

        # find the log of the cqts
        cqt = np.log1p(cqt)
        return cqt

    def compute_mel(self, audio: np.ndarray, \
                    n_mels: int=229, n_fft: int=2048, hop_length: int | None = None,
                    power_db: bool=False) -> np.ndarray:
        """
            Compute the Mel spectrogram for a given audio segment.
            This returns the log of the squared magnitude of the Mel spectrogram.

            Args:
                audio (np.ndarray): Audio segment
                n_mels (int): Number of mel bands
                n_fft (int): FFT size
                hop_length (int | None): Hop length for Mel spectrogram computation

            Returns:
                np.ndarray: Mel spectrogram of the audio segment
        """
        # This assumes padding on both ends
        # (without padding, size of window needs to be subtracted)
        if hop_length is None:
            numerator = (self.window_size * self.sample_rate) 
            denominator = (self.frame_rate * self.window_size) - 1
            hop_length = int(numerator / denominator)
        mel = librosa.feature.melspectrogram(y=audio, hop_length=hop_length, 
                                             sr=self.sample_rate, 
                                             n_mels=n_mels, n_fft=n_fft)
        
        if power_db:
            # convert to dB scale
            mel = librosa.power_to_db(mel, ref=np.max)
            return mel
        
        # convert to squared magnitude
        mel = np.abs(mel)**2

        # find the log of the mel spectrogram and clamp
        # to avoid log(0)
        mel = np.clip(mel, a_min=1e-10, a_max=None)
        mel = np.log(mel)
        return mel
