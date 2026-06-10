# Imports
import numpy as np
import librosa
from nnAudio2.features.mel import MelSpectrogram
from nnAudio2.features.cqt import CQT
import torch

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

            Args
            ----
                sample_rate (int): Sample rate of audio
                window_size (float): Window size of audio if you don't want to calculate the 
                                     hop_length
                frame_rate (int): Frame rate of audio

            Returns
            --------
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

            Args
            -----
                audio (np.ndarray): Audio segment
                bins_per_octave (int): Number of bins per octave
                num_octaves (int): Number of octaves
                hop_length (int | None): Hop length for CQT computation

            Returns
            --------
                cqt (np.ndarray): CQT of the audio segment
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
                    n_mels: int=229, n_fft: int=2048, fmin: int=0, fmax: float | None=None,\
                    hop_length: int | None = None,
                    power_db: bool=False, htk:bool=False) -> np.ndarray:
        """
            Compute the Mel spectrogram for a given audio segment.
            This returns the log of the squared magnitude of the Mel spectrogram.

            Args
            -----
                audio (np.ndarray): Audio segment
                n_mels (int): Number of mel bands
                n_fft (int): FFT size
                hop_length (int | None): Hop length for Mel spectrogram computation
                power_db (bool): Convert to db scale
                htk (bool): Default is False. Uses htk instead of slaney

            Returns
            --------
                mel (np.ndarray): Mel spectrogram of the audio segment
        """
        # This assumes padding on both ends
        # (without padding, size of window needs to be subtracted)
        if hop_length is None:
            numerator = (self.window_size * self.sample_rate) 
            denominator = (self.frame_rate * self.window_size) - 1
            hop_length = int(numerator / denominator)
        mel = librosa.feature.melspectrogram(y=audio, hop_length=hop_length, 
                                             sr=self.sample_rate, 
                                             n_mels=n_mels, n_fft=n_fft, 
                                             fmin=fmin, fmax=fmax, htk=htk)
        
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
    
    def compute_mel_nnaudio(self, audio: torch.Tensor, n_mels: int=229, n_fft: int=2048, \
        fmin: int=0, fmax: float | None=None, hop_length: int = 512, htk:bool=False,\
        window: str="hann", pad_mode: str="reflect", center: bool=True) -> torch.Tensor:
        """
            Compute the Mel spectrogram for a given audio segment.

            Args
            -----
                audio (torch.Tensor): Audio segment
                n_mels (int): Number of mel bands
                n_fft (int): FFT size or window size.
                fmin (int): Minimum frequency
                fmax (float | None): Maximum frequency
                hop_length (int): Hop length for Mel spectrogram computation.
                htk (bool): Default is False.
                window (str): Default window is hann
                pad_mode (str): Default pad mode is 'reflect'.
                center (bool): Center the window for STFT computation.

            Returns
            --------
                spec (torch.Tensor): Mel spectrogram of the audio segment
        """
        mel = MelSpectrogram(
            sr=self.sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, \
            htk=htk, fmin=fmin, fmax=fmax, pad_mode=pad_mode, center=center, \
            window=window
        )

        spec = mel(audio)
        return spec
    
    def compute_cqt_nnaudio(self, audio: torch.Tensor, n_bins: int=84, bins_per_octave: int=12, \
        fmin: int=32.7, fmax: float | None=None, hop_length: int = 512, \
        window: str="hann", pad_mode: str="reflect", center: bool=True) -> torch.Tensor:
        """
            Compute the CQT spectrogram for a given audio segment.

            Args
            -----
                audio (torch.Tensor): Audio segment
                n_bins (int): Number of frequency bins
                bins_per_octave (int): Number of bins per octave
                fmin (int): Minimum frequency
                fmax (float | None): Maximum frequency
                hop_length (int): Hop length for Mel spectrogram computation.
                window (str): Default window is hann
                pad_mode (str): Default pad mode is 'reflect'.
                center (bool): Center the window for STFT computation.

            Returns
            --------
                spec (torch.Tensor): CQT spectrogram of the audio segment
        """
        cqt = CQT(sr=self.sample_rate, bins_per_octave=bins_per_octave, n_bins=n_bins, hop_length=hop_length, \
            fmin=fmin, fmax=fmax, pad_mode=pad_mode, center=center, window=window
            )
        spec = cqt(audio)
        return spec
