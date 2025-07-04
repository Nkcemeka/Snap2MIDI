"""
File: handcrafted_features.py
Author: Chukwuemeka L. Nkama
Date: 4/11/2025
Description: Handcrafted features for audio
             processing!
"""

# Imports
import numpy as np
import librosa
from typing import List, Optional, Tuple

# Create class for handcrafted features
class HandcraftedFeatures:
    def __init__(self, \
            sample_rate: int = 22050, \
            window_size: float = 2048, \
            pr_rate: int = 4) -> None:
        """
            Default constructor for HandcraftedFeatures

            Args:
                sample_rate (int): Sample rate of audio
                window_size (float): Window size of audio
                pr_rate (int): Pitch rate of audio

            Returns:
                None
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.pr_rate = pr_rate

    def compute_cqt(self, audio: np.ndarray, bins_per_octave: int=24,\
                     num_octaves: int=6) -> np.ndarray:
        """
            Compute the CQT for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
                bins_per_octave (int): Number of bins per octave
                num_octaves (int): Number of octaves
            Returns:
                np.ndarray: CQT of the audio segment
        """
        # compute hop length to get CQT
        numerator = self.window_size * self.sample_rate
        denominator = (self.pr_rate * self.window_size) - 1
        hop_length = int(numerator / denominator)
        cqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=int(bins_per_octave*num_octaves), \
                          bins_per_octave=bins_per_octave, hop_length=hop_length)

        # convert to squared magnitude
        cqt = np.abs(cqt)**2 

        # find the log of the cqts
        cqt = np.log1p(cqt)
        return cqt

    def compute_mel(self, audio: np.ndarray, \
                    n_mels: int=229, n_fft: int=2048) -> np.ndarray:
        """
            Compute the Mel spectrogram for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
            Returns:
                np.ndarray: Mel spectrogram of the audio segment
        """
        # This assumes padding on both ends
        # (without padding, size of window needs to be subtracted)
        numerator = (self.window_size * self.sample_rate) 
        denominator = (self.pr_rate * self.window_size) - 1
        hop_length = int(numerator / denominator)
        mel = librosa.feature.melspectrogram(y=audio, hop_length=hop_length, 
                                             sr=self.sample_rate, 
                                             n_mels=n_mels, n_fft=n_fft)
        
        # convert to squared magnitude
        mel = np.abs(mel)**2
        # find the log of the mel spectrogram and clamp
        # to avoid log(0)
        mel = np.clip(mel, a_min=1e-10, a_max=None)
        mel = np.log(mel)
        return mel
