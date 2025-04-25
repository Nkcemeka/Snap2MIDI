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
            window_size: int = 2048, \
            pr_rate: int = 4) -> None:
        """
            Default constructor for HandcraftedFeatures

            Args:
                sample_rate (int): Sample rate of audio
                window_size (int): Window size of audio
                pr_rate (int): Pitch rate of audio

            Returns:
                None
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.pr_rate = pr_rate

    def compute_cqt(self, audio, bins_per_octave=24):
        """
            Compute the CQT for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
            Returns:
                np.ndarray: CQT of the audio segment
        """
        # compute hop length to get CQT
        numerator = self.window_size * self.sample_rate
        denominator = (self.pr_rate * self.window_size) - 1
        hop_length = int(numerator / denominator)
        cqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=144, \
                          bins_per_octave=bins_per_octave, hop_length=hop_length)

        # convert to squared magnitude
        cqt = np.abs(cqt)**2 

        # find the log of the cqts
        cqt = np.log1p(cqt)
        return cqt

    def compute_mel(self, audio, n_mels=229, n_fft=2048):
        """
            Compute the Mel spectrogram for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
            Returns:
                np.ndarray: Mel spectrogram of the audio segment
        """
        numerator = (self.window_size * self.sample_rate)
        denominator = self.pr_rate - 1
        hop_length = int(numerator / denominator)
        mel = librosa.feature.melspectrogram(y=audio, hop_length=hop_length, sr=self.sample_rate, n_mels=n_mels)
        
        # convert to squared magnitude
        mel = np.abs(mel)**2
        # find the log of the mel spectrogram and clamp
        # to avoid log(0)
        mel = np.clip(mel, a_min=1e-10, a_max=None)
        mel = np.log(mel)
        return mel
