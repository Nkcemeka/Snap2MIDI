"""
    File: framed_signal.py
    Author: Chukwuemeka L. Nkama
    Description: A class that breaks down audio into
                 windowed segments
"""

# Imports
import numpy as np 
import librosa

class FramedAudio:
    def __init__(self, audio_path: str, hop_size: float, \
                 frame_size: float, sample_rate: float | None=None) -> None:
        """ 
            Args:
                audio_path (str): Path to audio file
                hop_size (float): hop size in seconds
                frame_size (float): frame size in seconds
                sample_rate (float): sample rate of audio file
            
            Returns:
                None
        """
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.framed_audio = self.generate_frames()
        self.index = 0
    
    def generate_frames(self) -> np.ndarray:
        """ 
            Generate audio frames

            Args:
                None
            
            Returns:
                framed_audio (np.ndarray): Shape of (num_frames, frame_size)
        """
        audio, sr = librosa.load(self.audio_path, sr=self.sample_rate)
        
        # Convert audio to MONO
        audio = librosa.to_mono(audio)

        self.hop_size = int(sr * self.hop_size)
        self.frame_size = int(self.frame_size * sr)

        # pad audio if necessary
        if (audio.shape[-1] - self.frame_size) % self.hop_size != 0:
            audio = np.pad(audio, (0, audio.shape[-1]%self.hop_size))

        # calc. no of frames
        num_frames = (audio.shape[-1] - self.frame_size)//self.hop_size + 1

        # Get framed audio
        framed_audio = np.zeros((num_frames, self.frame_size))
        for i in range(num_frames):
            framed_audio[i] = audio[i*self.hop_size:\
                                    (i*self.hop_size)+self.frame_size]
        
        return framed_audio

    def __getitem__(self, index: int) -> np.ndarray:
        return self.framed_audio[index]

    def __len__(self) -> int:
        return len(self.framed_audio)

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self) -> np.ndarray:
        if self.index < len(self.framed_audio):
            framed_audio =  self.framed_audio[self.index]
            self.index += 1
            return framed_audio
        raise StopIteration
