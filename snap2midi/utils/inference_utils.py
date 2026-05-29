import torch
import numpy as np
from snap2midi.extractor.utils.handcrafted_features import HandcraftedFeatures as hf

def get_mel(audio_frame: np.ndarray, hf: hf, n_mels: int, n_fft: int, hop_length: int):
    """ 
        Get the mel spectrogram for an audio frame.

        Args
        ----
            audio_frame (np.ndarray): Audio frame
            hf (hf): Handcrafted feature object
            n_mels (int): Number of mel bands
            n_fft (int): FFT size
            hop_length (int): Hop length for mel spectrogram computation

        Returns
        -------
            mel (torch.Tensor): Computed mel spectrogram.
    """
    mel = hf.compute_mel(audio_frame, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel = torch.tensor(mel, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    mel = mel.T
    mel = mel.unsqueeze(0)
    return mel

def stitch(arr: np.ndarray):
    """ 
        Stitches the results from the model
        based on a 50% overlap sliding window.
        This uses Kong's method and is a numpy-
        based implementation.

        Args
        ----
            arr (np.ndarray): Results from model
        
        Returns
        -------
            result (np.ndarray): Stitched results
    """
    # Stitches the results from the model
    # so that everything aligns
    arr = arr[:, :-1, :]
    result = []
    num_segments, num_frames, _ = arr.shape
    factor_75 = int(num_frames * 0.75)
    factor_25 = int(num_frames * 0.25)
    result.append(arr[0, :factor_75])
    for each in range(1, num_segments - 1):
        result.append(arr[each, factor_25 : factor_75])
    result.append(arr[-1, factor_25:])
    result = np.concatenate(result)
    return result


def stitch_tensor(arr: torch.Tensor):
    """ 
        Stitches the results from the model
        based on a 50% overlap sliding window.
        This uses Kong's method and is a tensor-
        based implementation.

        Args
        ----
            arr (torch.Tensor): Results from model
        
        Returns
        -------
            result_concat (torch.Tensor): Stitched results
    """
    # Stitches the results from the model
    # so that everything aligns
    arr = arr[:, :-1]
    result = []
    num_segments, num_frames, _ = arr.shape
    factor_75 = int(num_frames * 0.75)
    factor_25 = int(num_frames * 0.25)
    result.append(arr[0, :factor_75])
    for each in range(1, num_segments - 1):
        result.append(arr[each, factor_25 : factor_75])
    result.append(arr[-1, factor_25:])
    result_concat = torch.concatenate(result)
    return result_concat
