import torch
import numpy as np

def get_mel(audio_frame: np.ndarray, hf, n_mels, n_fft, hop_length):
    mel = hf.compute_mel(audio_frame, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel = torch.tensor(mel, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    mel = mel.T
    mel = mel.unsqueeze(0)
    return mel

def stitch(arr: np.ndarray):
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
