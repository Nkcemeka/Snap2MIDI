from .utilities import frames_to_note, half_stride, transcription_metrics, \
    hft_frame_metrics, transcription_velocity_metrics
import torch
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from .inference import load_hft, mel_from_audio

@torch.no_grad()
def evaluate_test(config: dict):
    """
        Calculate the transcription metrics for a test set.

        Args:
            config (dict): Configuration dictionary containing
                           useful parameters and info.

        Returns:
            scores (dict): Dictionary containing the transcription metrics.
    """
    model = load_hft(config)
    test_dir = config["test_path"]
    test_files = sorted(Path(test_dir).glob("*.npz"))

    # The test set stores audio now, so the feature is rebuilt here from these
    # params rather than read back from disk. That introduces a way to evaluate
    # against a feature the extraction never produced -- pass mel_bins=229 to a
    # 256-bin extraction and you would silently score a different spectrogram.
    # n_bins is the model's input width, so requiring them equal pins it down.
    if config["mel_bins"] != config["n_bins"]:
        raise ValueError(
            f"mel_bins ({config['mel_bins']}) must match the model's n_bins "
            f"({config['n_bins']}); pass the same feature params extraction used.")

    trans_metrics = defaultdict(list)
    frame_metrics = defaultdict(list)

    for file in tqdm(test_files, total=len(test_files), desc="Extracting results...."):
        data = np.load(file, allow_pickle=True)

        # Extraction stores the waveform now, so the whole-track feature is
        # computed here. The collated division slab is a .npy and so is not
        # picked up by this glob.
        feature = mel_from_audio(torch.from_numpy(data['audio']), config).numpy()
        ref_notes = data['notes'] # dictionary of note events
        ref_frames = data['label_frames'].astype(int) # binary array of shape [num_frames, num_pitches]

        # Get the model output
        output = half_stride(model, feature, shift=32, config=config)
        onset = output[-4]
        offset = output[-3]
        frames = output[-2]
        velocity = output[-1]   

        # Convert regression roll to MIDI notes
        est_notes = frames_to_note(onset, offset, frames, velocity, config)

        for key, value in transcription_metrics(est_notes, ref_notes).items():
            trans_metrics[f"note_{key}"].append(value)
        
        for key, value in transcription_velocity_metrics(est_notes, ref_notes).items():
            trans_metrics[f"note_vel_{key}"].append(value)
        
        # Calculate frame metrics
        frame_scores = hft_frame_metrics(ref_notes, frames, config)
        for key, value in frame_scores.items():
            frame_metrics[f"frame_{key}"].append(value)

    # Calculate the average scores
    scores = {key: round(np.mean(value).item(), 3) for key, value in trans_metrics.items()}
    frame_scores = {key: round(np.mean(value).item(), 3) for key, value in frame_metrics.items()}
    return scores, frame_scores
