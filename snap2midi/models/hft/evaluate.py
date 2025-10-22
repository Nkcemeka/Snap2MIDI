from .utilities import frames_to_note, half_stride, transcription_metrics, \
    hft_frame_metrics, transcription_velocity_metrics
import torch
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from .inference import load_hft


@torch.no_grad()
def evaluate_test(config: dict):
    """
        Calculate the transcription metrics for a test set.

        Args:
            test_dir (str): Directory containing the test set files.

        Returns:
            scores (dict): Dictionary containing the transcription metrics.
    """
    model = load_hft(config)
    test_dir = "data/hft/feature/test/"
    test_files = sorted(Path(test_dir).glob("*.npz"))

    trans_metrics = defaultdict(list)
    frame_metrics = defaultdict(list)

    for file in tqdm(test_files, total=len(test_files)-1, desc="Extracting results...."):
        if "dataset_feature.npz" in str(file):
            continue

        data = np.load(file, allow_pickle=True)
        feature = data['feature']
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
    scores = {key: round(np.mean(value).item(), 2) for key, value in trans_metrics.items()}
    frame_scores = {key: round(np.mean(value).item(), 2) for key, value in frame_metrics.items()}
    return scores, frame_scores

