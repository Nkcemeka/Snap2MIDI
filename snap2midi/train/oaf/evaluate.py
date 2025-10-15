from .inference import load_oaf
import torch
from snap2midi.utils.eval_mir import note_extract, notes_to_frames, multipitch_metrics
import numpy as np
from collections import defaultdict
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset_oaf import OAFDataset
from tqdm import tqdm

@torch.no_grad()
def evaluate_test(config):
    # set model to eval mode
    dataset = OAFDataset(["./data/oaf/test"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config.get("checkpoint_path", None)
    assert Path(checkpoint_path).exists(), f"Checkpoint path {checkpoint_path} does not exist"
    model = load_oaf(config)

    # Initialize metrics
    metrics_frames = defaultdict(list)
    metrics_notes = defaultdict(list)
    metrics_velocities = defaultdict(list)

    threshold = config.get("threshold", 0.5)
    pitch_offset = config.get("pitch_offset", 21)
    frame_rate = config.get("frame_rate", 31.25)

    for i, (x, y_frame, y_onset, y_velocity,\
             label_weights, audio) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting results...."):
        x = x.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_velocity = y_velocity.to(device).float()
        label_weights = label_weights.to(device).float()

        y_frame = y_frame[0]
        y_onset = y_onset[0]
        y_velocity = y_velocity[0]


        # Forward pass
        on_preds, _, frame_preds, vel_preds = model(x)
        on_preds = torch.sigmoid(on_preds)[0]
        frame_preds = torch.sigmoid(frame_preds)[0]
        vel_preds = vel_preds[0]

        note_preds, int_preds, vels = note_extract(on_preds, frame_preds, \
                                               vel_preds, onset_thresh=threshold, \
                                               frame_thresh=threshold)
        # clamp velocities to [0, 1] using numpy
        vels = np.clip(vels, 0, 1)
        vels = 80*vels + 10 # This was recommended in the paper
        
        # convert int_preds to time in seconds
        int_preds = int_preds / frame_rate
        note_preds += pitch_offset

        note_gt, int_gt, vel_gt = note_extract(y_onset, \
                                            y_frame, \
                                            y_velocity)
        int_gt = int_gt / frame_rate
        note_gt += pitch_offset

        frame_pred = notes_to_frames(note_preds-pitch_offset, (int_preds*frame_rate).astype(int), \
                                  on_preds.shape)
        frame_gt = notes_to_frames(note_gt-pitch_offset, (int_gt * frame_rate).astype(int), \
                               y_frame.shape)

        # convert the pitches to hz
        note_gt = 440 * (2 ** ((note_gt - 69) / 12))
        note_preds = 440 * (2 ** ((note_preds - 69) / 12))

        # Get the note-level scores
        note_off_scores_tuple = prf(
            ref_intervals=int_gt, ref_pitches=note_gt, \
            est_intervals=int_preds, est_pitches=note_preds, offset_ratio=None)
        
        note_off_scores = {'Precision_no_offset': note_off_scores_tuple[0],
                       'Recall_no_offset': note_off_scores_tuple[1]}

        note_scores_tuple = prf(ref_intervals=int_gt, ref_pitches=note_gt,
            est_intervals=int_preds, est_pitches=note_preds)
        
        note_scores = {'Precision': note_scores_tuple[0],
                       'Recall': note_scores_tuple[1]}
        
        
        # Get the note-level-with velocities scores
        note_vel_scores_tuples = prf_vel(
            ref_intervals=int_gt, ref_pitches=note_gt, ref_velocities=vel_gt,
            est_intervals=int_preds, est_pitches=note_preds, est_velocities=vels, velocity_tolerance=0.1)
        
        note_vel_scores = {'Precision': note_vel_scores_tuples[0],
                       'Recall': note_vel_scores_tuples[1]}

        # Get the frame-level scores
        frame_scores_dict = multipitch_metrics(frame_gt, frame_pred, frame_rate)
        frame_scores = {'Precision': frame_scores_dict['Precision'],
                        'Recall': frame_scores_dict['Recall']}


        # Append the scores
        for key in note_scores.keys():
            metrics_notes[f"note_"+key].append(note_scores[key])
        for key in note_off_scores.keys():
            metrics_notes[f"note_"+key].append(note_off_scores[key])
        for key in frame_scores.keys():
            metrics_frames[f"frame_"+key].append(frame_scores[key])
        for key in note_vel_scores.keys():
            metrics_velocities[f"note_vel_"+key].append(note_vel_scores[key])

    # Calculate the average scores
    avg_metrics_notes = {key: round(np.mean(value).item(), 2) for key, value in metrics_notes.items()}
    avg_metrics_frames = {key: round(np.mean(value).item(), 2) for key, value in metrics_frames.items()}
    avg_metrics_velocities = {key: round(np.mean(value).item(), 2) for key, value in metrics_velocities.items()}

    avg_metrics_notes.update(avg_metrics_velocities)
    return avg_metrics_notes, avg_metrics_frames
