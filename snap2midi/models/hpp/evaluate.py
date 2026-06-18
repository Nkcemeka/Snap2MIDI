from .inference import load_hpp
import torch
from snap2midi.utils.eval_mir import note_extract, notes_to_frames, multipitch_metrics
import numpy as np
from collections import defaultdict
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset_hpp import HPPDataset
from tqdm import tqdm
from nnAudio2.features.mel import MelSpectrogram

@torch.no_grad()
def evaluate_test(config):
    """ 
        Evaluate the OAF model
        based on the test set.

        Args
        ----
            config (dict): Configuration dictionary
        
        Returns
        -------
            avg_metrics_notes (dict): Note metrics
            avg_metrics_frames (dict): Frame metrics
    """
    # set model to eval mode
    dataset = HPPDataset(config, [f"{config["test_path"]}"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config.get("checkpoint_path", None)
    config["checkpoint_path"] = checkpoint_path
    assert Path(checkpoint_path).exists(), f"Checkpoint path {checkpoint_path} does not exist"
    model = load_hpp(config)

    # Initialize metrics
    metrics_frames = defaultdict(list)
    metrics_notes = defaultdict(list)
    metrics_velocities = defaultdict(list)

    threshold = config.get("threshold", 0.5)
    pitch_offset = config.get("pitch_offset", 21)
    frame_rate = config.get("frame_rate", 31.25)

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating results...."):
        audio = data["audio"]
        y_onset = data["onset"]
        y_offset = data["offset"]
        y_frame = data["frame"]
        y_velocity = data["velocity"]


        audio = audio.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_velocity = y_velocity.to(device)
        y_offset = y_offset.to(device)

        y_frame = y_frame[0]
        y_onset = y_onset[0]
        y_velocity = y_velocity[0]
        y_offset = y_offset[0]

        frame_num = y_frame.size()[-2]
        piano_roll_size = y_frame.size()[-2:]
        model.frame_num = frame_num
        model.piano_roll_size = piano_roll_size
        clip_len = 10240


        # Forward pass
        if frame_num <= clip_len:
            torch.cuda.empty_cache()
            predictions = model(audio)
        else:
            n_step = frame_num
            clip_list = [clip_len] * (n_step//clip_len)
            res = n_step % clip_len
            if(n_step > clip_len and res != 0):
                clip_list[-1] -= (clip_len - res)//2
                clip_list += [res + (clip_len - res)//2]
            
            begin = 0
            predictions = {}
            for clip in clip_list:
                end = begin + clip
                audio_i = audio[:, config["hop_length"]*begin:config["hop_length"]*end]
                torch.cuda.empty_cache()
                model.frame_num = clip
                predictions_i = model(audio_i)
                for key, item in predictions_i.items():
                    item = item.squeeze()
                    if(key in predictions):
                        predictions[key] = torch.cat([predictions[key], item], dim=0)
                    else:
                        predictions[key] = item
                begin += clip

        on_preds = predictions['onset']
        frame_preds = predictions['frame']
        vel_preds = predictions["velocity"]
        offset_preds = predictions["offset"]
        on_preds = on_preds.squeeze()
        frame_preds = frame_preds.squeeze()
        vel_preds = vel_preds.squeeze()

        # print(on_preds.max(), on_preds.min())
        # exit(1)
        note_preds, int_preds, vels = note_extract(on_preds, frame_preds, \
                                               vel_preds, onset_thresh=threshold, \
                                               frame_thresh=threshold)
        
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

        frame_pred = frame_pred[:frame_gt.shape[0], :]

        # convert the pitches to hz
        note_gt = 440 * (2 ** ((note_gt - 69) / 12))
        note_preds = 440 * (2 ** ((note_preds - 69) / 12))

        # Get the note-level scores
        note_off_scores_tuple = prf(
            ref_intervals=int_gt, ref_pitches=note_gt, \
            est_intervals=int_preds, est_pitches=note_preds, offset_ratio=None)
        
        note_off_scores = {'Precision_no_offset': note_off_scores_tuple[0],
                       'Recall_no_offset': note_off_scores_tuple[1], \
                        'F1_no_offset': note_off_scores_tuple[2]}

        note_scores_tuple = prf(ref_intervals=int_gt, ref_pitches=note_gt,
            est_intervals=int_preds, est_pitches=note_preds)
        
        note_scores = {'Precision': note_scores_tuple[0],
                       'Recall': note_scores_tuple[1],
                       'F1': note_scores_tuple[2]}
        
        
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
        torch.cuda.empty_cache()

    # Calculate the average scores
    avg_metrics_notes = {key: round(np.mean(value).item(), 3) for key, value in metrics_notes.items()}
    avg_metrics_frames = {key: round(np.mean(value).item(), 3) for key, value in metrics_frames.items()}
    avg_metrics_velocities = {key: round(np.mean(value).item(), 3) for key, value in metrics_velocities.items()}

    avg_metrics_notes.update(avg_metrics_velocities)
    return avg_metrics_notes, avg_metrics_frames
