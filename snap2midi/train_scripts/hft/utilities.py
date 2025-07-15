"""
    File: utilities.py
    Author: Chukwuemeka L. Nkama
    Date: 7/14/2025
    Description: Utility script for HFT!
"""

# Imports
import torch
import numpy as np
import pretty_midi


def half_stride(model, feature, shift: int, config: dict):
    """
        Transcription through the half-stride strategy 
        from the HfT paper.

        Args:
            model (torch.nn.Module): The trained HfT model.
            feature (np.ndarray): The input feature of shape [num_frames, n_mels].
            shift (int): The number of frames to shift the window.
            config (dict): The configuration dictionary containing input parameters.
    """

    # feature is of shape [num_frames, n_mels]
    feature = np.array(feature, dtype=np.float32)

    # For the half-sride, we hop by half the number of frames
    half_frames = int(config["input"]["num_frame"]/2)

    # We will pad the feature behind and in front
    back_margin = config["input"]["margin_b"]
    front_margin = config["input"]["margin_f"]
    num_bins = config["feature"]["n_bins"]
    num_notes = config["midi"]["num_note"]

    # We will pad the feature with zeros behind (to see reason, why we added shift 
    # behind, see the paper)
    pad_behind = np.zeros((back_margin + shift, num_bins), dtype=np.float32)

    # now, we want to move our window such that the entire length is a multiple of the half_frames
    # np.ceil below allows us know what length to add to make full_length a multiple of half_frames
    # why add half_frame??? Now, when we get to the last segemtn (if feature is not a multuple of half_frames),
    # considering the backward padding, we need N frames to be able to get the last segment
    # so, if remainder is what is needed to make the feature a multiple of half_frames, we need an additional
    # half_frames to be able to get the last segment
    # so, we add half_frames to the remainder
    full_length = feature.shape[0] + back_margin + front_margin + half_frames
    remainder = int(np.ceil(full_length / half_frames) * half_frames - full_length)

    # Now we pad the feature in front
    # We subtract shift since we added shift for the back padding
    pad_forward = np.zeros((front_margin + remainder + half_frames - shift, num_bins), dtype=np.float32)
    input = torch.from_numpy(np.concatenate([pad_behind, feature, pad_forward], axis=0))

    # init our output arrays
    output_onset_1st = np.zeros((feature.shape[0]+remainder, num_notes), dtype=np.float32)
    output_offset_1st = np.zeros_like(output_onset_1st, dtype=np.float32)
    output_frames_1st = np.zeros_like(output_onset_1st, dtype=np.float32)
    output_velocity_1st = np.zeros_like(output_onset_1st, dtype=np.int8)

    output_onset_2nd = np.zeros_like(output_onset_1st, dtype=np.float32)
    output_offset_2nd = np.zeros_like(output_offset_1st, dtype=np.float32)
    output_frames_2nd = np.zeros_like(output_frames_1st, dtype=np.float32)
    output_velocity_2nd = np.zeros_like(output_velocity_1st, dtype=np.int8)

    # Set model to eval mode
    model.eval()
    for i in range(0, feature.shape[0], half_frames):
        x = input[i:i+back_margin+config["input"]["num_frame"]+front_margin]
        x = (x.T).unsqueeze(0)  # shape [1, n_mels, num_frames]
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            output_onset_1st_i, output_offset_1st_i, output_frames_1st_i, output_velocity_1st_i, \
                attention, output_onset_2nd_i, output_offset_2nd_i, output_frames_2nd_i, output_velocity_2nd_i = model(x)
            
        output_onset_1st[i:i+half_frames] = (output_onset_1st_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_offset_1st[i:i+half_frames] = (output_offset_1st_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_frames_1st[i:i+half_frames] = (output_frames_1st_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_velocity_1st[i:i+half_frames] = (output_velocity_1st_i.squeeze(0))[shift:shift+half_frames].argmax(2).cpu().detach().numpy()
        output_onset_2nd[i:i+half_frames] = (output_onset_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_offset_2nd[i:i+half_frames] = (output_offset_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_frames_2nd[i:i+half_frames] = (output_frames_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_velocity_2nd[i:i+half_frames] = (output_velocity_2nd_i.squeeze(0))[shift:shift+half_frames].argmax(2).cpu().detach().numpy()
    
    return (output_onset_1st, output_offset_1st, output_frames_1st, output_velocity_1st,
            output_onset_2nd, output_offset_2nd, output_frames_2nd, output_velocity_2nd)


def local_maxima(reg_roll: np.ndarray, frame: int) -> bool:
    """
        local_maxima checks if there is a local maximum
        at [reg_roll[n] (a triangle shape with x[n] as the peak)

        Args:
        -----
            reg_roll (np.ndarray): Regression roll of shape (num_frames) at a given pitch
            frame (int): frame position
        
        Returns:
        --------
            maxim (bool): True/False at x[n]
    """
    # initialize the maxim flags
    maxim_left = True
    maxim_right = True

    for i in range(frame - 1, -1, -1):
        if (reg_roll[frame] < reg_roll[i]):
            maxim_left = False
            break
        elif (reg_roll[frame] > reg_roll[i]):
            maxim_left = True
            break

    for i in range(frame + 1, len(reg_roll)):
        if (reg_roll[frame] < reg_roll[i]):
            maxim_right = False
            break
        elif (reg_roll[frame] > reg_roll[i]):
            maxim_right = True
            break

    maxim = maxim_left and maxim_right
    return maxim


def onset_time_from_regression(onset: np.ndarray, frame: int, dist_frames_secs: float, pitch: int) -> float:
    """
        Calculate the onset time from the regression output.
        Args:
        -----
            onset (np.ndarray): Regression output of shape (num_frames, num_pitches)
            frame (int): The frame at which the onset is detected
            dist_frames_secs (float): The distance in seconds between frames
            pitch (int): The pitch index for which the onset time is calculated
        Returns:
        --------
            onset_time (float): The calculated onset time in seconds
    """
    if onset[frame - 1, pitch] == onset[frame, pitch]:
        onset_time = frame
    elif onset[frame - 1, pitch] < onset[frame+1, pitch]:
        # (A, B, C) where A is prev and C is next
        # Here A is less than C
        # See docs for this repo for the comple derivations
        # eqn: (yc - ya)/ 2*(yb - ya) 
        yc = onset[frame+1, pitch]
        ya = onset[frame - 1, pitch]
        yb = onset[frame, pitch]
        onset_dist_from_b = (yc - ya) / (2*(yb - ya))

        # Onset for this case occurs to the right of B
        onset_time = (frame * dist_frames_secs) + (onset_dist_from_b * dist_frames_secs)
    else:
        # A is greater than C
        # eqn: (ya - yc) / 2*(yb - yc)
        yc = onset[frame+1, pitch]
        ya = onset[frame - 1, pitch]
        yb = onset[frame, pitch]
        onset_dist_from_b = (ya - yc) / (2*(yb - yc))

        # Onset for this case occurs to the left of B
        onset_time = (frame * dist_frames_secs) - (onset_dist_from_b * dist_frames_secs)
    return onset_time


def frames_to_note(onset, offset, frames, velocity, threshold_onset, threshold_offset, threshold_frames, config):
    """
        Convert the onset, offset, frames, and velocity outputs to a list of notes.

        Args:
        -----
            onset (np.ndarray): Onset regression output of shape (num_frames, num_pitches).
            offset (np.ndarray): Offset regression output of shape (num_frames, num_pitches).
            frames (np.ndarray): Frames binary output of shape (num_frames, num_pitches).
            velocity (np.ndarray): Velocity output of shape (num_frames, num_pitches).
            threshold_onset (float): Threshold for detecting onsets.
            threshold_offset (float): Threshold for detecting offsets.
            threshold_frames (float): Threshold for detecting frames.
            config (dict): Configuration dictionary containing MIDI parameters.

        Returns:
        --------
            notes (list): List of detected notes with their pitch, onset, offset, and velocity.
    """
    # note that onsets and offset are regression outputs and indicate the distance to the nearest note onset or offset
    # frames is a binary output indicating the presence of a note at that frame
    notes = []
    dist_frames_secs = float(config['feature']['hop_sample'] / config['feature']['sr'])

    for pitch in range(config['midi']['num_note']):
        onset_detect = []
        for frame in range(onset.shape[0]):
            if (onset[frame, pitch]) >= threshold_onset:
                if local_maxima(onset[:, pitch], frame):
                    # calc. the actual onset time is based on Kong's eqns (see Section C)
                    onset_time = onset_time_from_regression(onset, frame, dist_frames_secs, pitch)
                    onset_detect.append({'frame': frame, 'onset': onset_time})
        
        offset_detect = []
        for frame in range(offset.shape[0]):
            if (offset[frame, pitch]) >= threshold_offset:
                if local_maxima(offset[:, pitch], frame):
                    # calc. the actual offset time is based on Kong's eqns (see Section C)
                    offset_time = onset_time_from_regression(offset, frame, dist_frames_secs, pitch)
                    offset_detect.append({'frame': frame, 'offset': offset_time})

        # We will now get the onsets and offsets and store them in notes
        next_onset_time = 0.0
        offset_time = 0.0
        time_frames = 0.0
        for i, onset_dict in enumerate(onset_detect):
            frame_onset = onset_dict['frame']
            onset_time = onset_dict['onset']

            if i+1 < len(onset_detect):
                # Get the next onset (this will help us with the offset)
                frame_next_onset = onset_detect[i+1]['frame']
                next_onset_time = onset_detect[i+1]['onset']
            else:
                # If this is the last onset, we will use the frames
                frame_next_onset = len(frames) - 1
                next_onset_time = (frame_next_onset - 1) * dist_frames_secs
            
            # Now, lets see if the offset is within the range of the next onset
            frame_offset = frame_onset + 1 # we assume this first
            offset_flag = False

            for j, offset_dict in enumerate(offset_detect):
                frame_offset = offset_dict['frame']
                offset_time = offset_dict['offset']

                if frame_onset < frame_offset:
                    # If the frame onset is less than the frame offset, we can assume that
                    # the offset is within the range of the next onset
                    offset_flag = True
                    break
            
            if frame_offset > frame_next_onset:
                # If the frame offset is greater than the next onset, we will use the next onset
                # as the offset
                frame_offset = frame_next_onset
                offset_time = next_onset_time
            
            # Now we will check for what the offset could be using the frames
            frame_idx = frame_onset + 1
            frames_flag = False
            for idx in range(frame_onset + 1, len(frames)):
                if frames[idx, pitch] < threshold_frames:
                    # If the frame is less than the threshold, we can assume that the note has ended
                    frame_idx = idx
                    frames_flag = True
                    time_frames = frame_idx * dist_frames_secs
                    break
            
            midi_pitch = int(pitch + config['midi']['note_min'])
            velocity_value = int(velocity[frame_onset, pitch])

            if offset_flag is False and frames_flag is False:
                # If there is no offset and no frames, we will use the next onset as the offset
                final_offset = float(next_onset_time)
            elif offset_flag is True and frames_flag is False:
                # If there is an offset but no frames, we will use the offset
                final_offset = float(offset_time)
            elif offset_flag is False and frames_flag is True:
                # If there is no offset but frames, we will use the frames
                final_offset = float(time_frames)
            else:
                # If there is both an offset and frames, we will use the minimum of the two
                final_offset = min(float(offset_time), float(time_frames))
            
            # Now we can append the note to the notes list
            notes.append({
                'pitch': midi_pitch,
                'onset': float(onset_time),
                'offset': final_offset,
                'velocity': velocity_value
            })
    
    # sort the notes by pitch and then by onset time
    notes.sort(key=lambda x: (x['pitch'], x['onset']))
    return notes

def notes_to_midi(notes, filename):
    """
        Convert a list of notes to a MIDI file.

        Args:
        -----
            notes (list): List of notes, where each note is a dictionary with keys 'pitch', 'onset', 'offset', and 'velocity'.
            filename (str): The name of the output MIDI file.
    """
    # Create a PrettyMIDI object and add the notes to it
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)
    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=note['velocity'],
            pitch=note['pitch'],
            start=note['onset'],
            end=note['offset']
        )
        piano.notes.append(midi_note)
    midi_obj.instruments.append(piano)
    midi_obj.write(filename+".mid")
    return midi_obj


            
