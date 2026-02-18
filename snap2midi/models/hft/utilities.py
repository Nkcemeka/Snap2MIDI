# Imports
import torch
import numpy as np
import pretty_midi
import mir_eval

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
    # For the half-sride, we hop by half the number of frames
    half_frames = int(config["num_frame"]/2)

    # We will pad the feature behind and in front
    back_margin = config["margin_b"]
    front_margin = config["margin_f"]
    num_bins = config["n_bins"]
    num_notes = config["num_note"]

    # We will pad the feature with -18.42 behind (to see reason, why we added shift 
    # behind, see the paper)
    # -18.42 was used in the main code; not sure why....tbh
    pad_behind = np.full((back_margin + shift, num_bins), -18.42068099975586, dtype=np.float32)

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
        x = input[i:i+back_margin+config["num_frame"]+front_margin]
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

# change this to event_time_from_regression
def event_time_from_regression(event: np.ndarray, frame: int, dist_frames_secs: float, pitch: int) -> float:
    """
        Calculate the event time from the regression output.
        Args:
        -----
            event (np.ndarray): Regression output of shape (num_frames, num_pitches)
            frame (int): The frame at which the event is detected
            dist_frames_secs (float): The distance in seconds between frames
            pitch (int): The pitch index for which the event time is calculated
        Returns:
        --------
            event_time (float): The calculated event time in seconds
    """
    if (frame == 0) or (frame == len(event)-1):
        event_time = frame * dist_frames_secs
    elif event[frame - 1, pitch] == event[frame + 1, pitch]:
        event_time = frame * dist_frames_secs
    elif event[frame - 1, pitch] < event[frame+1, pitch]:
        # (A, B, C) where A is prev and C is next
        # Here A is less than C
        # See docs for this repo for the comple derivations
        # eqn: (yc - ya)/ 2*(yb - ya) 
        yc = event[frame+1, pitch]
        ya = event[frame - 1, pitch]
        yb = event[frame, pitch]
        event_dist_from_b = (yc - ya) / (2*(yb - ya))

        # Event for this case occurs to the right of B
        event_time = (frame * dist_frames_secs) + (event_dist_from_b * dist_frames_secs)
    else:
        # A is greater than C
        # eqn: (ya - yc) / 2*(yb - yc)
        yc = event[frame+1, pitch]
        ya = event[frame - 1, pitch]
        yb = event[frame, pitch]
        event_dist_from_b = (ya - yc) / (2*(yb - yc))

        # Event for this case occurs to the left of B
        event_time = (frame * dist_frames_secs) - (event_dist_from_b * dist_frames_secs)
    return event_time


def frames_to_note(onset, offset, frames, velocity, config):
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
    threshold_onset = config["onset_threshold"]
    threshold_offset = config["offset_threshold"]
    threshold_frames = config["frame_threshold"]
    # note that onsets and offset are regression outputs and indicate the distance to the nearest note onset or offset
    # frames is a binary output indicating the presence of a note at that frame
    notes = []
    dist_frames_secs = float(config['hop_sample'] / config['sr'])

    for pitch in range(config['num_note']):
        onset_detect = []
        for frame in range(onset.shape[0]):
            if (onset[frame, pitch]) >= threshold_onset:
                if local_maxima(onset[:, pitch], frame):
                    # calc. the actual onset time is based on Kong's eqns (see Section C)
                    onset_time = event_time_from_regression(onset, frame, dist_frames_secs, pitch)
                    onset_detect.append({'frame': frame, 'onset': onset_time})
        
        offset_detect = []
        for frame in range(offset.shape[0]):
            if (offset[frame, pitch]) >= threshold_offset:
                if local_maxima(offset[:, pitch], frame):
                    # calc. the actual offset time is based on Kong's eqns (see Section C)
                    offset_time = event_time_from_regression(offset, frame, dist_frames_secs, pitch)
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
                frame_next_onset = len(frames)
                next_onset_time = (frame_next_onset - 1) * dist_frames_secs
            
            # Now, lets see if the offset is within the range of the next onset
            frame_offset = frame_onset + 1 # we assume this first
            offset_flag = False

            for j, offset_dict in enumerate(offset_detect):
                if frame_onset < offset_dict['frame']:
                    # If the frame onset is less than the frame offset, we can assume that
                    # the offset is within the range of the next onset
                    frame_offset = offset_dict['frame']
                    offset_time = offset_dict['offset']
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
            for idx in range(frame_onset + 1, frame_next_onset):
                if frames[idx, pitch] < threshold_frames:
                    # If the frame is less than the threshold, we can assume that the note has ended
                    frame_idx = idx
                    frames_flag = True
                    time_frames = frame_idx * dist_frames_secs
                    break
            
            midi_pitch = int(pitch + config['note_min'])
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
                if frame_offset <= frame_idx:
                    # If the frame offset is less than the frame index, we will use the frame offset
                    final_offset = float(offset_time)
                else:
                    # If the frame offset is greater than the frame index, we will use the frame index
                    final_offset = float(time_frames)
                #final_offset = min(float(offset_time), float(time_frames))
            
            # Now we can append the note to the notes list
            if velocity_value > 0:
                notes.append({
                    'pitch': midi_pitch,
                    'onset': float(onset_time),
                    'offset': final_offset,
                    'velocity': velocity_value
                })

            if onset_time >= final_offset:
                assert False, f"Onset time {onset_time} should be less than offset time {frames_flag} {offset_flag}, {frame_onset}, {frame_offset},\
                      {final_offset}, {time_frames}, {offset_time} for pitch {midi_pitch}"

            # Fix the offset of the previous note if it comes after the onset of the current note
            if len(notes) > 1 and \
            (notes[len(notes)-1]['pitch'] == notes[len(notes)-2]['pitch']) and \
                   (notes[len(notes)-1]['onset'] < notes[len(notes)-2]['offset']):
                    notes[len(notes)-2]['offset'] = notes[len(notes)-1]['onset']

    # sort the notes by pitch and then by onset time
    notes.sort(key=lambda x: (x['pitch'], x['onset']))
    return notes

def notes_to_midi(notes):
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
    return midi_obj

def transcription_metrics(est_notes: list, ref_notes: list):
    """
        Calculate the transcription metrics using mir_eval.

        Args:
            est_notes (list): List of estimated notes
            ref_notes (list): List of reference notes

        Returns:
            scores (dict): Dictionary containing the transcription metrics.
    """
    est_int = []
    ref_int = []
    est_p = []
    ref_p = []

    for note in est_notes:
        onset_time = note['onset']
        offset_time = note['offset']
        est_pitch = note['pitch']
        # convert est_pitch to frequency 
        est_freq = pretty_midi.note_number_to_hz(est_pitch)
        assert 20 <= est_freq <= 20000, f"Estimated frequency {est_freq} is out of range (20-20000 Hz)"
        assert onset_time < offset_time, f"Onset time should be less than offset time: {onset_time} >= {offset_time}"
        est_int.append([onset_time, offset_time])
        est_p.append(est_freq)

    for note in ref_notes:
        onset_time = note['onset'] if type(note['onset']) is float else note['onset'].item()
        offset_time = note['offset'] if type(note['offset']) is float else note['offset'].item()
        ref_pitch = note['pitch'] if type(note['pitch']) is int else note['pitch'].item()

        # convert ref_pitch to frequency
        ref_freq = pretty_midi.note_number_to_hz(ref_pitch)
        assert 20 <= ref_freq <= 20000, f"Reference frequency {ref_freq} is out of range (20-20000 Hz)"
        ref_int.append([onset_time, offset_time])
        ref_p.append(ref_freq)

    est_int = np.array(est_int, dtype=np.float32)
    ref_int = np.array(ref_int, dtype=np.float32)
    est_p = np.array(est_p, dtype=np.float32)
    ref_p = np.array(ref_p, dtype=np.float32)

    # Calculate the transcription metrics using mir_eval
    scores = mir_eval.transcription.evaluate(
        ref_intervals=ref_int, ref_pitches=ref_p,
        est_intervals=est_int, est_pitches=est_p)
    
    results = {}
    for key in scores:
        if key in ['Precision', 'Recall', 'Precision_no_offset', 'Recall_no_offset']:
            results[key] = scores[key]
    return results



def transcription_velocity_metrics(est_notes: list, ref_notes: list):
    """
        Calculate the transcription velocity metrics using mir_eval.

        Args:
            est_notes (list): List of estimated notes
            ref_notes (list): List of reference notes

        Returns:
            scores (dict): Dictionary containing the transcription velocity metrics.
    """
    est_int = []
    ref_int = []
    est_p = []
    ref_p = []
    est_vel = []
    ref_vel = []

    for note in est_notes:
        onset_time = note['onset'] if type(note['onset']) is float else note['onset'].item()
        offset_time = note['offset'] if type(note['offset']) is float else note['offset'].item()
        est_pitch = note['pitch'] if type(note['pitch']) is int else note['pitch'].item()
        est_velocity = note['velocity'] if type(note['velocity']) is int else note['velocity'].item()
        # convert est_pitch to frequency 
        est_freq = pretty_midi.note_number_to_hz(est_pitch)
        est_int.append([onset_time, offset_time])
        est_p.append(est_freq)
        est_vel.append(est_velocity)

    for note in ref_notes:
        onset_time = note['onset']
        offset_time = note['offset']
        ref_pitch = note['pitch']
        ref_velocity = note['velocity']
        # convert ref_pitch to frequency
        ref_freq = pretty_midi.note_number_to_hz(ref_pitch)
        ref_int.append([onset_time, offset_time])
        ref_p.append(ref_freq)
        ref_vel.append(ref_velocity)

    est_int = np.array(est_int, dtype=np.float32)
    ref_int = np.array(ref_int, dtype=np.float32)
    est_p = np.array(est_p, dtype=np.float32)
    ref_p = np.array(ref_p, dtype=np.float32)
    est_vel = np.array(est_vel, dtype=np.int8)
    ref_vel = np.array(ref_vel, dtype=np.int8)

    # Calculate the transcription metrics using mir_eval
    scores = mir_eval.transcription_velocity.evaluate(
        ref_intervals=ref_int, ref_pitches=ref_p, ref_velocities=ref_vel,
        est_intervals=est_int, est_pitches=est_p, est_velocities= est_vel
    )

    results = {}
    for key in scores:
        if key in ['Precision', 'Recall', 'Precision_no_offset', 'Recall_no_offset']:
            results[key] = scores[key]

    return results


def hft_frame_metrics(ref_notes, est_frames, config):
    """
        Calculate the frame metrics according to the 
        HfT paper or more specifically, the codebase.
    """
    thresh_frames = config["frame_threshold"]
    # Get duration of reference notes
    duration_ref = 0.0
    for note in ref_notes:
        if note['offset'] > duration_ref:
            duration_ref = note['offset']
    
    # Get the number of frames in the reference notes
    # dist btw two frames in samples is config['feature']['hop_sample'] (samples/frame)
    # dist_frames_secs = float(config['hop_sample'] / config['sr']) (secs/frame)
    num_frames_ref = int(duration_ref * config['sr'] / config['hop_sample'] + 0.5)
    ref_arr = np.zeros((num_frames_ref, 128), dtype=int)

    for note in ref_notes:
        onset_frame = int(note['onset'] * config['sr'] / config['hop_sample'] + 0.5)
        offset_frame = int(note['offset'] * config['sr'] / config['hop_sample'] + 0.5)
        pitch = note['pitch']
        ref_arr[onset_frame:offset_frame, pitch] = 1
    
    num_frames = min(ref_arr.shape[0], est_frames.shape[0])

    # Calculate the frame metrics
    ref_time = []
    est_time = []
    ref_freqs = []
    est_freqs = []
    for frame in range(num_frames):
        time = frame * config['hop_sample'] / config['sr']
        ref_time.append(time)
        est_time.append(time)

        # For the reference, we will get the frequencies where a note is present
        midi_pitch_ref = np.where(ref_arr[frame, :] > 0)[0]
        if len(midi_pitch_ref) > 0:
            ref_freqs.append(pretty_midi.note_number_to_hz(midi_pitch_ref))
        else:
            ref_freqs.append(np.array([]))

        # For the estimated frames, we will get the frequencies where a note is present
        midi_pitch_est = np.where(est_frames[frame, :] >= thresh_frames)[0]
        if len(midi_pitch_est) > 0:
            # est_frames array might have less than 128
            est_freqs.append(pretty_midi.note_number_to_hz(midi_pitch_est + config['note_min']))
        else:
            est_freqs.append(np.array([]))
    
    # Now we can calculate the frame metrics
    ref_time = np.array(ref_time, dtype=np.float32)
    est_time = np.array(est_time, dtype=np.float32)
    scores = mir_eval.multipitch.evaluate(
        ref_time=ref_time, ref_freqs=ref_freqs,
        est_time=est_time, est_freqs=est_freqs
    )

    results = {}
    for key in scores:
        if key in ['Precision', 'Recall']:
            results[key] = scores[key]
    return results

