"""
File: eval_mir.py
Author: Chukwuemeka L. Nkama
Date: 4/9/2025
Description: MIR_Eval Evaluation Script for transcription!
"""

# Imports
import numpy as np
import mir_eval
import torch

# Function that takes a piano roll and gets the
# ref and est intervals and pitches
def get_intervals_and_pitches(piano_roll: np.ndarray, \
                              frame_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the intervals and pitches from a piano roll.
    
    Args:
        piano_roll (np.ndarray): Piano roll.
        frame_rate (int): Frames per second

    Returns:
        intervals (np.ndarray): Onset and offset events.
        pitches (np.ndarray): Pitches of events.
    """

    # pad the piano roll at the top and bottom with zeros
    padded_roll = np.pad(piano_roll.astype(int), ((1, 1), (0, 0)), mode='constant')
    event_roll = np.diff(padded_roll, axis=0)

    # Get the onset and offset for each note
    onset_events = np.argwhere(event_roll == 1)
    offset_events = np.argwhere(event_roll == -1)

    # Get pitches and intervals
    pitches = onset_events[:, 1]
    assert onset_events[:, 0].shape == offset_events[:, 0].shape, \
        f"{onset_events[:, 0].shape}, {offset_events[:, 0].shape}."

    intervals = np.concatenate((onset_events[:, 0].reshape(-1, 1), \
                                offset_events[:, 0].reshape(-1, 1)), axis=1)

    # Convert intervals to seconds
    intervals_arr = intervals / frame_rate

    # Convert pitches to Hz
    pitches_arr = 440 * (2 ** ((pitches - 69) / 12))

    return intervals_arr, pitches_arr

# Function to calculate transcription metrics
def transcription_metrics(pred: np.ndarray, gt: np.ndarray, \
                          frame_rate: int|None=None) -> dict | None:
    """
    Calculate transcription metrics using mir_eval.
    Args:
        pred (np.ndarray): Predicted piano roll.
        gt (np.ndarray): Ground truth piano roll.
        frame_rate (int): Frames per second

    Returns:
        scores (dict): Dictionary containing the calculated metrics.
    """
    if frame_rate is None:
        raise ValueError("Frame rate must be specified.")
    
    # Get the ref and est intervals and pitches
    ref_i, ref_p = get_intervals_and_pitches(gt, frame_rate)
    est_i, est_p = get_intervals_and_pitches(pred, frame_rate)

    if len(ref_p) == 0:
        return None

    scores = mir_eval.transcription.evaluate(
        ref_intervals=ref_i, ref_pitches=ref_p, \
        est_intervals=est_i, est_pitches=est_p)
    return scores

def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(int)
    gt = gt.astype(int)
    true_pos = np.logical_and(pred == 1, gt == 1).sum()
    false_pos = np.logical_and(pred == 1, gt == 0).sum()
    false_neg = np.logical_and(pred == 0, gt == 1).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    accuracy = true_pos / (true_pos + false_pos + false_neg) \
          if (true_pos + false_pos + false_neg) > 0 else 0

    scores = {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }
    return scores

# Function to get multipitch intervals and pitches
def get_multipitch_intervals_and_pitches(piano_roll: np.ndarray, \
                                         frame_rate: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Get the multipitch intervals and pitches from a piano roll.
    Args:
        piano_roll (np.ndarray): Piano roll.
        frame_rate (int): Frames per second
    Returns:
        times (np.ndarray): Times of the active notes.
        pitches list(np.ndarray): Pitches of the active notes.
    """
    times = np.arange(piano_roll.shape[0])

    # max_freq is 5000Hz which in midi pitch is 111
    # and min_freq is 20Hz which in midi pitch is 16
    min_freq_idx = 16
    max_freq_idx = 112
    # Get the pitches of the active notes
    pitches = [np.nonzero(piano_roll.astype(int)[t, min_freq_idx:max_freq_idx])[0] + min_freq_idx \
               for t in times]

    # Convert times to seconds
    times = times / frame_rate
    # Convert pitches to Hz 
    pitches_arr = [440 * (2 ** ((p - 69) / 12)) for p in pitches]

    return times, pitches_arr   

# Function to calculate multipitch metrics
def multipitch_metrics(pred: np.ndarray, gt: np.ndarray, \
                       frame_rate: int|None=None) -> dict:
    """
    Calculate multipitch metrics using mir_eval.
    Args:
        pred (np.ndarray): Predicted multipitch.
        gt (np.ndarray): Ground truth multipitch.
        frame_rate (int): Frames per second

    Returns:
        scores: Dictionary containing the calculated metrics.
    """
    if frame_rate is None:
        raise ValueError("Frame rate must be specified.")
    
    # Get the ref time and pitches
    ref_time, ref_freqs = get_multipitch_intervals_and_pitches(gt, frame_rate)
    est_time, est_freqs = get_multipitch_intervals_and_pitches(pred, frame_rate)
    
    scores = mir_eval.multipitch.evaluate(
        ref_time=ref_time, ref_freqs=ref_freqs, \
        est_time=est_time, est_freqs=est_freqs
    )
    return scores


def note_extract(onset_roll: torch.Tensor, frame_roll: torch.Tensor, \
                 velocity_roll: torch.Tensor, onset_thresh: float=0.5, \
                 frame_thresh: float=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract notes from the piano roll.
    Args:
        onset_roll (torch.Tensor): Onset roll.
        frame_roll (torch.Tensor): Frame roll.
        velocity_roll (torch.Tensor): Velocity roll.
        onset_thresh (float): Onset threshold.
        frame_thresh (float): Frame threshold.
    Returns:
        notes (numpy.ndarray): Notes.
        intervals (numpy.ndarray): Intervals of the notes.
        vels (numpy.ndarray): Velocities of the notes.
    Credits: https://github.com/jongwook/onsets-and-frames
    """
    # Get the onsets and frames    
    onsets = (onset_roll > onset_thresh).cpu().int()
    frames = (frame_roll > frame_thresh).cpu().int()
    # Due to frame resolution, we could have continuous onsets
    # if notes are sustained. That makes no sense, right?
    onset_events = (torch.cat([onsets[:1, :], onsets[1:, :] -  onsets[:-1, :]], \
                             dim = 0) == 1).nonzero()
    
    notes = []
    intervals = []
    vels = []

    for loc in onset_events:
        time = loc[0].item()
        note = loc[1].item()

        onset = time 
        offset = time
        velocities = []

        # As long as the note is on, we keep adding the velocities
        while onsets[offset, note].item() or frames[offset, note].item():
            if onsets[offset, note].item():
                # If the onset is on, we add the velocity
                velocities.append(velocity_roll[offset, note].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
                
        # If the note is off, we store results
        if offset > onset:
            notes.append(note)
            intervals.append([onset, offset])
            vels.append(np.mean(velocities) if len(velocities) > 0 else 0)
    
    return np.array(notes), np.array(intervals), np.array(vels)


def notes_to_frames(notes: np.ndarray, intervals: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert notes and intervals to a piano roll.
    Args:
        notes (np.ndarray): Notes.
        intervals (np.ndarray): Intervals of the notes.
        shape (tuple): Shape of the piano roll.
    Returns:
        roll (np.ndarray): Piano roll.
    Credits: https://github.com/jongwook/onsets-and-frames
    """
    # Create a piano roll of zeros
    roll = np.zeros(shape)
    for note, (onset, offset) in zip(notes, intervals):
        roll[onset:offset, note] = 1
    return roll

def notes_to_frames_vels(notes: np.ndarray, intervals: np.ndarray,
                         velocities: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert notes and intervals to a piano roll.
    Args:
        notes (np.ndarray): Notes.
        intervals (np.ndarray): Intervals of the notes.
        shape (tuple): Shape of the piano roll.
    Returns:
        roll (np.ndarray): Piano roll.
    Credits: https://github.com/jongwook/onsets-and-frames
    """
    # Create a piano roll of zeros
    roll = np.zeros(shape)
    for note, (onset, offset), velocity in zip(notes, intervals, velocities):
        roll[onset:offset, note] = velocity
    return roll

def events_to_roll(events: list[dict], shape: tuple, frame_rate: float) -> np.ndarray:
    roll = np.zeros(shape)
    
    if events is None:
        return roll
    
    for i, event in enumerate(events):
        onset_time = int(event['onset_time'] * frame_rate)
        offset_time = int(event['offset_time'] * frame_rate)
        midi_note = event['midi_note']
        velocity = event['velocity']

        # We will ignore velocity for now
        # convert time to frame index
        roll[onset_time:offset_time, midi_note] = 1
    return roll


def output_dict_to_events(output_dict, onset_threshold, offset_threshold,
                          frame_threshold, pedal_offset_threshold,
                          frames_per_second):
    """
        Convert the output probabilities of a transription model to events.

    Args:
        output_dict: dict, {
        'reg_onset_output': (frames_num, classes_num), 
        'reg_offset_output': (frames_num, classes_num), 
        'frame_output': (frames_num, classes_num), 
        'velocity_output': (frames_num, classes_num), 
        ...}

    Returns:
        est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
        offset_time, piano_note and velocity. E.g. [
            [39.74, 39.87, 27, 0.65], 
            [11.98, 12.11, 33, 0.69], 
            ...]
    """

    # ------ 1. Process regression outputs to binarized outputs ------
    # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
    # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

    # Calculate binarized onset output from regression output
    # using neighbour equals to 2 means we consider two frames on either side of our onset
    # Remember J is 5...it means our local maximum algorithm can detect two successive onsets
    # that are four frames apart which is 4/100 seconds which is 40ms. This is stated in the paper
    (onset_output, onset_shift_output) = \
        get_binarized_output_from_regression(
            reg_output=output_dict['reg_onset_output'], 
            threshold=onset_threshold, neighbour=2)

    output_dict['onset_output'] = onset_output  # Values are 0 or 1
    output_dict['onset_shift_output'] = onset_shift_output  

    # Calculate binarized offset output from regression output
    (offset_output, offset_shift_output) = \
        get_binarized_output_from_regression(
            reg_output=output_dict['reg_offset_output'], 
            threshold=offset_threshold, neighbour=4)

    output_dict['offset_output'] = offset_output  # Values are 0 or 1
    output_dict['offset_shift_output'] = offset_shift_output

    # ------ 2. Process matrices results to event results ------
    # Detect piano notes from output_dict
    est_on_off_note_vels = output_dict_to_detected_notes(output_dict, frame_threshold, frames_per_second)   
    return est_on_off_note_vels



def output_dict_to_pedals(output_dict, pedal_offset_threshold,
                          frames_per_second):
    """
        Convert the output probabilities of a transription model to events.

    Args:
        output_dict: dict, {
        'reg_onset_output': (frames_num, classes_num), 
        'reg_offset_output': (frames_num, classes_num), 
        'frame_output': (frames_num, classes_num), 
        'velocity_output': (frames_num, classes_num), 
        ...}

    Returns:
        est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
        and offset_time. E.g. [
            [0.17, 0.96], 
            [1.17, 2.65], 
            ...]
    """

    # ------ 1. Process regression outputs to binarized outputs ------
    # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
    # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

    # Calculate binarized onset output from regression output

    if 'reg_pedal_onset_output' in output_dict.keys():
        """Pedal onsets are not used in inference. Instead, frame-wise pedal
        predictions are used to detect onsets. We empirically found this is 
        more accurate to detect pedal onsets."""
        pass

    if 'reg_pedal_offset_output' in output_dict.keys():
        # Calculate binarized pedal offset output from regression output
        (pedal_offset_output, pedal_offset_shift_output) = \
            get_binarized_output_from_regression(
                reg_output=output_dict['reg_pedal_offset_output'], 
                threshold=pedal_offset_threshold, neighbour=4)

        output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
        output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

    # ------ 2. Process matrices results to event results ------
    if 'reg_pedal_onset_output' in output_dict.keys():
        # Detect piano pedals from output_dict
        est_pedal_on_offs = output_dict_to_detected_pedals(output_dict, frames_per_second)

    else:
        est_pedal_on_offs = None    

    return est_pedal_on_offs

def get_binarized_output_from_regression(reg_output, threshold, neighbour):
    """Calculate binarized output and shifts of onsets or offsets from the
    regression results.

    Args:
        reg_output: (frames_num, classes_num)
        threshold: float
        neighbour: int

    Returns:
        binary_output: (frames_num, classes_num)
        shift_output: (frames_num, classes_num)
    """
    binary_output = np.zeros_like(reg_output)
    shift_output = np.zeros_like(reg_output)
    (frames_num, classes_num) = reg_output.shape
    
    for k in range(classes_num):
        x = reg_output[:, k]
        for n in range(neighbour, frames_num - neighbour):
            if x[n] > threshold and is_monotonic_neighbour(x, n, neighbour):
                binary_output[n, k] = 1

                """See Section III-D in [1] for deduction.
                [1] Q. Kong, et al., High-resolution Piano Transcription 
                with Pedals by Regressing Onsets and Offsets Times, 2020."""
                if x[n - 1] > x[n + 1]:
                    # eq: 11: Code on repo had x[n+1] - x[n-1], why?????
                    shift = (x[n - 1] - x[n + 1]) / (x[n] - x[n + 1]) / 2
                else:
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                shift_output[n, k] = shift

    return binary_output, shift_output

def is_monotonic_neighbour(x, n, neighbour):
    """Detect if values are monotonic in both side of x[n].

    Args:
        x: (frames_num,)
        n: int
        neighbour: int

    Returns:
        monotonic: bool
    """
    monotonic = True
    for i in range(neighbour):
        if x[n - i] < x[n - i - 1]:
            monotonic = False
        if x[n + i] < x[n + i + 1]:
            monotonic = False

    return monotonic


def output_dict_to_detected_notes(output_dict, frame_threshold, frames_per_second):
    """Postprocess output_dict to piano notes.

    Args:
        output_dict: dict, e.g. {
        'onset_output': (frames_num, classes_num),
        'onset_shift_output': (frames_num, classes_num),
        'offset_output': (frames_num, classes_num),
        'offset_shift_output': (frames_num, classes_num),
        'frame_output': (frames_num, classes_num),
        'onset_output': (frames_num, classes_num),
        ...}

    Returns:
        est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
        MIDI notes and velocities. E.g.,
        [[39.7375, 39.7500, 27., 0.6638],
            [11.9824, 12.5000, 33., 0.6892],
            ...]
    """
    est_tuples = []
    est_midi_notes = []
    classes_num = output_dict['frame_output'].shape[-1]

    for piano_note in range(classes_num):
        """Detect piano notes"""
        est_tuples_per_note = note_detection_with_onset_offset_regress(
            frame_output=output_dict['frame_output'][:, piano_note], 
            onset_output=output_dict['onset_output'][:, piano_note], 
            onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
            offset_output=output_dict['offset_output'][:, piano_note], 
            offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
            velocity_output=output_dict['velocity_output'][:, piano_note], 
            frame_threshold=frame_threshold)
        
        est_tuples += est_tuples_per_note
        est_midi_notes += [piano_note] * len(est_tuples_per_note)

    if not est_tuples:
        return None
    
    est_tuples = np.array(est_tuples)   # (notes, 5)
    """(notes, 5), the five columns are onset, offset, onset_shift, 
    offset_shift and normalized_velocity"""

    est_midi_notes = np.array(est_midi_notes) # (notes,)

    onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / frames_per_second
    offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / frames_per_second
    velocities = est_tuples[:, 4]
    
    est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
    """(notes, 3), the three columns are onset_times, offset_times and velocity."""

    est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

    return est_on_off_note_vels

def output_dict_to_detected_pedals(output_dict, frames_per_second):
    """Postprocess output_dict to piano pedals.

    Args:
        output_dict: dict, e.g. {
        'pedal_frame_output': (frames_num,),
        'pedal_offset_output': (frames_num,),
        'pedal_offset_shift_output': (frames_num,),
        ...}

    Returns:
        est_on_off: (notes, 2), the two columns are pedal onsets and pedal
        offsets. E.g.,
            [[0.1800, 0.9669],
            [1.1400, 2.6458],
            ...]
    """
    frames_num = output_dict['pedal_frame_output'].shape[0]
    
    est_tuples = pedal_detection_with_onset_offset_regress(
        frame_output=output_dict['pedal_frame_output'][:, 0], 
        offset_output=output_dict['pedal_offset_output'][:, 0], 
        offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], 
        frame_threshold=0.5)

    est_tuples = np.array(est_tuples)
    """(notes, 2), the two columns are pedal onsets and pedal offsets"""
    
    if len(est_tuples) == 0:
        #return np.array([])
        return None

    else:
        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) /frames_per_second
        est_on_off = np.stack((onset_times, offset_times), axis=-1)
        est_on_off = est_on_off.astype(np.float32)
        return est_on_off
    

def note_detection_with_onset_offset_regress(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played."""
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    """bgn --------- offset_occur --- frame_disappear"""
                    fin = offset_occur
                else:
                    """bgn --- offset_occur --------- frame_disappear"""
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def pedal_detection_with_onset_offset_regress(frame_output, offset_output, 
    offset_shift_output, frame_threshold):
    """Process prediction array to pedal events information.
    
    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift], 
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533], 
        [1909, 1947, 0.30730522, -0.45764327], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples