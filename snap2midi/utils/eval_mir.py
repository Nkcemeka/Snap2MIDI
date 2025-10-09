# Imports
import numpy as np
import mir_eval
import torch

# Function that takes a piano roll and gets the
# ref and est intervals and pitches
def get_intervals_and_pitches(piano_roll: np.ndarray, \
                              frame_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
        Get the intervals and pitches from a piano roll.
        
        Args:
        ------
            piano_roll (np.ndarray): Piano roll.
            frame_rate (float): Frames per second

        Returns:
        -------
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

# Function to calculate transcription metrics from piano roll
def transcription_metrics_roll(pred: np.ndarray, gt: np.ndarray, \
                          frame_rate: float) -> dict | None:
    """
        Calculate transcription metrics using mir_eval.

        Args:
        -----
            pred (np.ndarray): Predicted piano roll.
            gt (np.ndarray): Ground truth piano roll.
            frame_rate (float): Frames per second

        Returns:
        --------
            scores (dict): Dictionary containing the calculated metrics.
    """
    
    # Get the ref and est intervals and pitches
    ref_i, ref_p = get_intervals_and_pitches(gt, frame_rate)
    est_i, est_p = get_intervals_and_pitches(pred, frame_rate)

    if len(ref_p) == 0:
        return None

    scores = mir_eval.transcription.evaluate(
        ref_intervals=ref_i, ref_pitches=ref_p, \
        est_intervals=est_i, est_pitches=est_p)
    return scores

# Function to calculate transcription metrics from notes and intervals
def transcription_metrics(note_arr_pred: np.ndarray, int_arr_pred: np.ndarray, 
                          note_arr_gt: np.ndarray, int_arr_gt: np.ndarray, frame_rate: float):
    """
        Calculate transcription metrics using mir_eval.

        Args:
        -----
            note_arr_pred (np.ndarray): Array of predicted notes.
            int_arr_pred (np.ndarray): Array of predicted intervals.
            note_arr_gt (np.ndarray): Array of ground truth notes.
            int_arr_gt (np.ndarray): Array of ground truth intervals.
            frame_rate (float): Frames per second

        Returns:
        --------
            scores (dict): Dictionary containing the calculated metrics.
    """

    # Check if the ground truth notes are empty
    if len(note_arr_gt) == 0:
        return None
    
    if int_arr_pred.shape[0] == 0:
        return None
    
    # convert the intervals to seconds
    int_arr_gt = (int_arr_gt / frame_rate) 
    int_arr_pred = (int_arr_pred / frame_rate)

    # convert the pitches to hz
    note_arr_gt = 440 * (2 ** ((note_arr_gt - 69) / 12))
    note_arr_pred = 440 * (2 ** ((note_arr_pred - 69) / 12))

    scores = mir_eval.transcription.evaluate(
        ref_intervals=int_arr_gt, ref_pitches=note_arr_gt, \
        est_intervals=int_arr_pred, est_pitches=note_arr_pred)

    return scores

# Function to calculate frame-level metrics from piano roll
def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
        Calculate frame-level metrics.

        Args:
        ----
            pred (np.ndarray): Predicted frame-level events.
            gt (np.ndarray): Ground truth frame-level events.

        Returns:
        --------
            dict: Dictionary containing the calculated metrics.
    """
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

# Function to get multipitch intervals and pitches from a piano roll
def get_multipitch_intervals_and_pitches(piano_roll: np.ndarray, \
            frame_rate: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """
        Get the multipitch intervals and pitches from a piano roll.

        Args:
        ------
            piano_roll (np.ndarray): Piano roll.
            frame_rate (float): Frames per second

        Returns:
        -------
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

# Function to calculate multipitch metrics from piano roll
def multipitch_metrics(ref_roll: np.ndarray, est_roll: np.ndarray, \
                       frame_rate: float, pitch_offset: int = 21) -> dict:
    """
        Calculate multipitch metrics using mir_eval.

        Args:
        ------
            ref_roll (np.ndarray): Reference multipitch roll.
            est_roll (np.ndarray): Estimated multipitch roll.
            frame_rate (float): Frames per second
            pitch_offset (int): Offset for the pitch values, default is 21.

        Returns:
        -------
            scores (dict): Dictionary containing the calculated metrics.
    """
    time_ref = np.arange(ref_roll.shape[0]) / frame_rate
    time_est = np.arange(est_roll.shape[0]) / frame_rate

    ref_freqs = [np.nonzero(ref_roll[t, :])[0] for t in range(ref_roll.shape[0])]
    est_freqs = [np.nonzero(est_roll[t, :])[0] for t in range(est_roll.shape[0])]

    # Convert frequencies to Hz
    ref_freqs = [np.array([mir_eval.util.midi_to_hz(p+pitch_offset) for p in freqs]) for freqs in ref_freqs]
    est_freqs = [np.array([mir_eval.util.midi_to_hz(p+pitch_offset) for p in freqs]) for freqs in est_freqs]

    
    scores = mir_eval.multipitch.evaluate(
        ref_time=time_ref, ref_freqs=ref_freqs, \
        est_time=time_est, est_freqs=est_freqs
    )

    return scores

# Function to calculate multipitch metrics
def multipitch_metrics_roll(pred: np.ndarray, gt: np.ndarray, \
                       frame_rate: float) -> dict:
    """
        Calculate multipitch metrics using mir_eval.

        Args:
        ------
            pred (np.ndarray): Predicted multipitch.
            gt (np.ndarray): Ground truth multipitch.
            frame_rate (float): Frames per second

        Returns:
        -------
            scores: Dictionary containing the calculated metrics.
    """
    
    # Get the ref time and pitches
    ref_time, ref_freqs = get_multipitch_intervals_and_pitches(gt, frame_rate)
    est_time, est_freqs = get_multipitch_intervals_and_pitches(pred, frame_rate)
    
    scores = mir_eval.multipitch.evaluate(
        ref_time=ref_time, ref_freqs=ref_freqs, \
        est_time=est_time, est_freqs=est_freqs
    )
    return scores

# Function to extract notes from piano roll
def note_extract(onset_roll: torch.Tensor, frame_roll: torch.Tensor, \
                 velocity_roll: torch.Tensor, onset_thresh: float=0.5, \
                 frame_thresh: float=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Extract notes from the piano roll.

        Args:
        ------
            onset_roll (torch.Tensor): Onset roll.
            frame_roll (torch.Tensor): Frame roll.
            velocity_roll (torch.Tensor): Velocity roll.
            onset_thresh (float): Onset threshold.
            frame_thresh (float): Frame threshold.

        Returns:
        --------
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
        ------
            notes (np.ndarray): Notes.
            intervals (np.ndarray): Intervals of the notes.
            shape (tuple): Shape of the piano roll.

        Returns:
        --------
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
        -------
        notes (np.ndarray): Notes.
        intervals (np.ndarray): Intervals of the notes.
        shape (tuple): Shape of the piano roll.
    
        Returns:
        --------
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
