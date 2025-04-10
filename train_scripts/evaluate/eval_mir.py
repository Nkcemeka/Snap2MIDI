"""
File: eval_mir.py
Author: Chukwuemeka L. Nkama
Date: 4/9/2025
Description: MIR_Eval Evaluation Script for transcription!
"""

# Imports
import numpy as np
import mir_eval

# Function that takes a piano roll and gets the
# ref and est intervals and pitches
def get_intervals_and_pitches(piano_roll, frame_rate):
    """
    Get the intervals and pitches from a piano roll.
    Args:
        piano_roll (numpy.ndarray): Piano roll.
        frame_rate (int): Frames per second
    Returns:
        intervals (numpy.ndarray): Onset and offset events.
        pitches (numpy.ndarray): Pitches of events.
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
    intervals = intervals / frame_rate

    # Convert pitches to Hz
    pitches = 440 * (2 ** ((pitches - 69) / 12))

    return intervals, pitches

# Function to calculate transcription metrics
def transcription_metrics(pred, gt, frame_rate=None):
    """
    Calculate transcription metrics using mir_eval.
    Args:
        pred (numpy.ndarray): Predicted piano roll.
        gt (numpy.ndarray): Ground truth piano roll.
        frame_rate (int): Frames per second
    Returns:
        scores: Dictionary containing the calculated metrics.
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

# Function to get multipitch intervals and pitches
def get_multipitch_intervals_and_pitches(piano_roll, frame_rate):
    """
    Get the multipitch intervals and pitches from a piano roll.
    Args:
        piano_roll (numpy.ndarray): Piano roll.
        frame_rate (int): Frames per second
    Returns:
        times (numpy.ndarray): Times of the active notes.
        pitches (numpy.ndarray): Pitches of the active notes.
    """
    times = np.arange(piano_roll.shape[0])
    pitches = [piano_roll.astype(int)[t, :].nonzero()[0] for t in times]

    # Convert times to seconds
    times = times / frame_rate
    # Convert pitches to Hz
    pitches = [440 * (2 ** ((p - 69) / 12)) for p in pitches]
    return times, pitches   

# Function to calculate multipitch metrics
def multipitch_metrics(pred, gt, frame_rate=None):
    """
    Calculate multipitch metrics using mir_eval.
    Args:
        pred (numpy.ndarray): Predicted multipitch.
        gt (numpy.ndarray): Ground truth multipitch.
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