"""
    File: utilities.py
    Author: Chukwuemeka L. Nkama
    Date: 6/10/2025
    Description: Utility functions for Kong model!
"""

# Import necessary libraries
import torch 
import numpy as np
from typing import Optional


def local_maxima(reg_roll: np.ndarray, frame: int, window: int) -> bool:
    """
        local_maxima checks if there is a local maximum
        at [reg_roll[n] (a triangle shape with x[n] as the peak)

        Args:
        -----
            reg_roll (np.ndarray): Regression roll of shape (num_frames) at a given pitch
            frame (int): frame position
            window (int): size of window to consider to left
                             and right of x(n)
        
        Returns:
        --------
            maxim (bool): True/False at x[n]
    """
    maxim = True
    for i in range(window):
        if (reg_roll[frame - i] < reg_roll[frame - i - 1]) or \
           (reg_roll[frame + i] < reg_roll[frame + i + 1]):
            maxim = False
    return maxim

def binarize(reg_roll: np.ndarray, thresh: float, window: int):
    """
        Calculates the binarized version of
        the regressed roll and returns it alongside
        its dist. from the nearest onset/offset
        etc.

        Args:
        ----
            reg_roll (np.ndarray): Regressed roll
            thresh (float): threshold
            window (int): window size
        
        Returns:
        --------
            bin_roll (np.ndarray): Binarized roll
            dist_roll (np.ndarray): Distance from nearest event
    """
    bin_roll = np.zeros(reg_roll.shape)
    dist_roll = np.zeros(reg_roll.shape)
    
    for pitch_class in range(reg_roll.shape[1]):
        reg_roll_pitch = reg_roll[:, pitch_class]
        for frame in range(window, reg_roll.shape[0] - window):
            if reg_roll_pitch[frame] > thresh and local_maxima(reg_roll_pitch, frame, window):
                bin_roll[frame, pitch_class] = 1

                if reg_roll_pitch[frame - 1] > reg_roll_pitch[frame + 1]:
                    # numerator should be the reverse but we do 
                    # this to allow the negative sign as the onset
                    # will be to the left of the frame center
                    num = reg_roll_pitch[frame + 1] - reg_roll_pitch[frame - 1]
                    den = reg_roll_pitch[frame] - reg_roll_pitch[frame + 1]
                else:
                    num = reg_roll_pitch[frame + 1] - reg_roll_pitch[frame - 1]
                    den = reg_roll_pitch[frame] - reg_roll_pitch[frame - 1]
                
                # distance between two adjacent frames is 1 so we have 0.5 instead
                # of dist_btw_frames/2
                dist = 0.5 * num/den
                dist_roll[frame, pitch_class] = dist
    return bin_roll, dist_roll


def note_detect_events(frame_arr: np.ndarray, onset_arr: np.ndarray, 
                onset_dist_arr: np.ndarray, offset_arr: np.ndarray, 
                offset_dist_arr: np.ndarray, vel_arr: np.ndarray, 
                frame_thresh: float) -> list:
    """
        Detect event occurences of a give note based on its frame, onset, offset, and velocity arr.

        Args:
        -----
            frame_arr (np.ndarray): Frame arr for a given pitch class
            onset_arr (np.ndarray): Onset arr for a given pitch class
            onset_dist_arr (np.ndarray): Onset distance arr for a given pitch class
            offset_arr (np.ndarray): Offset arr for a given pitch class
            offset_dist_arr (np.ndarray): Offset distance arr for a given pitch class
            vel_arr (np.ndarray): Velocity arr for a given pitch class
            frame_thresh (float): Frame threshold

        Returns:
        --------
            note_events (list): Detected events for a note: (onset, offset, onset_dist, offset_dist,
            velocity)
    """
    note_events: list = []
    frames = frame_arr.shape[0]
    on_frames = None # Onset time in frames
    off_active = None # Offset active at some frame
    frame_inactive = None # time in frames where the frame is inactive (No note there)

    for frame in range(frames):
        if onset_arr[frame] == 1:
            # This means an onset occurs at that frame
            if on_frames:
                # This means consecutive onsets
                # Hence our offset will be at the last frame
                # since we have a new onset
                off_frames = max(frame - 1, 0)
                off_dist = 0 # offset distance will be zero since we chopped it to last frame
                note_events.append([on_frames, off_frames, onset_dist_arr[on_frames], \
                                    off_dist, vel_arr[on_frames]])
            on_frames = frame

        if on_frames and frame > on_frames:
            # Since frame is > on_frames, it means
            # we have to get an offset
            # However, offset detection is usually ambiguous as
            # is mentioned in literature, so we depend on 
            # frame_inactive instead
            if frame_arr[frame] <= frame_thresh and not frame_inactive:
                frame_inactive = frame
            
            if offset_arr[frame] == 1 and not off_active:
                off_active =  frame
            
            if frame_inactive:
                # We depend on the frame_inactive to confirm the occur. 
                # of an offset
                if off_active and off_active - on_frames > frame_inactive - off_active:
                    off_frames = off_active
                else:
                    off_frames = frame_inactive
                note_events.append([on_frames, off_frames, onset_dist_arr[on_frames],\
                                    offset_dist_arr[off_frames], vel_arr[on_frames]])
                on_frames, frame_inactive, off_active = None, None, None

            if on_frames and (frame - on_frames >= 600 or frame == onset_arr.shape[0] - 1):
                # Produce a synthetic offset as note has been on for way too long!
                off_frames = frame
                note_events.append([on_frames, off_frames, onset_dist_arr[on_frames],\
                                    offset_dist_arr[off_frames], vel_arr[on_frames]])
                on_frames, frame_inactive, off_active = None, None, None

    # Sort the note events list by the onset itme
    note_events.sort(key=lambda event: event[0])
    return note_events



def pedal_detect_events(frame_arr: np.ndarray, offset_arr: np.ndarray, 
                offset_dist_arr: np.ndarray, frame_thresh: float) -> list:
    """
        Detect event occurences for the pedal based on its frame and offset arr.

        Args:
        -----
            frame_arr (np.ndarray): Frame arr for a given pitch class
            offset_arr (np.ndarray): Offset arr for a given pitch class
            offset_dist_arr (np.ndarray): Offset distance arr for a given pitch class
            frame_thresh (float): Frame threshold

        Returns:
        --------
            pedal_events (list): Detected pedal events: (onset, offset, onset_dist, offset_dist)
    """
    pedal_events: list = []
    frames = frame_arr.shape[0]
    on_frames = None # Onset time in frames
    off_active = None # Offset active at some frame
    frame_inactive = None # time in frames where the frame is inactive (No note there)

    for frame in range(1, frames):
        if frame_arr[frame] >= frame_thresh and frame_arr[frame] > frame_arr[frame - 1]:
            # Detection of pedal onset
            if not on_frames:
                on_frames = frame
        
        if on_frames and frame > on_frames:
            # Get the pedal offset
            if frame_arr[frame] <= frame_thresh and not frame_inactive:
                frame_inactive = frame
            
            if offset_arr[frame] == 1 and not off_active:
                off_active = frame
            
            if off_active:
                off_frames = off_active
                # note that onset_pedal_dist is set as 0 (see paper for reason)
                pedal_events.append([on_frames, off_frames, 0, offset_dist_arr[frame]])
                on_frames, frame_inactive, off_active = None, None, None
            
            if frame_inactive and frame - frame_inactive >= 10:
                # Frame has been inactive for a while
                off_frames = frame_inactive
                pedal_events.append([on_frames, off_frames, 0, offset_dist_arr[frame]])
                on_frames, frame_inactive, off_active = None, None, None

    # Sort the note events list by the onset itme
    pedal_events.sort(key=lambda event: event[0])
    return pedal_events

def get_note_events(model_output: dict, on_thresh: float, off_thresh: float, 
                    frame_thresh: float, frames_per_second: float) -> Optional[np.ndarray]:
    """
        get_note_events extracts the note events from the model output.

    Args:
    -----
        model_output (dict): Model output
        on_thresh (float): Threshold for onset detection.
        off_thresh (float): Threshold for offset detection.
        frame_thresh (float): Frame threshold for detecting active frames.
        frames_per_second (float): Frames per second of the model output.

    Returns:
    --------
        est_events (np.ndarray): Estimated events: [onset, offset, note, norm_vel]
    """

    # Binarize the onset regression roll
    bin_onset_roll, bin_onset_dist_roll = binarize(model_output['reg_onset_roll'],\
                                        on_thresh, window=2)
    model_output['onset_roll'] = bin_onset_roll
    model_output['onset_dist_roll'] = bin_onset_dist_roll

    # Binarize the offset regression roll
    bin_offset_roll, bin_offset_dist_roll = binarize(model_output['reg_offset_roll'],\
                                        off_thresh, window=2)
    model_output['offset_roll'] = bin_offset_roll
    model_output['offset_dist_roll'] = bin_offset_dist_roll

    # Get the events
    all_events: list = []
    all_notes: list = []

    for note in range(model_output['frame_roll'].shape[-1]):
        note_events = note_detect_events(
            model_output['frame_roll'][:, note],
            model_output['onset_roll'][:, note],
            model_output['onset_dist_roll'][:, note],
            model_output['offset_roll'][:, note],
            model_output['offset_dist_roll'][:, note],
            model_output['velocity_roll'][:, note],
            frame_thresh
        )
        all_events.extend(note_events)
        all_notes.extend([note] * len(note_events))
    
    if not all_events:
        return None

    events_arr = np.array(all_events) # columns: (onset, offset, onset_dist, offset_dist, velocity)
    notes = np.array(all_notes)

    # convert onset and offset frames to seconds (with the dist from the actual onset/offset)
    onset_seconds = (events_arr[:, 0] + events_arr[:, 2])/ frames_per_second
    offset_seconds = (events_arr[:, 1] + events_arr[:, 3]) / frames_per_second
    
    events = np.stack((onset_seconds, offset_seconds, \
                             notes, events_arr[:, 4]), axis=-1)
    events = events.astype(np.float32)
    return events



def get_pedal_events(model_output: dict, pedal_thresh: float, frame_thresh: float, 
                     frames_per_second: float) -> Optional[np.ndarray]:
    """
        get_pedal_events extracts the pedal events from the model output.

    Args:
    -----
        model_output (dict): Model output
        pedal_thresh (float): Threshold for pedal
        frames_per_second (float): Frames per second of the model output.

    Returns:
    --------
        est_pedal_events (np.ndarray): Estimated pedal events: [onset, offset]
    """

    # Binarize the pedal offset regression roll (Kong does not use the pedal
    # onset regression roll; it does not work)
    bin_pedal_off_roll, bin_pedal_off_dist_roll = binarize(model_output['reg_pedal_offset_roll'],\
                                        pedal_thresh, window=4)
    model_output['pedal_offset_roll'] = bin_pedal_off_roll
    model_output['pedal_offset_dist_roll'] = bin_pedal_off_dist_roll

    # Get the events
    all_pedal_events = pedal_detect_events(
            model_output['pedal_frame_roll'][:, 0],
            model_output['pedal_offset_roll'][:, 0],
            model_output['pedal_offset_dist_roll'][:, 0],
            frame_thresh # original code used 0.5 here, weird....
    )
    
    if not all_pedal_events:
        return None
    
    events_arr = np.array(all_pedal_events)

    # convert onset and offset frames to seconds (with the dist from the actual onset/offset)
    onset_seconds = (events_arr[:, 0] + events_arr[:, 2])/ frames_per_second
    offset_seconds = (events_arr[:, 1] + events_arr[:, 3]) / frames_per_second
    
    pedal_events = np.stack((onset_seconds, offset_seconds), axis=-1)
    pedal_events = pedal_events.astype(np.float32)
    return pedal_events