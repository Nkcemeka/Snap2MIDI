"""
    File: utilities.py
    Author: Chukwuemeka L. Nkama
    Date: 7/14/2025
    Description: Utility script for HFT!
"""

# Imports
import torch
import numpy as np


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
    num_bins = config["input"]["num_bins"]
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
    input = torch.concatenate([pad_behind, feature, pad_forward], axis=0)

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
        output_velocity_1st[i:i+half_frames] = (output_velocity_1st_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_onset_2nd[i:i+half_frames] = (output_onset_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_offset_2nd[i:i+half_frames] = (output_offset_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_frames_2nd[i:i+half_frames] = (output_frames_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
        output_velocity_2nd[i:i+half_frames] = (output_velocity_2nd_i.squeeze(0))[shift:shift+half_frames].cpu().detach().numpy()
    
    return (output_onset_1st, output_offset_1st, output_frames_1st, output_velocity_1st,
            output_onset_2nd, output_offset_2nd, output_frames_2nd, output_velocity_2nd)


def frames_to_note(onset, offset, frames, velocity, threshold_onset, threshold_offset, threshold_frames):
    # note that onsets and offset are regression outputs and indicate the distance to the nearest note onset or offset
    # frames is a binary output indicating the presence of a note at that frame
    
    pass