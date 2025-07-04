""" 
    File: onset_inference.py
    Author: Chukwuemeka L. Nkama
    Date: 6/13/2025
    Description: Inference script for Onsets and Frames model!
"""

# Import necessary libraries
import torch
import json
import argparse
import numpy as np
from snap2midi.utils.inference_utils import get_mel, stitch_tensor
from snap2midi.train_scripts.onsets_and_frames.onsets_and_frames import OnsetsAndFrames
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.utils.eval_mir import note_extract
from collections import defaultdict
from typing import Optional
import pretty_midi

def load_onset(config: dict):
    # Load the necessary components from the config
    path = config["onset_checkpoint"]
    in_features = config["in_features"]
    out_features = config["out_features"]
    factor = config["factor"]
    complexity = config["model_complexity"]
    model = OnsetsAndFrames(input_features=in_features, output_features=out_features, 
                            factor=factor, model_complexity=complexity)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model

def inference(audio_path: str, config: dict, feature_str: str, filename: Optional[str]):
    window_size = config["window_size"]
    sr = config["sample_rate"]
    pr_rate = config["frame_rate"]
    threshold = config["threshold"]

    # Frame the audio and init handcrafted features
    framed_audio = FramedAudio(audio_path=audio_path, hop_size=window_size//2, \
                               frame_size=window_size, sample_rate=sr)
    len_audio = framed_audio.len_audio
    hf = HandcraftedFeatures(sample_rate=sr, window_size=window_size, \
                             pr_rate=pr_rate)
    

    # Get the models
    onset_model = load_onset(config)
    onset_result: list = []
    frame_result: list = []
    vel_result: list = []

    for each in framed_audio:
        if feature_str == "mel":
            feature = get_mel(each, hf, config["in_features"])
        
        with torch.inference_mode():
            on_preds, off_preds, _, frame_preds, vel_preds = onset_model(feature)
            on_preds = torch.sigmoid(on_preds)
            frame_preds = torch.sigmoid(frame_preds)
        
        onset_result.append(on_preds)
        frame_result.append(frame_preds)
        vel_result.append(vel_preds)
    
    # concatenate the results
    onset_result_concat = torch.concatenate(onset_result)
    frame_result_concat = torch.concatenate(frame_result)
    vel_result_concat = torch.concatenate(vel_result)

    # Stitch the results
    onset_result_stitch = stitch_tensor(onset_result_concat)[:len_audio]
    frame_result_stitch = stitch_tensor(frame_result_concat)[:len_audio]
    vel_result_stitch = stitch_tensor(vel_result_concat)[:len_audio]

    note_preds, int_preds, vels = note_extract(onset_result_stitch, frame_result_stitch, \
                                               vel_result_stitch, onset_thresh=threshold, \
                                               frame_thresh=threshold)

    # Save events to a MIDI file
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for each in range(len(note_preds)):
        pitch = note_preds[each]
        onset, offset = int_preds[each]/pr_rate
        vel = vels[each]
        
        note = pretty_midi.Note(
                velocity=int(vel*127), pitch=int(pitch), start=onset, end=offset
        )
        piano.notes.append(note)
    
    midi_obj.instruments.append(piano)
    if filename is None:
        return midi_obj
    midi_obj.write(f'{filename}.mid')

if __name__ == "__main__":
    # Example config dictionary
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    parser.add_argument("--name", type=str, default="transcription", help="Name of the output file")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to the audio file")
    parser.add_argument("--feature_str", type=str, default="mel", help="Feature type to use (e.g., 'mel', 'mfcc')")
    args = parser.parse_args()

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()
    
    # parse JSON file
    config = json.loads(content)
    inference(args.audio_path, config, args.feature_str, args.name)