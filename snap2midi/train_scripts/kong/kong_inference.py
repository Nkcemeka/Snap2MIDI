""" 
    File: kong_inference.py
    Author: Chukwuemeka L. Nkama
    Date: 6/9/2025
    Description: Inference script for Kong's model!
"""

# Import necessary libraries
import torch
import json
import argparse
import numpy as np
from snap2midi.utils.inference_utils import get_mel, stitch
from snap2midi.train_scripts.kong.kong import KongModel, KongPedal
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.train_scripts.kong.utilities import get_note_events, get_pedal_events
from collections import defaultdict
import pretty_midi

def load_kong(config: dict):
    # Load the necessary components from the config
    path = config["kong_checkpoint"]
    classes = config["classes"]
    cmp = config["cmp"]
    num_features = config["num_features"]
    momentum = config["momentum"]
    factors = config["factors"]

    # Initialize the Kong model with the loaded components
    model = KongModel(classes, num_features, momentum, cmp=cmp, factors=factors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model

def load_pedal(config: dict):

    # Load the necessary components from the config
    path = config["pedal_checkpoint"]
    num_features = config["num_features"]
    cmp = config["cmp"]
    momentum = config["momentum"]
    factors = config["factors"]

    # Initialize the KongPedal model with the loaded components
    model = KongPedal(num_features, momentum, cmp=cmp, factors=factors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model


def inference(audio_path: str, config: dict, feature_str: str, filename: str):
    window_size = config["window_size"]
    sr = config["sample_rate"]
    pr_rate = config["frame_rate"]

    # Frame the audio and init handcrafted features
    framed_audio = FramedAudio(audio_path=audio_path, hop_size=window_size//2, \
                               frame_size=window_size, sample_rate=sr)
    len_audio = framed_audio.len_audio
    hf = HandcraftedFeatures(sample_rate=sr, window_size=window_size, \
                             pr_rate=pr_rate)
    

    # Get the models
    kong_model = load_kong(config)
    pedal_model = load_pedal(config)
    result_dict: dict = defaultdict(list)

    for each in framed_audio:
        if feature_str == "mel":
            feature = get_mel(each, hf, config["num_features"])
        
        with torch.inference_mode():
            output_dict = kong_model(feature)
            output_dict_pedals = pedal_model(feature)

            for key in output_dict:
                result_dict[key].append(output_dict[key].cpu().detach().numpy())
            
            for key in output_dict_pedals:
                result_dict[key].append(output_dict_pedals[key].cpu().detach().numpy())

    for key in result_dict:
        result_dict[key] = np.concatenate(result_dict[key])
    
    # perform stitching
    for key in result_dict:
        result_dict[key] = stitch(result_dict[key])[:len_audio]

    # Get note and pedal events
    est_note_events = get_note_events(
            result_dict, config["onset_threshold"],
            config["offset_threshold"],
            config["frame_threshold"],
            frames_per_second=pr_rate
        )

    est_pedal_events = get_pedal_events(
        result_dict,
        config["pedal_threshold"],
        config["frame_threshold"],
        frames_per_second=pr_rate
    )

    # Save events to a MIDI file
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    if est_note_events is not None:
        for (onset, offset, pitch, vel) in est_note_events:

            note = pretty_midi.Note(
                velocity=int(vel*127), pitch=int(pitch), start=onset, end=offset
            )
            piano.notes.append(note)
    else:
        raise RuntimeError("No note events detected. Please check the model output and thresholds.")
    

    if est_pedal_events is not None:
        for (onset, offset) in est_pedal_events:
            sustain_on = pretty_midi.ControlChange(
                number=64, value=127, time=onset
            )
            sustain_off = pretty_midi.ControlChange(
                number=64, value=0, time=offset
            )
            piano.control_changes.append(sustain_on)
            piano.control_changes.append(sustain_off)
    else:
        raise RuntimeError("No pedal events detected. Please check the model output and thresholds.")
    
    midi_obj.instruments.append(piano)
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