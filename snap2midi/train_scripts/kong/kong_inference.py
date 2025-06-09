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
from snap2midi.train_scripts.kong.kong import KongModel, KongPedal
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.utils.eval_mir import output_dict_to_pedals, output_dict_to_events
from collections import defaultdict
import pretty_midi

def load_kong(config: dict):
    # Load the necessary components from the config
    path = config["kong_checkpoint"]
    classes = config["classes"]
    cmp = config["cmp"]
    clue = config["clue"]
    momentum = config["momentum"]
    factors = config["factors"]

    # Initialize the Kong model with the loaded components
    model = KongModel(classes, clue, momentum, cmp=cmp, factors=factors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model

def load_pedal(config: dict):

    # Load the necessary components from the config
    path = config["pedal_checkpoint"]
    clue = config["clue"]
    cmp = config["cmp"]
    momentum = config["momentum"]
    factors = config["factors"]

    # Initialize the KongPedal model with the loaded components
    model = KongPedal(clue, momentum, cmp=cmp, factors=factors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model

def get_mel(audio_frame: np.ndarray, hf, n_mels):
    mel = hf.compute_mel(audio_frame, n_mels=n_mels)
    mel = torch.tensor(mel, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    mel = mel.T
    mel = mel.unsqueeze(0)
    return mel

def stitch(arr: np.ndarray):
    # Stitches the results from the Kong model
    # so that everything aligns
    arr = arr[:, :-1]
    result = []
    num_segments, num_frames, _ = arr.shape
    factor_75 = int(num_frames * 0.75)
    factor_25 = int(num_frames * 0.25)
    result.append(arr[0, :factor_75])
    for each in range(1, num_segments - 1):
        result.append(arr[each, factor_25 : factor_75])
    result.append(arr[-1, factor_25:])
    result = np.concatenate(result)
    return result


def inference(audio_path: str, config: dict, feature_str: str):
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
            feature = get_mel(each, hf, config["clue"])
        
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
    est_note_events = output_dict_to_events(
            result_dict, onset_threshold=config["onset_threshold"],
            offset_threshold=config["offset_threshold"],
            frame_threshold=config["frame_threshold"],
            pedal_offset_threshold=config["pedal_offset_threshold"],
            frames_per_second=pr_rate
        )

    est_pedal_events = output_dict_to_pedals(
        result_dict,
        pedal_offset_threshold=config["pedal_offset_threshold"],
        frames_per_second=pr_rate
    )

    # Save events to a MIDI file
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for (onset, offset, pitch, vel) in est_note_events:
        note = pretty_midi.Note(
            velocity=int(vel*127), pitch=int(pitch), start=onset, end=offset
        )
        piano.notes.append(note)
    
    for (onset, offset) in est_pedal_events:
        sustain_on = pretty_midi.ControlChange(
            number=64, value=127, time=onset
        )
        sustain_off = pretty_midi.ControlChange(
            number=64, value=0, time=offset
        )
        piano.control_changes.append(sustain_on)
        piano.control_changes.append(sustain_off)
    
    midi_obj.instruments.append(piano)
    midi_obj.write('final.mid')


if __name__ == "__main__":
    # Example config dictionary
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    #parser.add_argument("--audio_path", type=str, default=None, help="Path to the audio file")
    args = parser.parse_args()

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()
    
    # parse JSON file
    config = json.loads(content)
    audio_path = "/home/nkcemeka/Documents/snap/snap2midi/runs/ConvShallowTranscriber/results/snap-test.wav"
    audio_path = "/home/nkcemeka/Documents/snap/snap2midi/train_scripts/kong/Nanana-audio.mp3"
    audio_path = "/home/nkcemeka/Documents/snap/snap2midi/train_scripts/kong/hymn.mp3"
    feature_str = 'mel'  # or 'mfcc', etc.
    
    inference(audio_path, config, feature_str)
        




