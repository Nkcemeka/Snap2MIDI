import torch
import numpy as np
from snap2midi.utils.roll_to_midi import conv_to_midi
from .shallow_network import ShallowTranscriber
from snap2midi.utils.inference_utils import get_mel, stitch_tensor
from snap2midi.extractors.utils.framed_signal import FramedAudio
from snap2midi.extractors.utils.handcrafted_features import HandcraftedFeatures
import argparse
import json
from typing import Optional


def load_shallow(config: dict):
    # Load the necessary components from the config
    path = config["shallow_checkpoint"]
    in_features = config["in_features"]
    hidden_units = config["hidden_units"]
    out_features = config["out_features"]
    model = ShallowTranscriber(in_features=in_features, hidden_units=hidden_units,\
                               out_features=out_features)
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
    model = load_shallow(config)
    result: list = []

    for each in framed_audio:
        if feature_str == "mel":
            feature = get_mel(each, hf, config["in_features"])
        
        with torch.inference_mode():
            preds = model(feature)
            preds = torch.sigmoid(preds) > threshold
        
        result.append(preds)
        
    
    # concatenate the results
    result_concat = torch.concatenate(result)

    # Stitch the results
    result_stitch = stitch_tensor(result_concat)[:len_audio]

    # Save events to MIDI file
    midi_obj = conv_to_midi(result_stitch.cpu().detach().numpy()*127, 
                        "", pr_rate)
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