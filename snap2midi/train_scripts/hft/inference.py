from snap2midi.train_scripts.hft.utilities import frames_to_note, notes_to_midi, half_stride, transcription_metrics
from snap2midi.train_scripts.hft.hft import *
import torch.nn as nn
import torch
import json
import torchaudio
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_feature_hft(audio_file: str, config: dict) -> torch.Tensor:
        """
            Get the feature for the audio file for the hFT-Transformer model by Sony.

            Args:
                audio_file (str): Path to the audio file
                config (dict): Configuration dictionary containing the parameters
            Returns:
                feature (torch.Tensor): Feature for the audio file
        """
        # Get the feature for the audio file
        # we use torchaudio to speed this up; librosa is too slow
        audio, sr = torchaudio.load(audio_file)
        audio = torch.mean(audio, dim=0)
        resample = torchaudio.transforms.Resample(sr, config["feature"]["sr"])
        audio = resample(audio)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["feature"]["sr"],
            n_fft=config["feature"]["fft_bins"],
            hop_length=config["feature"]["hop_sample"],
            win_length=config["feature"]["window_length"],
            n_mels=config["feature"]["mel_bins"],
            pad_mode=config["feature"]["pad_mode"],
            norm="slaney"
        )
        feature = mel_transform(audio)
        feature = (torch.log(feature + config['feature']['log_offset'])).T
        return feature

def init_weights(m):
    """
        Initialize weights of the model using Xavier uniform initialization.

        Args:
            m (torch.nn.Module): The module to initialize weights for.
    """
    if hasattr(m, 'weight') and (m.weight.dim() > 1):
        nn.init.xavier_uniform_(m.weight.data)

def load_config(config_path):
    """
    Load the configuration from the JSON file.

    Returns:
        dict: Configuration dictionary containing parameters for the model.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

train_config_path = "./confs/train_hft_config.json"
train_config = load_config(train_config_path)
config = load_config(train_config["paths"][1])

encoder = HFTEncoder(n_margin=config['input']['margin_b'],
                        n_frame=config['input']['num_frame'],
                        n_bin=config['feature']['n_bins'],
                        cnn_channel=train_config["cnn_channel"],
                        cnn_kernel= train_config["cnn_kernel"],
                        d=train_config["d"],
                        n_layers=train_config["enc_layer"],
                        num_heads=train_config["enc_head"],
                        pff_dim= train_config["pff_dim"],
                        dropout=train_config["dropout"],
                        device="cuda" if torch.cuda.is_available() else "cpu")

decoder = HFTDecoder(n_frame=config['input']['num_frame'],
                        n_bin=config['feature']['n_bins'],
                        n_note=config['midi']['num_note'],
                        n_velocity=config['midi']['num_velocity'],
                        d=train_config["d"],
                        n_layers=train_config["dec_layer"],
                        num_heads=train_config["dec_head"],
                        pff_dim= train_config["pff_dim"],
                        dropout=train_config["dropout"],
                        device="cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = train_config["save_dir"]
model = HFT(encoder=encoder, decoder=decoder)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.apply(init_weights)  # Initialize weights
checkpoint = torch.load(train_config["resume_path"], map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint['model_state_dict'])

def inference(model, audio_file, config: dict, shift=32):
    feature = get_feature_hft(audio_file, config)
    output = half_stride(model, feature, shift, config)
    onset = output[-4]
    offset = output[-3]
    frames = output[-2]
    velocity = output[-1]
    
    # Convert regression roll to MIDI notes
    notes = frames_to_note(onset, offset, frames, velocity, 
                           0.5, 0.5, 0.5, config)

    # Convert notes to MIDI format
    midi = notes_to_midi(notes, "happy")
    return midi


def test_set_metrics(model: nn.Module, test_dir: str, config: dict):
    """
        Calculate the transcription metrics for a test set.

        Args:
            test_dir (str): Directory containing the test set files.

        Returns:
            scores (dict): Dictionary containing the transcription metrics.
    """
    test_files = sorted(Path(test_dir).glob("*.npz"))
    trans_metrics = defaultdict(list)

    for file in tqdm(test_files, total=len(test_files)-1):
        if "dataset_feature.npz" in str(file):
            continue

        data = np.load(file, allow_pickle=True)
        feature = data['feature']
        ref_notes = data['notes'] # dictionary of note events
        ref_frames = data['label_frames'].astype(int) # binary array of shape [num_frames, num_pitches]

        # Get the model output
        output = half_stride(model, feature, shift=32, config=config)
        onset = output[-4]
        offset = output[-3]
        frames = output[-2]
        velocity = output[-1]   

        # Convert regression roll to MIDI notes
        est_notes = frames_to_note(onset, offset, frames, velocity, 
                           0.5, 0.5, 0.5, config)


        for key, value in transcription_metrics(est_notes, ref_notes).items():
            trans_metrics[key].append(value)

    # Calculate the average scores
    scores = {key: np.mean(value) for key, value in trans_metrics.items()}
    print("Transcription Metrics:"  , scores)
    return scores


#inference(model, "./happy-know.wav", config)
test_set_metrics(model, "../../extractors/maps_segments/feature/test", config)

