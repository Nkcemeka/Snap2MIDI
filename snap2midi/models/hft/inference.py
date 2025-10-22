from .utilities import frames_to_note, notes_to_midi, half_stride
from .hft import *
import torch
import torchaudio
from .train_hft import init_weights

def load_hft(config):
    # Load/initialize the model
    encoder = HFTEncoder(n_margin=config['margin_b'],
                         n_frame=config['num_frame'],
                         n_bin=config['n_bins'],
                         cnn_channel=config["cnn_channel"],
                         cnn_kernel=config["cnn_kernel"],
                         d=config["d"],
                         n_layers=config["enc_layer"],
                         num_heads=config["enc_head"],
                         pff_dim=config["pff_dim"],
                         dropout=config["dropout"],
                         device="cuda" if torch.cuda.is_available() else "cpu")

    decoder = HFTDecoder(n_frame=config['num_frame'],
                         n_bin=config['n_bins'],
                         n_note=config['num_note'],
                         n_velocity=config['num_velocity'],
                         d=config["d"],
                         n_layers=config["dec_layer"],
                         num_heads=config["dec_head"],
                         pff_dim=config["pff_dim"],
                         dropout=config["dropout"],
                         device="cuda" if torch.cuda.is_available() else "cpu")
    
    model = HFT(encoder=encoder, decoder=decoder)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.apply(init_weights)
    checkpoint = torch.load(config["checkpoint_path"], map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

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
        resample = torchaudio.transforms.Resample(sr, config["sr"])
        audio = resample(audio)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sr"],
            n_fft=config["fft_bins"],
            hop_length=config["hop_sample"],
            win_length=config["window_length"],
            n_mels=config["mel_bins"],
            pad_mode=config["pad_mode"],
            norm="slaney"
        )
        feature = mel_transform(audio)
        feature = (torch.log(feature + config['log_offset'])).T
        return feature

def inference(config: dict):
    model = load_hft(config)
    shift = config["shift"]
    audio_file = config["audio_path"]
    feature = get_feature_hft(audio_file, config)
    output = half_stride(model, feature, shift, config)
    onset = output[-4]
    offset = output[-3]
    frames = output[-2]
    velocity = output[-1]
    
    # Convert regression roll to MIDI notes
    notes = frames_to_note(onset, offset, frames, velocity, config)

    # Convert notes to MIDI format
    midi_obj = notes_to_midi(notes)
    if config["filename"] is None:
         return midi_obj
    midi_obj.write(config["filename"]+".mid")
