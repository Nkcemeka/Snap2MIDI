import torch
import numpy as np
from snap2midi.models.oafv2.oafv2 import OnsetsAndFramesV2
from snap2midi.utils.eval_mir import note_extract
import pretty_midi
import librosa
from nnAudio2.features.mel import MelSpectrogram

def load_oafv2(config: dict):
    """ 
        Load the onset and frames model.

        Args
        ----
            config (dict): Config dictionary
        
        Returns
        -------
            model (nn.Module): Onsets and Frames model.
    """
    # Load the necessary components from the config
    path = config["checkpoint_path"]
    model = OnsetsAndFramesV2(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    model.eval()
    return model

def inference(config: dict):
    """ 
        Perform inference

        Args
        ----
            config (dict): Config dictionary
        
        Returns
        -------
            midi_obj (pretty_midi.PrettyMIDI): PrettyMIDI object.
    """
    filename = config["filename"]
    audio_path = config["audio_path"]
    frame_rate = config["frame_rate"]
    threshold = config["threshold"]
    pitch_offset = config["pitch_offset"]
    sr = config["sample_rate"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = librosa.load(audio_path, sr=sr)
    onset_model = load_oafv2(config)
    mel = MelSpectrogram(
            sr=config["sample_rate"], n_fft=config["n_fft"], n_mels=config["n_mels"],\
            hop_length=config["hop_length"], htk=config["htk"], fmin=config["fmin"], \
            fmax=config["fmax"], pad_mode=config["pad_mode"], center=config["center"], \
            window=config["window"]
    ).to(device)

    with torch.inference_mode():
        audio = torch.from_numpy(audio).to(device)
        spec = mel(audio)
        spec = torch.log(torch.clamp(spec, min=1e-5)).transpose(-1, -2)
        on_preds, off_preds, _, frame_preds, vel_preds = onset_model(spec)
        on_preds = on_preds[0]
        frame_preds = frame_preds[0]
        vel_preds = vel_preds[0]

    note_preds, int_preds, vels = note_extract(on_preds, frame_preds, \
                                               vel_preds, onset_thresh=threshold, \
                                               frame_thresh=threshold)
    note_preds += pitch_offset

    # Save events to a MIDI file
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for each in range(len(note_preds)):
        pitch = note_preds[each]
        onset, offset = int_preds[each]/frame_rate
        vel = vels[each] * 127
        
        note = pretty_midi.Note(
                velocity=int(vel), pitch=int(pitch), start=onset, end=offset
        )
        piano.notes.append(note)
    
    midi_obj.instruments.append(piano)
    if filename is None:
        return midi_obj
    midi_obj.write(f'{filename}.mid')
