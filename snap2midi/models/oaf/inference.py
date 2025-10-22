# Import necessary libraries
import torch
import numpy as np
from snap2midi.utils.inference_utils import get_mel
from snap2midi.models.oaf.oaf import OnsetsAndFrames
from snap2midi.extractor.utils.handcrafted_features import HandcraftedFeatures
from snap2midi.utils.eval_mir import note_extract
import pretty_midi
import librosa

def load_oaf(config: dict):
    # Load the necessary components from the config
    path = config["checkpoint_path"]
    model = OnsetsAndFrames(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    model.eval()
    return model

def inference(config: dict):
    filename = config["filename"]
    audio_path = config["audio_path"]
    feature_str = config["feature_str"]
    window_size = config["window_size"]
    frame_rate = config["frame_rate"]
    threshold = config["threshold"]
    pitch_offset = config["pitch_offset"]
    sr = config["sample_rate"]

    audio, sr = librosa.load(audio_path, sr=sr)

    hf = HandcraftedFeatures(sample_rate=sr, window_size=window_size, \
                             frame_rate=frame_rate)
    
    # Get the models
    onset_model = load_oaf(config)

    if feature_str == "mel":
        feature = get_mel(audio, hf, config["in_features"], config["mel_n_fft"], config["hop_length"])

    with torch.inference_mode():
        #on_preds, _, _, frame_preds, vel_preds = onset_model(feature)
        on_preds, _, frame_preds, vel_preds = onset_model(feature)
        on_preds = torch.sigmoid(on_preds)[0]
        frame_preds = torch.sigmoid(frame_preds)[0]
        vel_preds = vel_preds[0]
    

    note_preds, int_preds, vels = note_extract(on_preds, frame_preds, \
                                               vel_preds, onset_thresh=threshold, \
                                               frame_thresh=threshold)
    # clamp velocities to [0, 1] using numpy
    vels = np.clip(vels, 0, 1)
    vels = 80*vels + 10 # This was recommended in the paper
    note_preds += pitch_offset

    # Save events to a MIDI file
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for each in range(len(note_preds)):
        pitch = note_preds[each]
        onset, offset = int_preds[each]/frame_rate
        vel = vels[each]
        
        note = pretty_midi.Note(
                velocity=int(vel), pitch=int(pitch), start=onset, end=offset
        )
        piano.notes.append(note)
    
    midi_obj.instruments.append(piano)
    if filename is None:
        return midi_obj
    midi_obj.write(f'{filename}.mid')
