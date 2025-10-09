import torch
import numpy as np
from .utilities import get_note_events, get_pedal_events, \
    load_kong, load_pedal, stitch
import librosa
import pretty_midi
from collections import defaultdict

def events(model, audio, sample_rate, window_size, device, frame_rate,\
                  on_thresh, off_thresh, pedal_thresh, frame_thresh, pedal_flag=False):
    result_dict: dict = defaultdict(list)
    window_samples = int(window_size * sample_rate)
    hop_samples = window_samples//2
    for start in range(0, len(audio), hop_samples):
        end = start + window_samples
        audio_segment = audio[start:end]

        if len(audio_segment) < window_samples:
            padding = window_samples - len(audio_segment)
            audio_segment = np.pad(audio_segment, (0, padding))

        # perform inference
        output_dict = model(torch.tensor(audio_segment.reshape(1, -1)).to(device))

        for key in output_dict.keys():
            result_dict[key].append(output_dict[key].cpu().detach().numpy())
    
    for key in result_dict:
        result_dict[key] = np.concatenate(result_dict[key])
    
    # perform stitching
    for key in result_dict:
        result_dict[key] = stitch(result_dict[key])[:len(audio)]

    # Get note events    
    if not pedal_flag:
        events = get_note_events(result_dict, on_thresh, off_thresh, \
                                       frame_thresh, frame_rate)
    else:
        events = get_pedal_events(result_dict, pedal_thresh, frame_thresh, frame_rate)
    return events

@torch.no_grad()
def inference(config: dict):
    note_model = load_kong(config)
    pedal_model = load_pedal(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    on_thresh = config["onset_threshold"]
    off_thresh = config["offset_threshold"]
    frame_thresh = config["frame_threshold"]
    frame_thresh = config["frame_threshold"]
    pedal_thresh = config["pedal_offset_threshold"]
    audio_file = config["audio_path"]
    sample_rate = config["sample_rate"]
    window_size = config["window_size"]
    frame_rate = config["frame_rate"]
    output_file = config["filename"]
    min_pitch = config["min_pitch"]

    # load the audio file
    audio = librosa.load(str(audio_file), sr=sample_rate, mono=True)[0]

    note_events = events(note_model, audio, sample_rate, window_size, device, frame_rate,
                            on_thresh, off_thresh, pedal_thresh, frame_thresh, pedal_flag=False)
    pedal_events = events(pedal_model, audio, sample_rate, window_size, device, frame_rate,
                            on_thresh, off_thresh, pedal_thresh, frame_thresh, pedal_flag=True)

    locs_invalid_note_events = np.where(note_events[:, 1] < note_events[:, 0])[0]
    locs_invalid_pedal_events = np.where(pedal_events[:, 1] < pedal_events[:, 0])[0]

    if len(locs_invalid_note_events) > 0:
        print(f"Warning: Found {len(locs_invalid_note_events)} invalid note events. Deleting them.")
        note_events = np.delete(note_events, locs_invalid_note_events, axis=0)

    if len(locs_invalid_pedal_events) > 0:
        print(f"Warning: Found {len(locs_invalid_pedal_events)} invalid pedal events. Deleting them.")
        pedal_events = np.delete(pedal_events, locs_invalid_pedal_events, axis=0)

    # Create a PrettyMIDI object
    midi_obj = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for each in range(len(note_events)):
        pitch = note_events[each, 2] + min_pitch
        onset, offset = note_events[each, 0], note_events[each, 1]
        vel = note_events[each, 3]

        # Create a Note object and add it to the piano instrument
        note = pretty_midi.Note(velocity=int(vel*128), pitch=int(pitch), start=onset, end=offset)
        piano.notes.append(note)

    # Add the pedal events
    for each in range(len(pedal_events)):
        onset, offset = pedal_events[each, 0], pedal_events[each, 1]

        # Create a ControlChange object and add it to the piano instrument
        cc = pretty_midi.ControlChange(number=64, value=127, time=onset)
        piano.control_changes.append(cc)
        cc = pretty_midi.ControlChange(number=64, value=0, time=offset)
        piano.control_changes.append(cc)

    midi_obj.instruments.append(piano)
    if output_file is None:
        return midi_obj
    midi_obj.write(output_file + ".mid")
