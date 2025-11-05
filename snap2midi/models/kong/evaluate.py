import h5py
import numpy as np
import torch
from tqdm import tqdm
import glob
import pretty_midi
from collections import defaultdict
from .utilities import get_note_events, get_pedal_events, \
    extend_pedal, load_extract_config, load_kong, load_pedal, stitch
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel
from mir_eval.util import midi_to_hz
from sklearn.metrics import precision_recall_fscore_support as prfs
import pprint

def get_midi_note_events(midi, start: float, end: float) -> list:
        note_events = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 

                    pitch = note.pitch

                    # Need to deal with the case where the note starts before
                    # the start of the segment and ends after the end of the segment
                    # Not sure if what I have done below is the right thing to do
                    note_events.append([note.start, note.end, pitch, note.velocity])
        
        return np.array(note_events)



def get_pedal_frames(midi: pretty_midi.PrettyMIDI, \
                  start: float, end: float, frame_rate: int) -> np.ndarray:

        duration = end - start
        num_frames = int(round(duration * frame_rate)) + 1

        # initialize the pedal frames 
        pedal_frames = np.zeros((num_frames,))

        CC_SUSTAIN_PEDAL = 64
        is_pedal_on = False
        
        pedal_events: list[list] = [] # contains [onset_time, offset_time]

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for cc in [_e for _e in instrument.control_changes
                           if _e.number == CC_SUSTAIN_PEDAL]:
                    
                    if cc.time >= end or cc.time < start:
                        continue

                    frame_now = int(round((cc.time - start) * frame_rate))
                    time_now = cc.time - start
                    is_current_pedal_on = (cc.value >= CC_SUSTAIN_PEDAL)

                    if not is_pedal_on and is_current_pedal_on:
                        frame_pedal_on = frame_now
                        is_pedal_on = True
                        onset_time = time_now
                        abs_onset_time = cc.time
                    elif is_pedal_on and not is_current_pedal_on:
                        # store the pedal information
                        # We add +1 due to python's indexing. Also, see num_frames above;
                        # + 1 was added to catch events that fall exactly at the right
                        # edge of the last frame (which can happen)
                        pedal_frames[frame_pedal_on:frame_now + 1] = 1
                        is_pedal_on = False
                        pedal_events.append([abs_onset_time, cc.time])
            
        if is_pedal_on:
            pedal_events.append([abs_onset_time, end])

            # calc. frame_pedal_on
            frame_pedal_on = int(round((onset_time) * frame_rate))
            
            if frame_pedal_on >= 0 and frame_pedal_on < num_frames:
                pedal_frames[frame_pedal_on:] = 1
        
        return pedal_frames, np.array(pedal_events)

def get_frames(midi: pretty_midi.PrettyMIDI, \
                  start: float, end: float, frame_rate: int, max_pitch: int, min_pitch: int) -> np.ndarray:

        duration = end - start
        num_frames = int(round(duration * frame_rate)) + 1
        num_pitches = max_pitch - min_pitch + 1

        # initialize the labels
        label_frames = np.zeros((num_frames, num_pitches), dtype=bool)

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 

                    pitch = note.pitch - min_pitch
                    
                    start_frame = int(round(frame_rate * (note.start - start)))
                    end_frame = int(round(frame_rate * (note.end - start)))

                    # prepare labels
                    label_frames[max(0, start_frame):min(end_frame + 1, num_frames), pitch] = 1
        return label_frames

@torch.no_grad()
def evaluate(config: dict):
    # glob all the h5 files in the test directory
    test_files = glob.glob("data/kong/test/*.h5")
    dataset_length = len(test_files)

    model = load_kong(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extraction_config = load_extract_config()
    frame_rate = extraction_config["frame_rate"]
    min_pitch = extraction_config["min_pitch"]
    max_pitch = extraction_config["max_pitch"]
    on_thresh = config["onset_threshold"]
    off_thresh = config["offset_threshold"]
    frame_thresh = config["frame_threshold"]
    frame_thresh = config["frame_threshold"]
    frame_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    note_metrics = {'note_Precision': [], 'note_Recall': [], 'note_F1': [], 'note_Precision_no_offset': [], \
                    'note_Recall_no_offset': [], 'note_F1_no_offset': [], 'note_vel_Precision': [], 'note_vel_Recall': [],\
                        'note_vel_F1': []}

    for i in tqdm(range(dataset_length), total=dataset_length, desc="Extracting results...."):
        file = test_files[i]
        with h5py.File(file, 'r') as hf:
            audio = hf["audio"][:]
            midi_path = hf.attrs["midi_path"]

            # convert audio to float32
            audio = (audio / 32767.0).astype(np.float32)
        
        window_size = extraction_config["window_size"]

        # load midi
        # sustain MIDI
        #midi = pedal_extend(midi_path)
        if extraction_config["extend_pedal"].item():
            midi = extend_pedal(pretty_midi.PrettyMIDI(midi_path))
        else:
            midi = pretty_midi.PrettyMIDI(midi_path)
        result_dict: dict = defaultdict(list)
        window_samples = int(window_size * extraction_config["sample_rate"])
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
        ref_note_events = get_midi_note_events(midi, 0, len(audio)/extraction_config["sample_rate"])        
        est_note_events = get_note_events(result_dict, on_thresh, off_thresh, \
                                          frame_thresh, frame_rate)
        
        # ensure that offset is >= onset times
        # Get locations of invalid estimated note events where offset < onset
        locs = np.where(est_note_events[:, 1] < est_note_events[:, 0])[0]

        if len(locs) > 0:
            invalid_events = est_note_events[locs]
            print(f"Invalid estimated note events where offset < onset: {invalid_events}")
            # Remove the invalid events
            est_note_events = np.delete(est_note_events, locs, axis=0)
            
        # Assert that there are no invalid events left
        assert np.all(est_note_events[:, 1] >= est_note_events[:, 0]), \
            f"Estimated note events have invalid timings.{est_note_events}"

        est_notes = est_note_events[:, 2]
        est_ints = est_note_events[:, :2]
        est_vels = est_note_events[:, 3]*128
        ref_notes = ref_note_events[:, 2]
        ref_ints = ref_note_events[:, :2]
        ref_vels = ref_note_events[:, 3]

        # update pitch of est_notes
        est_notes += min_pitch

        # convert pitches to Hz
        est_notes = midi_to_hz(est_notes)
        ref_notes = midi_to_hz(ref_notes)

        # Get the note-level metrics
        score_notes_off = prf(ref_ints, ref_notes, est_ints, est_notes)
        notes_precision, notes_recall = score_notes_off[0], score_notes_off[1]
        note_f1 = score_notes_off[2]
        
        score_notes_no_off = prf(ref_ints, ref_notes, est_ints, est_notes, offset_ratio=None)
        notes_prec_no_off, notes_recall_no_off = score_notes_no_off[0], score_notes_no_off[1]
        note_f1_no_off = score_notes_no_off[2]

        score_notes_vel = prf_vel(ref_ints, ref_notes, ref_vels, est_ints, est_notes, est_vels)
        notes_vel_prec, notes_vel_recall = score_notes_vel[0], score_notes_vel[1]
        note_vel_f1 = score_notes_vel[2]

        note_metrics["note_Precision"].append(notes_precision)
        note_metrics["note_Recall"].append(notes_recall)
        note_metrics["note_F1"].append(note_f1)
        note_metrics["note_Precision_no_offset"].append(notes_prec_no_off)
        note_metrics["note_Recall_no_offset"].append(notes_recall_no_off)
        note_metrics["note_F1_no_offset"].append(note_f1_no_off)
        note_metrics["note_vel_Precision"].append(notes_vel_prec)
        note_metrics["note_vel_Recall"].append(notes_vel_recall)
        note_metrics["note_vel_F1"].append(note_vel_f1)

        # Get the frame-level metrics
        ref_frame_roll = get_frames(midi, 0, len(audio)/extraction_config["sample_rate"], \
                                    frame_rate, max_pitch, min_pitch)
        est_frame_roll = (result_dict["frame_roll"] >= 0.3).astype(int)

        # chop the rolls
        est_frame_roll = est_frame_roll[:ref_frame_roll.shape[0], :]
        ref_frame_roll = ref_frame_roll[:est_frame_roll.shape[0], :]

        frame_scores = prfs(ref_frame_roll.flatten(), est_frame_roll.flatten())
        frame_metrics['Precision'].append(frame_scores[0][1])
        frame_metrics['Recall'].append(frame_scores[1][1])
        frame_metrics['F1'].append(frame_scores[2][1])

    # Find average of frame and note metrics
    for key in frame_metrics:
        frame_metrics[key] = round(np.mean(frame_metrics[key]).item(), 2)

    for key in note_metrics:
        note_metrics[key] = round(np.mean(note_metrics[key]).item(), 2)

    print("Frame metrics are: ")
    pprint.pprint(frame_metrics)
    print("\nNote metrics are: ")
    pprint.pprint(note_metrics)

    return frame_metrics, note_metrics

@torch.no_grad()
def evaluate_pedal(config: dict):
    test_files = glob.glob("data/kong/test/*.h5")
    dataset_length = len(test_files)

    model = load_pedal(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extraction_config = load_extract_config()
    frame_rate = extraction_config["frame_rate"]
    frame_thresh = config["frame_threshold"]
    pedal_thresh = config["pedal_offset_threshold"]

    frame_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    event_metrics = {'events_Precision': [], 'events_Recall': [], 'events_F1': [], 'events_Precision_no_offset': [], \
                    'events_Recall_no_offset': [], 'events_F1_no_offset': []}

    for i in tqdm(range(dataset_length), total=dataset_length, desc="Extracting results...."):
        file = test_files[i]
        with h5py.File(file, 'r') as hf:
            audio = hf["audio"][:]
            midi_path = hf.attrs["midi_path"]

            # convert audio to float32
            audio = (audio / 32767.0).astype(np.float32)

        window_size = extraction_config["window_size"]

        # load midi
        # sustain MIDI
        midi = extend_pedal(pretty_midi.PrettyMIDI(midi_path))
        result_dict: dict = defaultdict(list)
        window_samples = int(window_size * extraction_config["sample_rate"])
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
        ref_frames, ref_events = get_pedal_frames(midi, 0, len(audio)/extraction_config["sample_rate"], frame_rate) 
        est_events = get_pedal_events(result_dict, pedal_thresh, frame_thresh, frame_rate)   

        if len(ref_events) == 0:
            continue 
        
        # ensure that offset is >= onset times
        # Get locations of invalid estimated note events where offset < onset
        if est_events is None:
            est_events = np.zeros((0, 2))

        locs = np.where(est_events[:, 1] < est_events[:, 0])[0]

        if len(locs) > 0:
            invalid_events = est_events[locs]
            print(f"Invalid estimated note events where offset < onset: {invalid_events}")
            # Remove the invalid events
            est_events = np.delete(est_events, locs, axis=0)

        # Assert that there are no invalid events left
        assert np.all(est_events[:, 1] >= est_events[:, 0]), \
            f"Estimated note events have invalid timings.{est_events}"

        est_ints = est_events[:, :2]
        ref_ints = ref_events

        # Get the event-level metrics
        score_notes_off = prf(ref_ints, np.ones(ref_ints.shape[0]), est_ints, np.ones(est_ints.shape[0]))
        events_precision, events_recall = score_notes_off[0], score_notes_off[1]
        events_f1 = score_notes_off[2]

        score_events_no_off = prf(ref_ints, np.ones(ref_ints.shape[0]), est_ints, np.ones(est_ints.shape[0]), offset_ratio=None)
        events_prec_no_off, events_recall_no_off, events_f1_no_off = score_events_no_off[0], score_events_no_off[1], score_events_no_off[2]

        event_metrics["events_Precision"].append(events_precision)
        event_metrics["events_Recall"].append(events_recall)
        event_metrics["events_F1"].append(events_f1)
        event_metrics["events_Precision_no_offset"].append(events_prec_no_off)
        event_metrics["events_Recall_no_offset"].append(events_recall_no_off)
        event_metrics["events_F1_no_offset"].append(events_f1_no_off)

        # Get the frame-level metrics
        ref_frame_roll = ref_frames
        est_frame_roll = (result_dict["pedal_frame_roll"] >= pedal_thresh).astype(int)

        # chop the rolls
        est_frame_roll = est_frame_roll[:ref_frame_roll.shape[0]]
        ref_frame_roll = ref_frame_roll[:est_frame_roll.shape[0]]

        frame_scores = prfs(ref_frame_roll.flatten(), est_frame_roll.flatten())
        frame_metrics['Precision'].append(frame_scores[0][1])
        frame_metrics['Recall'].append(frame_scores[1][1])
        frame_metrics['F1'].append(frame_scores[2][1])
    
    # Find average of frame and note metrics
    for key in frame_metrics:
        frame_metrics[key] = round(np.mean(frame_metrics[key]).item(), 2)

    for key in event_metrics:
        event_metrics[key] = round(np.mean(event_metrics[key]).item(), 2)

    # format the print output
    print("Frame metrics are: ")
    pprint.pprint(frame_metrics)
    print("\nEvent metrics are: ")
    pprint.pprint(event_metrics)

    return frame_metrics, event_metrics
