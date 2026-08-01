# Imports
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional
import h5py
import collections
import pretty_midi
import copy
import torch
from tqdm import tqdm
pretty_midi.pretty_midi.MAX_TICK = 1e10

class NoteSeg:
    """
        Returns note events and pedal
        events within a given time segment
    """
    def __init__(self, midi_data: dict, start: float, end: float):
        """
            Initialize the NoteSeg object

            Args:
                midi_data (dict): compact MIDI from parse_midi_compact
                start (float): Start time of the segment
                end (float): End time of the segment
        """
        self.midi_data = midi_data
        self.start = start
        self.end = end
        self.new_midi = pretty_midi.PrettyMIDI()
        self.chopped_notes = []
        self.set_notes()
        self.set_pedals()

    def set_notes(self):
        """
            Set the note events within the given time segment
            to the new_midi object
        """
        for instrument in self.midi_data["instruments"]:
            # create a new instrument
            new_instrument = pretty_midi.Instrument(program=instrument["program"],
                                                    is_drum=instrument["is_drum"],
                                                    name=instrument["name"])

            # add the new instrument to the new midi object
            self.new_midi.instruments.append(new_instrument)
            if not instrument["is_drum"]:
                notes = instrument["notes"]
                # Select the overlapping notes with numpy rather than walking
                # every note in the performance. A 10 s window keeps a few dozen
                # of the several thousand a MAESTRO file holds, and this runs
                # once per training item.
                if len(notes) == 0:
                    continue
                keep = (notes[:, 0] <= self.end) & (notes[:, 1] >= self.start)
                for onset, offset, pitch, velocity in notes[keep]:
                    if offset > self.end:
                        offset = self.end
                        mask_flag = True
                    else:
                        mask_flag = False

                    # create a new note
                    new_note = pretty_midi.Note(start=onset, end=offset,
                                                pitch=int(pitch), velocity=int(velocity))

                    # add the new note to the new instrument
                    new_instrument.notes.append(new_note)

                    if mask_flag:
                        # This is to deal with the masking effect
                        self.chopped_notes.append(new_note)

    def set_pedals(self):
        """
            Set the pedal events within the given time segment
            to the new_midi object

            The sustain on/off pairing is a state machine over the whole
            performance and does not depend on the segment, so it is done once
            at parse time; only the time filter belongs here.
        """
        pedal_events = self.midi_data["pedal_events"]
        if len(pedal_events):
            keep = ((pedal_events[:, 0] < self.end)
                    & (pedal_events[:, 1] > self.start))
            pedal_events = pedal_events[keep]

        # Now, what we do is to add the pedal events to the new midi object
        for instrument in self.new_midi.instruments:
            if not instrument.is_drum:
                for onset, offset in pedal_events:
                    if offset > self.end:
                        offset = self.end

                    cc_on = pretty_midi.ControlChange(number=64, value=127, time=onset)
                    cc_off = pretty_midi.ControlChange(number=64, value=0, time=offset)
                    instrument.control_changes.append(cc_on)
                    instrument.control_changes.append(cc_off)


def parse_midi_compact(midi_path: str) -> dict:
    """Parse one MIDI file into the minimum this dataset actually reads.

    kong_dataset used to call pretty_midi.PrettyMIDI(midi_path) inside
    __getitem__, re-parsing an entire ~10 minute performance to label a single
    10 s excerpt. Measured on MAESTRO that is a median of 63 ms per item against
    ~37 ms of augmentation, and each file is parsed once per excerpt it
    contains -- roughly 600 times.

    An LRU cache of PrettyMIDI objects does not fix it: a parsed performance is
    ~8.1 MB, and because training shuffles globally over 962 files the hit rate
    tracks cache_size / n_files. 64 entries buys 6.6% at 518 MB per worker; a
    useful hit rate needs ~7.8 GB per worker.

    So keep only what NoteSeg reads -- note tuples and paired sustain events --
    as numpy arrays. That is ~40x smaller, small enough to hold every file at
    once, and because the bulk is numpy buffers rather than Python objects it
    survives a fork without being copied into each worker.

    Args
    ----
        midi_path (str): path to the MIDI file

    Returns
    -------
        dict: {"instruments": [{program, is_drum, name, notes}], "pedal_events"}
            where notes is (N, 4) of [start, end, pitch, velocity] and
            pedal_events is (K, 2) of [onset_time, offset_time].
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    instruments = []
    for instrument in midi.instruments:
        notes = np.array([(n.start, n.end, n.pitch, n.velocity)
                          for n in instrument.notes],
                         dtype=np.float64).reshape(-1, 4)
        instruments.append({
            "program": instrument.program,
            "is_drum": instrument.is_drum,
            "name": instrument.name,
            "notes": notes,
        })

    # Pair sustain on/off exactly as set_pedals used to, scanning instruments in
    # order because the state carries across them.
    pedal_events = []
    pedal_on = None
    for instrument in midi.instruments:
        for cc in instrument.control_changes:
            if cc.number == 64:
                if cc.value >= 64 and pedal_on is None:
                    pedal_on = cc.time
                elif cc.value < 64 and pedal_on is not None:
                    pedal_events.append((pedal_on, cc.time))
                    pedal_on = None

    return {
        "instruments": instruments,
        "pedal_events": np.array(pedal_events, dtype=np.float64).reshape(-1, 2),
    }


class KongDataset(Dataset):
    def __init__(self, emb_path: str, extend_pedal: bool = True,
                 augmentator=None) -> None:
        """
            Instantiate dataset class.

            Args
            ----
                emb_path (str): path to npz files containing audio and feature data.
                extend_pedal (bool): Whether to extend note offsets. Default is True.
                augmentator (Augmentator | None): Applied to the excerpt's
                    waveform before it is returned. Labels are derived from the
                    MIDI and are untouched. Leave None for a clean baseline.
        """
        super().__init__()
        if emb_path is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        self.data = [] # path to npz files
        self.extend_pedal_flag = extend_pedal
        self.augmentator = augmentator

        # Set by the epoch callback so augmentation is redrawn each epoch. Left
        # at 0 the augmentator applies identical damage to identical excerpts
        # for the whole run -- no crash, just a fixed pre-corrupted dataset.
        self.epoch = 0
        assert Path(emb_path).exists(), f"{emb_path} does not exist."
        self.data.extend(sorted(Path(emb_path).rglob("*.h5")))

        # Load the min and max pitches from the extraction config
        splits = emb_path.split("/")
        if splits[-1] == "":
            splits = splits[:-1]
        base_path = "/".join(splits[:-1])
        with h5py.File(f"{base_path}/extraction_config.h5", "r") as hf:
            self.min_pitch = hf.attrs["min_pitch"]
            self.max_pitch = hf.attrs["max_pitch"]
            self.frame_rate = hf.attrs["frame_rate"]
    
        
        self.seg_list = []
        self.split = str(Path(emb_path).stem)
        for path in self.data:
            with h5py.File(str(path), 'r') as hf:
                if "train"==self.split or "val"==self.split:
                    midi_path = hf.attrs['midi_path']
                    audio_len = hf['audio'].shape[-1]
                    window_samples = hf.attrs['window_samples']
                    hop_samples = hf.attrs['hop_samples']
                    start_list = np.arange(0, audio_len - window_samples + 1, hop_samples).tolist()
                    zipped_data = list(zip([path]*len(start_list), [midi_path]*len(start_list), start_list))
                    self.seg_list.extend(zipped_data)
                else:
                    # for test set, we use the whole file
                    midi_path = hf.attrs['midi_path']
                    self.seg_list.append((path, midi_path, 0))

        # Parse every MIDI once, here in the parent process, rather than once
        # per item inside __getitem__. See parse_midi_compact for why this is a
        # store rather than a cache. Built before the dataloader forks, so the
        # workers share these arrays instead of each holding a copy.
        midi_paths = sorted({str(m) for _, m, _ in self.seg_list})
        self.midi_store = {
            p: parse_midi_compact(p)
            for p in tqdm(midi_paths, desc=f"Parsing {self.split} MIDI")
        }
        
        

    def __getitem__(self, idx: int) -> dict:
        """
            Get item at a particular index.

            Args
            ----
                idx (int): Index to segment list

            Returns
            -------
                dict: A dictionary containing the audio and labels
        """
        # metadata (tuple): (path to h5 file, path to midi file, start sample)
        metadata = self.seg_list[idx]
        path, midi_path, start_sample = metadata
        item_dict = {}

        with h5py.File(path, 'r') as hf:
            end_sample = start_sample + hf.attrs['window_samples']
            audio = hf['audio'][start_sample:end_sample]

            # convert audio to float32
            audio = (audio / 32767.0).astype(np.float32)

            sr = hf.attrs["sample_rate"]
            if self.augmentator is not None:
                # Seeded from the excerpt's identity rather than call order, so
                # the same excerpt in the same epoch draws the same augmentation
                # no matter the batch size, worker count or step. The labels
                # below come from the MIDI and are unaffected.
                audio = self.augmentator(
                    audio,
                    track_id=Path(path).stem,
                    excerpt_start=start_sample / sr,
                    epoch=self.epoch,
                )

            item_dict['audio'] = audio

            # the MIDI was parsed once at construction; see parse_midi_compact
            midi = self.midi_store[str(midi_path)]
            start_time = start_sample / sr
            end_time = end_sample / sr
            label_dict  = self._get_label_roll(midi, start_time, end_time, self.frame_rate)

            # remove note_events and pedal_events from label_dict (we don't need them)
            label_dict.pop('note_events')
            label_dict.pop('pedal_events')

            # update item_dict with label_dict
            item_dict.update(label_dict)
        return item_dict

    def __len__(self) -> int:
        """ 
            Return the number of dataset
            items.

            Returns
            -------
                length (int): length of segment items.
        """
        if self.split == "train":
            return len(self.seg_list)
        else:
            # We return 100 to reduce training time
            return 100
        
    
    def _get_label_roll(self, midi: dict, \
                  start: float, end: float, frame_rate: int) -> dict:
        """
            Get the label and pedal rolls for a given audio segment.
            The regressed rolls generated follow Kong's model!

            Args
            ----
                midi (dict): compact MIDI from parse_midi_compact
                start (float): Start time in seconds
                end (float): End time in seconds
                frame_rate (int): The frame rate of the piano roll

            Returns
            -------
                target_dict: {
                    label_frames (np.ndarray): Frames label
                    label_onsets (np.ndarray): Onsets label
                    label_offsets (np.ndarray): Offsets label
                    label_reg_onsets (np.ndarray): Label regression onsets for the audio segment
                    label_reg_offsets (np.ndarray): Label regression offsets for the audio segment
                    label_velocities (np.ndarray): Label velocities for the audio segment
                    pedal_onset (np.ndarray): Pedal onset label
                    pedal_frames (np.ndarray): Pedal frames label
                    pedal_offset (np.ndarray): Pedal offset label
                    pedal_reg_onset (np.ndarray): Regressed pedal onset label
                    pedal_reg_offset (np.ndarray): Regressed pedal offset label
                    mask_roll (np.ndarray): Mask roll label to remove all events that do not occur
                                            in the audio segment
                    note_events (np.ndarray): Array of note events for the audio segment
                    pedal_events (np.ndarray): Array of pedal events for the audio segment
                } 
        """
        # init target_dict and set the keys to None
        target_dict: dict[str, Optional[np.ndarray]] = {
            'label_frames': None,
            'label_onsets': None,
            'label_offsets': None,
            'label_reg_onsets': None,
            'label_reg_offsets': None,
            'label_velocities': None,
            'pedal_onset': None,
            'pedal_frames': None,
            'pedal_offset': None,
            'pedal_reg_onset': None,
            'pedal_reg_offset': None,
            'mask_roll': None,
            'note_events': None
        }

        duration = end - start
        num_frames = int(round(duration * frame_rate)) + 1
        num_pitches = self.max_pitch - self.min_pitch + 1

        # initialize the labels
        label_frames = np.zeros((num_frames, num_pitches))
        label_onsets = np.zeros((num_frames, num_pitches))
        label_offsets = np.zeros((num_frames, num_pitches))
        label_velocities = np.zeros((num_frames, num_pitches))
        label_reg_onsets = np.ones((num_frames, num_pitches))
        label_reg_offsets = np.ones((num_frames, num_pitches))
        pedal_onset = np.zeros((num_frames,))
        pedal_frames = np.zeros((num_frames,))
        pedal_offset = np.zeros((num_frames,))
        pedal_reg_onset = np.ones((num_frames,))
        pedal_reg_offset = np.ones((num_frames,))
        mask_roll = np.ones((num_frames, num_pitches))

        note_events = []  # contains (onset, offset, pitch, velocity)

        # Create a NoteSeg object to get the note events corresponding to the audio segment
        note_seg_obj = NoteSeg(midi, start, end)
        new_midi = note_seg_obj.new_midi
        #new_midi = self._pedal_extend(midi)

        if self.extend_pedal_flag:
            new_midi = self.extend_pedal(new_midi)

        # set the mask roll for chopped notes
        for note in note_seg_obj.chopped_notes:
            pitch = note.pitch - self.min_pitch
            start_frame = int(round(frame_rate * (note.start - start)))
            end_frame = int(round(frame_rate * (note.end - start)))
            mask_roll[max(0, start_frame):end_frame+1, pitch] = 0


        for instrument in new_midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch = note.pitch - self.min_pitch

                    note_events.append([note.start, note.end, pitch, note.velocity])
                    start_frame = int(round(frame_rate * (note.start - start)))
                    end_frame = int(round(frame_rate * (note.end - start)))

                    note_onset = note.start
                    note_offset = note.end

                    label_frames[max(0, start_frame):min(end_frame + 1, num_frames), pitch] = 1
                    label_velocities[max(0, start_frame):min(end_frame + 1, num_frames), pitch] = note.velocity
                    label_offsets[end_frame, pitch] = 1
                    label_reg_offsets[end_frame, pitch] = \
                        (note_offset - start) - (end_frame / frame_rate)

                    if start_frame >= 0:
                        label_onsets[start_frame, pitch] = 1
                        label_reg_onsets[start_frame, pitch] = \
                        (note_onset - start) - (start_frame / frame_rate)
                    else:
                        # We will never get here
                        mask_roll[:end_frame + 1, pitch] = 0
        
        for pitch in range(num_pitches):
            label_reg_onsets[:, pitch] = self._get_reg(label_reg_onsets[:, pitch], frame_rate)
            label_reg_offsets[:, pitch] = self._get_reg(label_reg_offsets[:, pitch], frame_rate)
        
        # Get the pedal events
        CC_SUSTAIN_PEDAL = 64
        temp = self._pedal_state_before_start(new_midi, start)
        if temp is not None:
            is_pedal_on, pedal_start_before_seg = temp[0], temp[1]
            onset_time = pedal_start_before_seg - start
            abs_onset_time = pedal_start_before_seg
            frame_pedal_on = int(round((onset_time) * frame_rate))  # if is_pedal_on, then this will be -ve
        else:
            is_pedal_on = False
        
        pedal_events: list[list] = [] # contains [onset_time, offset_time]

        for instrument in new_midi.instruments:
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
                        pedal_offset[frame_now] = 1
                        pedal_reg_offset[frame_now] = \
                        (time_now) - (frame_now / frame_rate)
                        
                        if frame_pedal_on >= 0:
                            pedal_onset[frame_pedal_on] = 1
                            pedal_reg_onset[frame_pedal_on] = \
                            (onset_time) - (frame_pedal_on / frame_rate)
                        
                        is_pedal_on = False
                        pedal_events.append([abs_onset_time, cc.time])
            
        if is_pedal_on:
            pedal_events.append([abs_onset_time, end])

            # calc. frame_pedal_on
            frame_pedal_on = int(round((onset_time) * frame_rate))
            
            if frame_pedal_on >= 0 and frame_pedal_on < num_frames:
                pedal_onset[frame_pedal_on] = 1
                pedal_reg_onset[frame_pedal_on] = \
                (onset_time) - (frame_pedal_on / frame_rate)
                pedal_frames[frame_pedal_on:] = 1
                pedal_offset[num_frames - 1] = 1
                pedal_reg_offset[num_frames - 1] = \
                    (end - start) - ((num_frames - 1) / frame_rate)
        
        # Get the pedal regressed values
        pedal_reg_onset = self._get_reg(pedal_reg_onset, frame_rate)
        pedal_reg_offset = self._get_reg(pedal_reg_offset, frame_rate)

        # update the target_dict
        target_dict['label_frames'] = label_frames
        target_dict['label_onsets'] = label_onsets
        target_dict['label_offsets'] = label_offsets
        target_dict['label_reg_onsets'] = label_reg_onsets
        target_dict['label_reg_offsets'] = label_reg_offsets
        target_dict['label_velocities'] = label_velocities
        target_dict['pedal_onset'] = pedal_onset
        target_dict['pedal_frames'] = pedal_frames
        target_dict['pedal_offset'] = pedal_offset
        target_dict['pedal_reg_onset'] = pedal_reg_onset
        target_dict['pedal_reg_offset'] = pedal_reg_offset
        target_dict['mask_roll'] = mask_roll
        target_dict['note_events'] = np.array(note_events)
        target_dict['pedal_events'] = np.array(pedal_events)
        return target_dict

    def extend_pedal(self, midi: pretty_midi.PrettyMIDI):
        """
            Extend the sustain events in the MIDI file
            according to original code from the Kong 
            Bytedance repo. This implementation is buggy, 
            but I am keeping it for reproducibility.

            Args
            ----
                midi (pretty_midi.PrettyMIDI): pretty midi object
            
            Returns
            -------
                ext_midi (pretty_midi.PrettyMIDI): Extended MIDI
        """
        # Extract the note events and pedal events
        note_events = []
        pedals = []
        pedal_events = []

        for instrument in midi.instruments:
            for cc in instrument.control_changes:
                if cc.number == 64:
                    pedals.append(cc)

        pedal_on = None
        for cc in pedals:
            if cc.value >= 64 and pedal_on is None:
                pedal_on = cc.time
            elif cc.value < 64 and pedal_on is not None:
                pedal_events.append({
                    'onset_time': pedal_on,
                    'offset_time': cc.time
                })
                pedal_on = None
        

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note_events.append({
                        'onset_time': note.start,
                        'offset_time': note.end,
                        'midi_note': note.pitch,
                        'velocity': note.velocity
                    })

        pedal_events_original = pedal_events.copy()
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())
        
        # Create a new pretty_midi object
        ex_midi = pretty_midi.PrettyMIDI()

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        ex_instrument = pretty_midi.Instrument(program=piano_program)

        for note_event in ex_note_events:
            ex_note = pretty_midi.Note(
                velocity=note_event['velocity'],
                pitch=note_event['midi_note'],
                start=note_event['onset_time'],
                end=note_event['offset_time']
            )
            ex_instrument.notes.append(ex_note)

        ex_midi.instruments.append(ex_instrument)

        # Add the pedal control changes
        for pedal_event in pedal_events_original:
            cc_on = pretty_midi.ControlChange(number=64, value=127, time=pedal_event['onset_time'])
            cc_off = pretty_midi.ControlChange(number=64, value=0, time=pedal_event['offset_time'])
            ex_instrument.control_changes.append(cc_on)
            ex_instrument.control_changes.append(cc_off)

        return ex_midi

    def _pedal_state_before_start(self, midi: pretty_midi.PrettyMIDI, start: float):
        """
            Check if the sustain pedal is pressed before the start of the audio segment.
            This function checks all the control changes for the sustain pedal and returns
            True if the pedal is pressed before the start of the audio segment, otherwise
            returns False.

            Args
            ----
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                start (float): Start time of the audio segment

            Returns
            --------
                bool: True if the pedal is pressed before the start of the audio segment, 
                      False otherwise
        """
        CC_SUSTAIN_PEDAL = 64
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for cc in reversed([_e for _e in instrument.control_changes if _e.number == CC_SUSTAIN_PEDAL]):
                    if cc.time < start:
                        return cc.value >= 64, cc.time
    
    def _pedal_extend(self, midi_path: str, CC_SUSTAIN=64):
        """
            Extend the sustain events in the MIDI file.
            This implementation is from Google and is not buggy
            unlike the original Kong implementation.
            
            Args
            ----
                midi_path (str): Path to the MIDI file.
                CC_SUSTAIN (int): MIDI control change number for sustain pedal. Default is 64.
            
            Returns
            -------
                midi_copy (pretty_midi.PrettyMIDI): MIDI file with extended sustain events.
        """
        # make a copy of the MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_copy = copy.deepcopy(midi)
        
        # we want to deal with events in the following order:
        # PEDAL_DOWN, PEDAL_UP, ONSET, OFFSET
        PEDAL_DOWN = 0
        PEDAL_UP = 1
        ONSET = 2
        OFFSET = 3

        # we will store all pedal and note events in a list
        # it will be sorted by time (pedal down, pedal up, onset, offset)
        events = []
        events.extend([(note.start, ONSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        events.extend([(note.end, OFFSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        
        for instrument in midi_copy.instruments:
            if not instrument.is_drum:
                for cc in instrument.control_changes:
                    if cc.number == CC_SUSTAIN:
                        if cc.value >= 64:
                            events.append((cc.time, PEDAL_DOWN, [cc, instrument]))
                        else:
                            events.append((cc.time, PEDAL_UP, [cc, instrument]))

        # sort the events by time and event type
        events.sort(key=lambda x: (x[0], x[1]))

        # We will keep a track of notes to extend (notes that fall within (pedal_down, pedal_up))
        # for each instrument
        extend_insts = collections.defaultdict(list)
        sus_insts = collections.defaultdict(bool) # stores sustain state for each instrument

        # We go through the events and extend the notes
        time = 0
        for time, event_type, event in events:
            if event_type == PEDAL_DOWN:
                sus_insts[event[1]] = True
            elif event_type == PEDAL_UP:
                sus_insts[event[1]] = False

                # If the pedal is up, we will end all of the notes
                # currently being extended
                ext_notes_inst = [] # Store new notes to extend
                for note in extend_insts[event[1]]:
                    if note.end < time:
                        # This note was extended, so we can end it
                        note.end = time
                    else:
                        # This note has not ended, so we still keep it
                        ext_notes_inst.append(note)
                extend_insts[event[1]] = ext_notes_inst
            elif event_type == ONSET:
                if sus_insts[event[1]]:
                    ext_notes_inst = []
                    # if sustain is on, we have to stop notes that are currently being extended
                    for note in extend_insts[event[1]]:
                        if note.pitch == event[0].pitch:
                            note.end = time
                            if note.start == note.end:
                                # it means this note now has zero duration,
                                # according to the official implementation, we should not keep it
                                event[1].notes.remove(note)
                        else:
                            # if the note is not the same as the one being extended,
                            # we can just add it to the list of notes to extend
                            ext_notes_inst.append(note)
                    extend_insts[event[1]] = ext_notes_inst
                
                # Add the new set of notes to extend to the list
                extend_insts[event[1]].append(event[0])
            elif event_type == OFFSET:
                # if susain is on, we will not end the note
                # so let's consider where sustain is off
                if not sus_insts[event[1]]:
                    if event[0] in extend_insts[event[1]]:
                        extend_insts[event[1]].remove(event[0])
            else:
                raise AssertionError(f"Unknown event type: {event_type}")    
        
        # End notes that are still being extended
        for instrument in extend_insts.values():
            for note in instrument:
                note.end = time
        
        return midi_copy
    
    def _get_reg(self, input: np.ndarray, frame_rate) -> np.ndarray:
        """
            Get the regression for a given roll using
            Kong's approach! The code is slightly different
            from the original.

            Args
            -----
                input (np.ndarray): Roll to regress
                frame_rate (int): The frame rate of the piano roll
                
            Returns
            -------
                output (np.ndarray): Regressed roll
        """
        step = 1. / frame_rate

        # Get the locations where the events (onsets/offsets) occur
        # since the initial regression matrix is 1 everwhere and an
        # onset/offset can occur whithin a frame resolution of 1/frame_rate
        # then 0.5 is enough to determine the locations of the events
        locts = np.where(input < 0.5)[0]
        if len(locts) == 0:
            output = np.ones_like(input)
        else:
            # Every frame takes its value from the nearest event, with the
            # changeover at floor((a + b) / 2) between neighbouring events a and
            # b -- frames before the first event and after the last one clamp to
            # it. Written as three nested loops this was 95% of the dataset's
            # per-item cost, called once per pitch per onset/offset roll, i.e.
            # 178 times an item. searchsorted over the midpoints assigns every
            # frame at once and reproduces the same boundaries exactly.
            frames = np.arange(len(input))
            mids = (locts[:-1] + locts[1:]) // 2
            nearest = locts[np.searchsorted(mids, frames, side="right")]
            output = step * (frames - nearest) - input[nearest]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output

def collate_fn(list_data_dict):
    """ 
        Collate function for dataloader. 

        Args
        ----
            list_data_dict (dict): contains a list of dicts
                                   for the batch.
        
        Returns
        -------
            np_data_dict (dict): Contains the appropriate data
                                 across the batch for each 
                                 key.
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    # convert to tensors
    for key in np_data_dict.keys():
        np_data_dict[key] = torch.tensor(np_data_dict[key])

    # if any of them is bool, convert to int
    for key in np_data_dict.keys():
        if np_data_dict[key].dtype == torch.bool:
            np_data_dict[key] = np_data_dict[key].int()

    return np_data_dict

def get_note_events(midi, start: float, end: float, min_pitch: int = 21) -> list:
        """ 
            Gets the note events as a list
            within a time range. 

            Args
            ----
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                start (float): Start time
                end (float): End time
                min_pitch (int): Minimum MIDI pitch for our MIDI notes
            
            Returns
            -------
                note_events (np.ndarray): List of note events.
        """
        note_events = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 

                    pitch = note.pitch - min_pitch

                    # Need to deal with the case where the note starts before
                    # the start of the segment and ends after the end of the segment
                    # Not sure if what I have done below is the right thing to do
                    note_events.append([note.start, note.end, pitch, note.velocity])
        
        return note_events
