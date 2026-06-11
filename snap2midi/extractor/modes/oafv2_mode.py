from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pretty_midi
import collections
import copy
import librosa


class _OAFV2Mode(_BaseMode):
    """ 
        _OAFV2Mode is a class for extracting
        the necessary data for further training
        the Jongwook implementation of Onsets and 
        Frames.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        # we want to deal with events in the following order:
        # PEDAL_DOWN, PEDAL_UP, ONSET, OFFSET when extending
        # pedals
        self.PEDAL_DOWN = 0
        self.PEDAL_UP = 1
        self.ONSET = 2
        self.OFFSET = 3
        self._oafv2()
    
    def _oafv2(self):
        """ 
            Performs extraction based on config.
        """
        config = self.config
        # Check if the path is None
        if self.path is None or not Path(self.path).exists():
            raise ValueError(f"Path {self.path} does not exist!")
        
        if config["dataset_name"] not in ["maps", "maestro"]:
            raise ValueError(f"Dataset {config['dataset_name']} not supported for OAFV2 mode!")
        
        if config["sample_rate"] is None:
            raise ValueError("sample_rate must be set!")
        
        self.sample_rate = config["sample_rate"]
        self.extend_pedal=self.config["extend_pedal"] # If to extend note offsets
        self.min_pitch = config["min_pitch"]
        self.max_pitch = config["max_pitch"]
        self.num_pitches = self.max_pitch - self.min_pitch + 1
        self.onset_frame_length = config["onset_frame_length"] # unit is in frames
        self.offset_frame_length = config["offset_frame_length"] # unit is in frames

        # Perform the extraction
        # extract the data for the train, val and test splits
        # Implement the extraction logic here
        if self.dataset_name == "maps":
            train_files, val_files, test_files = self._get_maps_train_val_test()
        elif self.dataset_name == "maestro":
            train_files, val_files, test_files = self._get_maestro_train_val_test()

        # create the directory for the splits
        for split in ["train", "val", "test"]:
            split_path = Path(f"./{self.save_name}/{split}")
            split_path.mkdir(parents=True, exist_ok=True)

        for _ in [("train", train_files), ("val", val_files), ("test", test_files)]:
            self._extract(_[1], _[0])

    def _extract(self, split_files: list, split: str):
        """
            Extract audio and MIDI files from the dataset.
            We specifically follow the extraction params
            used in Jongwook's OAF implementation.
        """
        # parse the (audio_file, midi_file) in split_files
        for audio_file, midi_file in tqdm(split_files, \
                        desc=f"Processing audio and MIDI files for {split} split..."):
            assert audio_file.exists(), f"{audio_file} does not exist"
            assert midi_file.exists(), f"{midi_file} does not exist"
            
            # load MIDI
            midi_obj = pretty_midi.PrettyMIDI(midi_file)
            if self.extend_pedal: 
                midi_obj = self._pedal_extend(midi_obj)
            
            audio, _ = librosa.load(audio_file, sr=self.sample_rate)
            assert self.sample_rate == _, f"[Loaded sample rate]{_} != {self.sample_rate}[Config sample rate]"
            hop_length = self.config["hop_length"]
            n_frames = len(audio - 1) // hop_length
            label = np.zeros((n_frames, self.num_pitches), dtype=np.float32)
            velocity = np.zeros((n_frames, self.num_pitches), dtype=np.float32)

            for instrument in midi_obj.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        onset = note.start
                        offset = note.end
                        pitch = note.pitch - self.min_pitch
                        vel = note.velocity

                        onset_frame_start = int(round(onset * self.sample_rate / hop_length))
                        onset_frame_end = min(n_frames, onset_frame_start + self.onset_frame_length)
                        offset_frame_start = int(round(offset * self.sample_rate / hop_length))
                        offset_frame_start = min(n_frames, offset_frame_start)
                        offset_frame_end = min(n_frames, offset_frame_start + self.offset_frame_length)

                        label[onset_frame_start:onset_frame_end, pitch] = 3
                        label[onset_frame_end:offset_frame_start, pitch] = 2
                        label[offset_frame_start:offset_frame_end, pitch] = 1
                        velocity[onset_frame_start:offset_frame_start, pitch] = vel/128

            if self.dataset_name != "slakh":
                store_path = f"./{self.save_name}/{split}/{str(audio_file.stem)}.npz"
            else:
                # For slakh, we need to get the track name
                # from the audio file path
                track_name = audio_file.parent.parent.stem
                store_path = f"./{self.save_name}/{split}/{track_name}_{str(audio_file.stem)}.npz"

            store_dict = {
                "audio": audio,
                "label": label,
                "velocity": velocity
            }

            np.savez(store_path, **store_dict)
    
    def _pedal_extend(self, midi, CC_SUSTAIN=64):
        """
            Extend the sustain events in the MIDI file.
            
            Args
            ----
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                CC_SUSTAIN (int): MIDI control change number for sustain pedal. Default is 64.
            
            Returns
            -------
                midi_copy (pretty_midi.PrettyMIDI): MIDI file with extended sustain events.
        """
        # make a copy of the MIDI file
        midi_copy = copy.deepcopy(midi)

        # we will store all pedal and note events in a list
        # it will be sorted by time (pedal down, pedal up, onset, offset)
        events = []
        events.extend([(note.start, self.ONSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        events.extend([(note.end, self.OFFSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        
        for instrument in midi_copy.instruments:
            if not instrument.is_drum:
                for cc in instrument.control_changes:
                    if cc.number == CC_SUSTAIN:
                        if cc.value >= 64:
                            events.append((cc.time, self.PEDAL_DOWN, [cc, instrument]))
                        else:
                            events.append((cc.time, self.PEDAL_UP, [cc, instrument]))

        # sort the events by time and event type
        events.sort(key=lambda x: (x[0], x[1]))

        # We will keep a track of notes to extend (notes that fall within (pedal_down, pedal_up))
        # for each instrument
        extend_insts = collections.defaultdict(list)
        sus_insts = collections.defaultdict(bool) # stores sustain state for each instrument

        # We go through the events and extend the notes
        time = 0
        for time, event_type, event in events:
            if event_type == self.PEDAL_DOWN:
                sus_insts[event[1]] = True
            elif event_type == self.PEDAL_UP:
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
            elif event_type == self.ONSET:
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
            elif event_type == self.OFFSET:
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
