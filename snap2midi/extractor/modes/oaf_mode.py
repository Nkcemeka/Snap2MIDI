from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pretty_midi
from snap2midi.extractor.utils import HandcraftedFeatures, SUPPORTED_FEATURES
import librosa
import collections
import copy
import bisect

class _OafFramedEvents:
    """
        OafFramedEvents is a class that generates 
        framed audio and MIDI events based on the idea 
        introduced in the Onsets and Frames paper.
        It splits audio into frames ensuring that
        the split does not occur in the middle of an event.
        And if a split must occur in the middle of an event,
        it uses a zero-crossing of the audio signal to determine
        the split point.

        Credits: `https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription`
        for the original implementation.This implementation is adapted to work with the snap2midi library.
    """
    def __init__(self, audio_path: str, midi_path: str, min_frame_secs: float, max_frame_secs: float, \
                 sample_rate: float, frame_rate: float, min_pitch: float = 21, max_pitch: float = 108,
                 onset_length: float = 32, offset_length: float = 32, ignore_duration: bool = False) -> None:
        """
            Args:
                audio_path (str): Path to audio file
                midi_path (str): Path to MIDI file
                min_frame_secs (float): Minimum frame duration in seconds
                max_frame_secs (float): Maximum frame duration in seconds (ideal duration we want for each segment)
                sample_rate (float): Sample rate of audio file
                frame_rate (float): Frame rate for labels
                min_pitch (int): Minimum pitch value (default is 21, A0)
                max_pitch (int): Maximum pitch value (default is 108, C8)
                onset_length (float): Length of onset in ms (default is 32)
                offset_length (float): Length of offset in ms (default is 32)
            
            Returns:
                None
        """
        # load the audio file
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

        # normalize audio
        #audio = librosa.util.normalize(audio)

        # load the MIDI file
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # Get the length of the midi file in seconds
        midi_length_secs = midi.get_end_time()

        # Get the length of the audio file in seconds
        audio_length_secs = librosa.get_duration(y=audio, sr=sample_rate)

        self.midi_length_secs = midi_length_secs
        self.audio_length_secs = audio_length_secs

        # pad audio if MIDI is longer than audio
        if midi_length_secs > audio_length_secs:
            # we use ceil because 3.1 samples for example is greater than 3 and has to be an integer
            # so we take it to 4.
            padding = int(np.ceil(midi_length_secs * sample_rate).item()) - audio.shape[0]
            check = 5*sample_rate # 5 seconds of padding
            if padding >= check:
                raise ValueError("Padding is too large, check your audio and MIDI files!")
            audio = np.pad(audio, (0, padding), mode='constant')

        self.audio = audio
        self.midi = midi
        self.sample_rate = sample_rate
        self.min_frame_secs = min_frame_secs
        self.max_frame_secs = max_frame_secs
        self.frame_rate = frame_rate

        # we want to deal with events in the following order:
        # PEDAL_DOWN, PEDAL_UP, ONSET, OFFSET
        self.PEDAL_DOWN = 0
        self.PEDAL_UP = 1
        self.ONSET = 2
        self.OFFSET = 3

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.num_pitches = max_pitch - min_pitch + 1
        self.onset_length = onset_length
        self.offset_length = offset_length

        if not ignore_duration:
            self.split_windows = self._get_split_windows()
        else:
            self.split_windows = [(0, midi_length_secs)]
            self.max_frame_secs = midi_length_secs

        self.sustained_midi = self._pedal_extend(CC_SUSTAIN=64)  # Extend the MIDI with sustain events
        self.audio_frames = []
        self.label_frames = []
        for i, (start, end) in enumerate(self.split_windows):
            # is this necessary? Given that we already checked 
            # the splits in the get_split_windows method?
            if end - start < self.min_frame_secs:
                continue

            # Get the label for the split window
            label = self._get_label(self.sustained_midi, self.max_frame_secs, start, end, self.frame_rate)
            self.label_frames.append(label)

            # Get the audio frame for the split window
            start_chunk = int(start * self.sample_rate)
            end_chunk = int((end - start) * self.sample_rate) + start_chunk
            audio_frame = self.audio[start_chunk:end_chunk]

            if len(audio_frame) < (self.max_frame_secs * self.sample_rate):
                # If the audio frame is shorter than the max frame duration,
                # we can pad it with zeros
                audio_frame = np.pad(audio_frame, \
                (0, int(self.max_frame_secs * self.sample_rate) - len(audio_frame)), \
                mode='constant')
            
            self.audio_frames.append(audio_frame)
        
        assert len(self.audio_frames) == len(self.label_frames), \
            "Number of audio frames and label frames must be equal"
    
    def __iter__(self):
        """
            Returns an iterator over the audio frames and label frames.
        """
        self.index = 0
        return self   
    
    def __next__(self) -> np.ndarray:
        if self.index < len(self.audio_frames):
            framed_audio =  self.audio_frames[self.index]
            label = self.label_frames[self.index]
            self.index += 1
            return framed_audio, label
        raise StopIteration

    def __len__(self) -> int:
        """
            Returns the number of framed events.
        """
        return len(self.audio_frames)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict]:
        """
            Get the audio frame and label frame at the given index.
            
            Args:
                index (int): Index of the audio frame and label frame
            
            Returns:
                tuple[np.ndarray, dict]: Audio frame and label frame at the given index
        """
        return self.audio_frames[index], self.label_frames[index]


    def _get_label(self, midi, duration: float, start: float, end: float, frame_rate: float, c=5) -> dict:
        """
            Get the label for the given MIDI file and duration.
            Args:
                midi (pretty_midi.PrettyMIDI): MIDI file
                duration (float): Duration of the label in seconds
                start (float): Start time of the label in seconds
                end (float): End time of the label in seconds
                frame_rate (float): Frame rate for the label
                c (int): Weighting parameter for the label weights
            Returns:
                dict: Label dictionary containing onset, offset, velocity and frame information
        """

        # Create the label dictionary
        label = {
            "label_onsets": None,
            "label_velocities": None,
            "label_frames": None,
            "label_weights": None
        }

        # Get the maximum velocity for this MIDI file
        for instrument in midi.instruments:
            if not instrument.is_drum:
                max_velocity = max(note.velocity for note in instrument.notes)
                

        # This is what the magenta implementation does
        # It appears different approaches are used to get the num of frames
        # Kong uses python's round, hFT uses (+0.5) and then adds 1 to the result
        num_frames = int(duration * frame_rate + 1) 

        # Create all the labels
        label_onsets = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
        label_offsets = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
        label_velocities = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
        label_frames = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
        label_weights = np.ones((num_frames, self.num_pitches), dtype=np.float32)

        # onset and offset length in secs
        onset_length_secs = self.onset_length / 1000.0
        offset_length_secs = self.offset_length / 1000.0

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 
                    start_frame = int((note.start - start) * frame_rate)
                    end_frame = int(np.ceil((note.end - start) * frame_rate).item())

                    # I used round here due to Jongwook's implementation
                    # I can come back to this later
                    onset_frame_length = int(round(onset_length_secs * frame_rate))
                    offset_frame_length = int(round(offset_length_secs * frame_rate))

                    onset_start = max(0, start_frame)
                    onset_end = min(num_frames, onset_start + onset_frame_length)
                    frame_end = min(num_frames, end_frame)
                    offset_end = min(num_frames, end_frame + offset_frame_length)

                    # Store information in the labels
                    label_onsets[onset_start:onset_end, \
                                 note.pitch - self.min_pitch] = 1.0
                    
                    label_frames[onset_start:frame_end, \
                                 note.pitch - self.min_pitch] = 1.0
                    
                    label_velocities[start_frame:frame_end, \
                                     note.pitch - self.min_pitch] = note.velocity / max_velocity
                    
                    label_offsets[end_frame:offset_end, \
                                  note.pitch - self.min_pitch] = 1.0
                    
                    label_weights[onset_start:onset_end, \
                                    note.pitch - self.min_pitch] = c
                    
                    label_weights[onset_end:frame_end, \
                                    note.pitch - self.min_pitch] = [
                                        c / delta for delta in range(1, frame_end - onset_end + 1)
                                    ]
                    
        # Update the label dictionary
        label["label_onsets"] = label_onsets
        label["label_velocities"] = label_velocities
        label["label_frames"] = label_frames
        label["label_weights"] = label_weights
        return label

    def _get_split_windows(self) -> list[tuple[float, float]]:
        """
            Gets the split windows based on the min and max frame durations.
            
            Returns:
                list[tuple[float, float]]: List of tuples with start and end times of the split windows
        """
        return self._extract_split_windows()

    def _extract_split_windows(self):
        """
            Extract split windows based on the min and max frame durations.
            If min_frame_secs == max_frame_secs, it will split the audio into fixed-size frames.
            If max_frame_secs > 0, it will use the get_splits method to split appropriately.
            If max_frame_secs <= 0, it will use the whole file as a single frame.
        """
        if self.min_frame_secs == self.max_frame_secs:
            split_points = range(0, self.midi_length_secs, self.min_frame_secs)
            split_windows = list(zip(split_points[:-1], split_points[1:]))
        elif self.max_frame_secs > 0:
            split_windows = self._extract_splits()
        else: 
            # Use the whole file
            split_points = [0, self.midi_length_secs]
            split_windows = list(zip(split_points[:-1], split_points[1:]))
        return split_windows

    def _extract_splits(self):
        """
            Extract split windows based on the min and max frame durations.
        """
        # check if there are note events in the MIDI file
        note_flag = False
        for instrument in self.midi.instruments:
            if not instrument.is_drum:
                # if there is at least one instrument with notes, we can proceed
                if instrument.notes:
                    note_flag = True
                    break
        
        if not note_flag:
            return []
        
        extended_midi = self._pedal_extend(CC_SUSTAIN=64)  # Get the extended MIDI file with sustain events

        # Get the gaps in the sustained and non-sustained MIDI events
        gaps_sus = self._get_gaps(extended_midi)
        gaps_non_sus = self._get_gaps(self.midi)

        # Get the start and end times of the gaps
        # We will use these to determine the split points
        gaps_sus_start = [gap[0] for gap in gaps_sus]
        gaps_sus_end = [gap[1] for gap in gaps_sus]
        gaps_non_sus_start = [gap[0] for gap in gaps_non_sus]
        gaps_non_sus_end = [gap[1] for gap in gaps_non_sus]

        splits = [0.] # Start with the first split at 0 seconds
        end_time = self.midi_length_secs

        # While we still have at least max_frame_secs left in the MIDI file
        # we will keep splitting
        while (end_time - splits[-1]) > self.max_frame_secs:
            split_point = splits[-1] + self.max_frame_secs

            # Check if the split point is within a gap in the sustained MIDI events
            # rather than naively going through all the gaps, we can use binary search
            idx = bisect.bisect_right(gaps_sus_end, split_point) # the end time of the gap just bigger than the split point

            if idx < len(gaps_sus_start) and split_point > gaps_sus_start[idx]:
                # this means that the split point is within a gap in the sustained MIDI events
                # so we can add it to the splits
                splits.append(split_point)
            elif idx == 0 or gaps_sus_start[idx - 1] <= splits[-1] + self.min_frame_secs:
                # if the split point is not within a gap in the sustained MIDI events,
                # and the proposed split point (mid of the last gap before split_point) 
                # will not create a frame greater than the min_frame_secs,
                # then we come here and consider the non-sustained MIDI events

                # find the index of the end time of the gap just bigger than the split point
                idx = bisect.bisect_right(gaps_non_sus_end, split_point)

                if idx < len(gaps_non_sus_start) and split_point > gaps_non_sus_start[idx]:
                    # if the split point is within a gap in the non-sustained MIDI events,
                    # we lookg for a zero crossing in the audio signal
                    zero_cross_start = gaps_non_sus_start[idx]
                    zero_cross_end = split_point
                    last_zero_cross = self._last_zero_crossing(self.audio, int(zero_cross_start * self.sample_rate), \
                        int(np.ceil(zero_cross_end * self.sample_rate).item())) # we use ceil so we avoid indexing errors
                    
                    if last_zero_cross:
                        last_zero_cross = float(last_zero_cross) / self.sample_rate
                        splits.append(last_zero_cross)
                    else:
                        # if there is no zero crossing, we can just use the split point
                        splits.append(split_point)
                else:
                    # This means we can't find any good place to make our cuts
                    # so we go through the entire range between [min_frame_secs, max_frame_secs]
                    # and look for a zero crossing
                    zero_cross_start = int(np.ceil((splits[-1] + self.min_frame_secs)*self.sample_rate).item()) + 1
                    zero_cross_end = zero_cross_start + (self.max_frame_secs - self.min_frame_secs) * self.sample_rate
                    zero_cross_end = int(np.ceil(zero_cross_end).item()) # not incl
                    last_zero_cross = self._last_zero_crossing(self.audio, zero_cross_start, zero_cross_end)

                    if last_zero_cross:
                        last_zero_cross = float(last_zero_cross) / self.sample_rate
                        splits.append(last_zero_cross)
                    else:
                        # if there is no zero crossing, we can just use the split point
                        splits.append(split_point)
            else:
                # if the split point is not within a gap in the sustained MIDI events,
                # and the proposed split point (mid of the last gap before split_point) 
                # will create a frame greater than the min_frame_secs,
                # then we can use the mean of the last gap before the split point
                splits.append(np.mean([gaps_sus_start[idx - 1], gaps_sus_end[idx - 1]]))
        
        # validate the splits
        if not self._validate_splits(splits):
            raise AssertionError("Invalid splits generated. Please check the MIDI file and the min/max frame durations.")

        return list(zip(splits[:-1], splits[1:]))
    

    def _validate_splits(self, splits: list[float]) -> bool:
        """
            Validate the splits to ensure they meet the criteria for min and max frame durations.

            Args:
                splits (list[float]): List of split points in seconds

            Returns:
                bool: True if the splits are valid, False otherwise
        """
        for i, j in zip(splits[:-1], splits[1:]):
            if j <= i:
                # split must be strictly increasing
                return False
       
            if j - i > self.max_frame_secs + 1e-8:
                # split must not exceed the max frame duration
                return False
            
            if j < self.midi_length_secs:
                if j - i < self.min_frame_secs - 1e-8:
                    # split must not be less than the min frame duration
                    return False
        if self.midi_length_secs - splits[-1] >= self.max_frame_secs:
            # last split must not exceed the max frame duration
            return False
        return True

    def _pedal_extend(self, CC_SUSTAIN=64):
        """
            Extend the sustain events in the MIDI file.
            
            Args:
                CC_SUSTAIN (int): MIDI control change number for sustain pedal. Default is 64.
            
            Returns:
                midi_copy (pretty_midi.PrettyMIDI): MIDI file with extended sustain events.
        """
        # make a copy of the MIDI file
        midi_copy = copy.deepcopy(self.midi)

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
        
        # save this MIDI file (debugging purposes)
        #midi_copy.write("test.mid")
        return midi_copy

    def _get_gaps(self, midi: pretty_midi.PrettyMIDI):
        """
            Get the gaps in the MIDI events where no notes are played
            or active.
            
            Args:
                midi (pretty_midi.PrettyMIDI): MIDI file with extended sustain events
            
            Returns:
                gaps (list): List of tuples with start and end times of the gaps
        """
        # A gap is defined as a time interval between the last active note end and the 
        # next active note start
        # first, let's extract the onset events
        onsets = []
        for instrument in midi.instruments: 
            if not instrument.is_drum:
                for note in instrument.notes:
                    onsets.append(note.start)
        
        # extract the offsets
        offsets = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    offsets.append(note.end)
        
        # sort the onsets and offsets
        onsets = sorted(onsets, reverse=True)
        offsets = sorted(offsets, reverse=True)

        # Again, a gap is defined as a time interval between the last active note end and the next
        # active note start
        num_active_notes = 0 # Tracks the current number of active notes
        gaps = []

        if onsets[-1] > 0:
            # if the first onset is greater than 0, we can add a gap from 0 to the first onset
            gaps.append(0.)
            gaps.append(onsets[-1])

        # remove the first onset and start tracking the active notes
        onsets.pop()  # remove the first onset
        num_active_notes += 1  # we have one active note now

        while onsets or offsets:
            if onsets and (onsets[-1] < offsets[-1]):
                # if the next onset is before the next offset
                if num_active_notes == 0:
                    # if there are no active notes, we can add a gap
                    gaps.append(onsets[-1])
                
                # if some notes are still active, then no gap, so we 
                # increment the count
                num_active_notes += 1
                onsets.pop()  # remove the onset
            else:
                # if the next offset is before the next onset
                # then a note has ended
                num_active_notes -= 1  # decrement the count of active notes
                if num_active_notes == 0:
                    gaps.append(offsets[-1])
                offsets.pop()  # remove the offset
            
        # check if there is a gap between the last event and the end of the MIDI file
        if gaps[-1] < self.midi_length_secs:
            gaps.append(self.midi_length_secs)
        else:
            # if the last gap is the same as the end of the MIDI file, we can remove it
            gaps.pop()
        
        # We need to ensure that the gaps are in pairs
        if len(gaps) % 2 != 0:
            raise ValueError("Gaps list is not in pairs, something went wrong!")

        # Convert the gaps to tuples
        gaps = list(zip(gaps[:-1], gaps[1:]))
        return gaps

    def _last_zero_crossing(self, audio: np.ndarray, start: int, end: int) -> int | None:
        
        # a zero crossing is defined as a point where the audio signal changes sign
        # we find the places where the signal goes from +ve to non-positive
        # or from -ve to non-negative
        pos = audio[start:end] > 0 # positive samples
        neg = audio[start:end] < 0 # negative samples
        nonneg = audio[start:end] >= 0 # non-negative samples
        nonpos = audio[start:end] <= 0 # non-positive samples

        # compare positive values in the pos array to the non-positive values in the nonpos array
        # so each value in pos is compared to the next value in nonpos
        compare_pos = np.logical_and(pos[:-1], nonpos[1:])

        # compare negative values in the neg array to the non-negative values in the nonneg array
        # so each value in neg is compared to the next value in nonneg
        compare_neg = np.logical_and(neg[:-1], nonneg[1:])

        zero_crossings = np.logical_or(compare_pos, compare_neg).nonzero()[0]
        if zero_crossings.size == 0:
            return None
        
        last_zero_cross = zero_crossings[-1] + start
        
        return last_zero_cross


class _OAFMode(_BaseMode):
    """
        OAFMode is a class that implements the Onsets and Frames (OAF) mode of operation.
        It extracts audio and MIDI files from the dataset and generates framed audio and MIDI events.
        It uses the OAFFramedEvents class to generate the framed events.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self._oaf(config)
    
    def _oaf(self, config):
        # Check if the path is None
        if self.path is None or not Path(self.path).exists():
            raise ValueError(f"Path {self.path} does not exist!")

        if config["frame_rate"] is None:
            raise ValueError("frame_rate must be set!")
        
        self.frame_rate = config["frame_rate"]

        if config["sample_rate"] is None:
            raise ValueError("sample_rate must be set!")
        self.sample_rate = config["sample_rate"]
        self.feature = config["feature"]
        self.feature_params = config["feature_params"]

        # Set feature params.
        if not self.feature_params:
            self.feature = "mel"
            self.feature_params = {"n_mels": 229, "mel_n_fft": 2048}
        
        # Perform the extraction
        self._extract()

    def _extract(self):
        """
            Extract audio and MIDI files from the dataset.
            Note that the original paper specifies that 20 second splits are used.
            If notes are active where the split occurs, a zero-crossing of the
            audio signal is used to determine the split point.
        """
        
        # Implement the extraction logic here
        if self.dataset_name == "maps":
            train_files, val_files, test_files = self._get_maps_train_val_test()
            train_files = train_files + val_files
            splits = ["train", "test"]
            splits_data = [train_files, test_files]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported!")
        
        print(f"Total train files found: {len(train_files)}, total test files found: {len(test_files)}")

        # create the directory for the splits
        for split in splits:
            split_path = Path(f"./{self.save_name}/{split}")
            split_path.mkdir(parents=True, exist_ok=True)

        for idx, split in tqdm(enumerate(splits), desc="Processing splits"):
            split_files = splits_data[idx]

            for audio_file, midi_file in tqdm(split_files, desc=f"Processing audio and MIDI files for {split} split..."):
                assert audio_file.exists(), f"{audio_file} does not exist"
                assert midi_file.exists(), f"{midi_file} does not exist"

                if split != "test":
                    framed_events = _OafFramedEvents(
                        audio_path=audio_file,
                        midi_path=midi_file,
                        min_frame_secs=self.config["min_frame_secs"],
                        max_frame_secs=self.config["max_frame_secs"],
                        sample_rate=self.config["sample_rate"],
                        frame_rate=self.config["frame_rate"],
                        min_pitch=self.config["min_pitch"],
                        max_pitch=self.config["max_pitch"],
                        onset_length=self.config["onset_length"],
                        offset_length=self.config["offset_length"]
                    )
                else:
                    framed_events = _OafFramedEvents(
                        audio_path=audio_file,
                        midi_path=midi_file,
                        min_frame_secs=self.config["min_frame_secs"],
                        max_frame_secs=self.config["max_frame_secs"],
                        sample_rate=self.config["sample_rate"],
                        frame_rate=self.config["frame_rate"],
                        min_pitch=self.config["min_pitch"],
                        max_pitch=self.config["max_pitch"],
                        onset_length=self.config["onset_length"],
                        offset_length=self.config["offset_length"],
                        ignore_duration=True  # For test split, we ignore duration
                    )

                for idx_event, framed_event in enumerate(framed_events):
                    store_dict = {'audio': None, 'feature': None}  
                    audio_frame, label = framed_event

                    feature = self._get_feature(audio_frame, self.feature, self.feature_params)
                    feature = feature.T # (time, embedding)

                    assert feature.shape[0] == label["label_frames"].shape[0], \
                    f"Feature {feature.shape} and label {label['label_frames'].shape} shapes do not match!"
                    
                    if self.dataset_name != "slakh":
                        store_path = f"./{self.save_name}/{split}/{str(audio_file.stem)}_{idx_event}.npz"
                    else:
                        # For slakh, we need to get the track name
                        # from the audio file path
                        track_name = audio_file.parent.parent.stem
                        store_path = f"./{self.save_name}/{split}/{track_name}_{str(audio_file.stem)}_{idx_event}.npz"

                    store_dict['audio'] = audio_frame
                    store_dict['feature'] = feature

                    for key in label.keys():
                        store_dict[key] = label[key]
                    
                    np.savez(store_path, **store_dict)

        print(f"Extraction complete! Total files extracted: {len(train_files) + len(test_files)}")
            

    def _get_feature(self, audio : np.ndarray, feature: str, feature_params: dict):
        """
            Get the feature for a given audio segment
            Args:
                audio (np.ndarray): Audio segment
                feature (str): Feature_type to extract
                feature_params (dict): Parameters for the feature extraction

            Returns:
                feature (np.ndarray): Feature for the audio segment
        """

        # Check if the feature is supported
        if feature not in SUPPORTED_FEATURES:
            raise ValueError(f"Feature {feature} not supported! \
                    Supported features are: {SUPPORTED_FEATURES}")

        # Create the HandcraftedFeatures object
        if feature in SUPPORTED_FEATURES:
            hf = HandcraftedFeatures(sample_rate=self.sample_rate, \
                    window_size=self.config["max_frame_secs"], frame_rate=self.frame_rate)
            
            if feature == "mel":
                n_mels = feature_params["n_mels"]
                n_fft = feature_params["mel_n_fft"]
                return hf.compute_mel(audio, n_mels=n_mels, n_fft=n_fft, \
                                      hop_length=feature_params.get("hop_length", None))
            elif feature == "cqt":
                bins_per_octave = feature_params["cqt_bins_oct"]
                num_octaves = feature_params["cqt_num_octaves"]
                return hf.compute_cqt(audio, bins_per_octave=bins_per_octave, \
                        num_octaves=num_octaves, hop_length=feature_params.get("hop_length", None))
