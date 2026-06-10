from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pretty_midi
from snap2midi.extractor.utils import HandcraftedFeatures, SUPPORTED_FEATURES
from snap2midi.extractor.utils.framed_signal import FramedAudio
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
        self.frame_rate = config["frame_rate"]
        self.feature = config["feature"]
        self.feature_params = config["feature_params"]
        self.min_pitch = config["min_pitch"]
        self.max_pitch = config["max_pitch"]
        self.num_pitches = self.max_pitch - self.min_pitch + 1
        self.onset_frame_length = config["onset_frame_length"] # unit is in frames
        self.offset_frame_length = config["offset_frame_length"] # unit is in frames

        # Set feature params.
        if not self.feature_params:
            self.feature = "mel"
            self.feature_params = {"n_mels": 229, "mel_n_fft": 2048, "hop_length": 512}

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
            hop_length = self.feature_params["hop_length"]
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

    
    # def _extract(self, split_files: list, split: str):
    #     """
    #         Extract audio and MIDI files from the dataset.
    #         We specifically follow the extraction params
    #         used in Jongwook's OAF implementation.
    #     """
    #     # parse the (audio_file, midi_file) in split_files
    #     for audio_file, midi_file in tqdm(split_files, \
    #                     desc=f"Processing audio and MIDI files for {split} split..."):
    #         assert audio_file.exists(), f"{audio_file} does not exist"
    #         assert midi_file.exists(), f"{midi_file} does not exist"
    #         frame_size = self.config["window_size"]
    #         audio_frames = FramedAudio(audio_file, hop_size=frame_size, frame_size=frame_size, \
    #             sample_rate=self.sample_rate)
            
    #         # load MIDI
    #         midi_obj = pretty_midi.PrettyMIDI(midi_file)
    #         if self.extend_pedal: 
    #             midi_obj = self._pedal_extend(midi_obj)
    
    #         for i, frame in enumerate(audio_frames):
    #             store_dict = {'audio': None, 'feature': None} 
    #             feature = self._get_feature(frame, self.feature, self.feature_params)
    #             start = i/self.frame_rate
    #             end = start + frame_size
    #             label = self._get_label(midi_obj, start, end, self.frame_rate)
                
    #             feature = feature.T # (time, embedding)
    #             feature = feature[:label["label_frames"].shape[0], :]

    #             assert feature.shape[0] == label["label_frames"].shape[0], \
    #             f"Feature {feature.shape} and label {label['label_frames'].shape} shapes do not match!"

    #             if self.dataset_name != "slakh":
    #                 store_path = f"./{self.save_name}/{split}/{str(audio_file.stem)}_{i}.npz"
    #             else:
    #                 # For slakh, we need to get the track name
    #                 # from the audio file path
    #                 track_name = audio_file.parent.parent.stem
    #                 store_path = f"./{self.save_name}/{split}/{track_name}_{str(audio_file.stem)}_{i}.npz"

    #             store_dict['audio'] = frame
    #             store_dict['feature'] = feature

    #             for key in label.keys():
    #                 store_dict[key] = label[key]
    #             np.savez(store_path, **store_dict)

    # def _get_label(self, midi: pretty_midi.PrettyMIDI, start: float, end: float, frame_rate: float) -> dict:
    #     """
    #         Get the label for the given MIDI file and duration.

    #         Args
    #         ------
    #             midi (pretty_midi.PrettyMIDI): PrettyMIDI object
    #             start (float): Start time in seconds of the window
    #             end (float): End time in seconds of the window
    #             frame_rate (float): Frame rate for the frames/windows

    #         Returns
    #         --------
    #             dict: Label dictionary containing onset, offset, velocity and frame information
    #     """

    #     # Create the label dictionary
    #     label = {
    #         "label_onsets": None,
    #         "label_velocities": None,
    #         "label_frames": None,
    #         "label_weights": None
    #     }

    #     # This is what the magenta implementation does
    #     # It appears different approaches are used to get the num of frames
    #     # Kong uses python's round, hFT uses (+0.5) and then adds 1 to the result
    #     duration = end - start
    #     num_frames = int(frame_rate * duration)

    #     # Create all the labels
    #     label_onsets = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
    #     label_offsets = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
    #     label_velocities = np.zeros((num_frames, self.num_pitches), dtype=np.float32)
    #     label_frames = np.zeros((num_frames, self.num_pitches), dtype=np.float32)

    #     # Get the maximum velocity for this MIDI file
    #     for instrument in midi.instruments:
    #         if not instrument.is_drum:
    #             max_velocity = max(note.velocity for note in instrument.notes)

    #     for instrument in midi.instruments:
    #         if not instrument.is_drum:
    #             for note in instrument.notes:
    #                 if note.start >= end and note.end <= start:
    #                     continue

    #                 start_frame = int((note.start - start) * frame_rate)
    #                 end_frame = int(np.ceil((note.end - start) * frame_rate).item())

    #                 onset_start = max(0, start_frame)
    #                 onset_end = min(num_frames, onset_start + self.onset_frame_length)
    #                 frame_end = min(num_frames, end_frame)
    #                 offset_end = min(num_frames, end_frame + self.offset_frame_length)
                    
    #                 # Store information in the labels
    #                 label_onsets[onset_start:onset_end, \
    #                              note.pitch - self.min_pitch] = 1.0
                    
    #                 label_frames[onset_start:frame_end, \
    #                              note.pitch - self.min_pitch] = 1.0

    #                 label_velocities[onset_start:frame_end, \
    #                                  note.pitch - self.min_pitch] = note.velocity / max_velocity
                    
    #                 label_offsets[end_frame:offset_end, \
    #                               note.pitch - self.min_pitch] = 1.0
                    
    #     # Update the label dictionary
    #     label["label_onsets"] = label_onsets
    #     label["label_offsets"] = label_offsets
    #     label["label_velocities"] = label_velocities
    #     label["label_frames"] = label_frames
    #     return label
    
    # def _get_feature(self, audio : np.ndarray, feature: str, feature_params: dict):
    #     """
    #         Get the feature for a given audio segment

    #         Args
    #         -----
    #             audio (np.ndarray): Audio segment
    #             feature (str): Feature_type to extract
    #             feature_params (dict): Parameters for the feature extraction

    #         Returns
    #         --------
    #             feature (np.ndarray): Feature for the audio segment
    #     """

    #     # Check if the feature is supported
    #     if feature not in SUPPORTED_FEATURES:
    #         raise ValueError(f"Feature {feature} not supported! \
    #                 Supported features are: {SUPPORTED_FEATURES}")

    #     # Create the HandcraftedFeatures object
    #     if feature in SUPPORTED_FEATURES:
    #         hf = HandcraftedFeatures(sample_rate=self.sample_rate, \
    #                 window_size=self.config["window_size"], frame_rate=self.frame_rate)
            
    #         if feature == "mel":
    #             n_mels = feature_params["n_mels"]
    #             n_fft = feature_params["mel_n_fft"]
    #             fmin = self.config["fmin"]
    #             fmax = self.config["fmax"]
    #             htk = self.config["use_htk"]
    #             return hf.compute_mel(audio, n_mels=n_mels, n_fft=n_fft, \
    #                 hop_length=feature_params.get("hop_length", None), \
    #                 fmin=fmin, fmax=fmax, htk=htk)
    #         elif feature == "cqt":
    #             bins_per_octave = feature_params["cqt_bins_oct"]
    #             num_octaves = feature_params["cqt_num_octaves"]
    #             return hf.compute_cqt(audio, bins_per_octave=bins_per_octave, \
    #                     num_octaves=num_octaves, hop_length=feature_params.get("hop_length", None))
    
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
