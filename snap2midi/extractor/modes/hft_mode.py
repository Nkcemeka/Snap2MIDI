from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import pretty_midi
import torchaudio
import json
import os

class Message:
    """ 
        Initializes a MIDI message. This is an utility class
        to help with MIDI message processing using pretty_midi
        instead of MIDO used in the hfTransformer implementation
        by Sony.
    """
    def __init__(self, time, type, note, velocity=0):
        """
            Args
            ------
                time (float): The time of the message in seconds.
                type (str): The type of the message (e.g., 'note_on', 'note_off').
                note (int): The MIDI note number.
                velocity (int, optional): The velocity of the note. Defaults to 0.
        """
        self.time = time
        self.type = type
        self.note = note
        self.velocity = velocity
    
    def __str__(self):
        return f"Message(time={self.time}, type={self.type}, note={self.note}, velocity={self.velocity})"

    def __repr__(self):
        return self.__str__()

class _HFTMode(_BaseMode):
    """
        HFTMode is a class that extracts audio and MIDI files from the dataset
        for training the hFT-Transformer model by Sony.
    """
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._hft(self.config)
    
    def _hft(self, config: dict):
        """
            Extract the features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.

            Args
            -----
                config (dict): Configuration dictionary containing the parameters   

            Returns
            --------
                None
        """
        self._extract_hft(config)
        self._collate_hft(config)
    
    def _collate_hft(self, config: dict):
        """
            Collate the features and labels for the hFT-Transformer model by Sony.
            This method processes the features and labels extracted from the audio and MIDI files
            and stores them in a single npz file for each split (train, val, test).

            Args
            -----
                config (dict): Configuration dictionary containing the parameters

            Returns
            ------
                None
        """
        # Process the train, val and test splits
        self._collate_hft_split(config, "train")
        self._collate_hft_split(config, "val")
        self._collate_hft_split(config, "test")
    
    def _collate_hft_split(self, config: dict, split: str):
        files_all = sorted(Path(f"{self.save_name}/audio/{split}").rglob("*.npz"))
        files_all_div = []
        n_divs = config[f"n_div_{split}"]

        for div in range(n_divs):
            files_all_div.append([])

        for i, f in enumerate(files_all):
            div = i % n_divs
            files_all_div[div].append(f)

        for div in range(n_divs):
            self._collate_hft_split_div(files_all_div[div], config, split, div)
    
    def _stride(self, config: dict) -> int:
        """
            Frames between the start of one track and the start of the next.

            Every array in a division -- audio, idx and all four label arrays --
            has to advance by this same amount or they drift apart, so it lives
            in one place. The old code advanced the feature slab by the track's
            own frame count while advancing the labels by max(feature, label),
            which silently offset the two whenever a MIDI file outlived its audio.
        """
        return config['input']['margin_f'] + config['input']['num_frame'] - 1

    def _fill_label_slab(self, files_all: list, num_frame_list: list, config: dict,
                         path: str, key: str, total_num_frame: int, dtype) -> None:
        """
            Write one label array for a division, laid out track after track.

            Uses open_memmap so the array is built on disk rather than in RAM --
            a full division of onsets is gigabytes and the old code held it all
            in memory before saving.

            Args
            -----
                files_all (list): Per-track npz files for this division
                num_frame_list (list): Frame count of each track, same order
                config (dict): Configuration dictionary containing the parameters
                path (str): Destination .npy path
                key (str): Key to read out of each per-track npz
                total_num_frame (int): Height of the division's arrays
                dtype: dtype of the destination array
        """
        slab = np.lib.format.open_memmap(
            path, mode='w+', dtype=dtype,
            shape=(total_num_frame, config['midi']['num_note']))

        loc_d = config['input']['margin_b']
        for i, each in enumerate(files_all):
            npz_file = np.load(each, allow_pickle=True)
            label = npz_file[key]
            slab[loc_d:loc_d + len(label), :] = label
            del npz_file
            loc_d += num_frame_list[i] + self._stride(config)

        slab.flush()
        del slab

    def _collate_hft_split_div(self, files_all: list, config: dict, split: str, div):
        """
            Does the collation for each split (train/test/validation)

            Args
            -----
                files_all (list): List of files for the division under consideration
                config (dict): Configuration dictionary containing the parameters
                split (str): training, validation or teat split.
                div (int): split division number
        """
        num_frame_list = [] # stores the number of frames for each file
        num_frame_audio_list = [] # frames the audio alone accounts for

        # the total number of frames read including the margins
        # we are trying to get a window of Nctx from L frames from the actual feature
        # with a hop size of 1. Total number of frames becomes L - Nctx + 1. Since this does
        # not add up to L, we add Nctx - 1 to the original and so we will process get L + Nctx - 1.
        # We also add the margin at the beginning and end of the feature
        # see explanation below. total_num_frame computes this total including the margins
        # The reason for this is explained due to the __getitem__ method we will make in the 
        # dataset class.
        total_num_frame = config["input"]["margin_b"]

        # the total or actual number of frames read from the features
        total_num_frame_idx = 0 

        for i, each in enumerate(files_all):
            # load the npz file
            npz_file = np.load(each, allow_pickle=True)
            num_frame_feature = int(npz_file['num_frames']) # frames the audio will produce
            num_frame_label = len(npz_file['label_frames']) # number of frames for MPE
            del npz_file # delete npz file to free memory

            # Get the number of frames based on the maximum for the feature and label
            num_frame = max(num_frame_feature, num_frame_label)
            num_frame_list.append(num_frame)
            num_frame_audio_list.append(num_frame_feature)

            # So, here we go: we have num_frame (L) for the number of features
            # But we want config['input']['num_frame'] (Nctx) instead. 
            # the expec. no of frames = (L - Nctx)/1 + 1
            # which is L - Nctx + 1. To get exactly L. no of frames, we add Nctx - 1
            # This gives us L + Nctx - 1; now we need some backward and forward margin
            # This leads to L + Nctx - 1 + Mf. Note that Mf for file i is Mb for file
            # i - 1;
            # Hence, Mb (start pad)| L1 + Nctx - 1 | Mf | L2 + Nctx - 1 | Mf etc.... 

            # We are interested in getting L frames because our idx in the __getitem__ can give
            # us the starting point of our window as any frame of L frames...
            total_num_frame += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
            total_num_frame_idx += num_frame

        # Where the MIDI outlives the audio, the division carries active labels
        # over frames the slab has no samples for -- the model is asked to
        # transcribe notes from silence. This was equally true of the old feature
        # slab, so it is not new, but it is invisible and worth saying out loud:
        # 0 of 139 train and 0 of 60 test MAPS tracks hit it, other datasets may.
        overrun = [(f.name, nl - na) for f, na, nl
                   in zip(files_all, num_frame_audio_list, num_frame_list) if nl > na]
        if overrun:
            worst = max(overrun, key=lambda x: x[1])
            hop_ms = 1000 * config['feature']['hop_sample'] / config['feature']['sr']
            print(f"  note: {len(overrun)}/{len(files_all)} {split} tracks have MIDI "
                  f"extending past the audio; labels there sit over silence. "
                  f"Worst: {worst[0]} by {worst[1]} frames ({worst[1] * hop_ms / 1000:.1f} s)")


        # Everything below is written as .npy rather than .npz. A npz is a zip
        # archive and cannot be memory-mapped, and the dataset needs to map these
        # so that dataloader workers share one copy instead of each loading its
        # own (which is where hFT's per-worker memory blow-up comes from).
        suffix = str(div).zfill(3) + ".npy"

        # dataset_idx (helps us keep track of the location of the actual features)
        print(f"Processing dataset_idx for {split}...")
        dataset_idx = np.zeros(total_num_frame_idx, dtype=np.int32)
        loc_i = 0
        loc_d = config['input']['margin_b']
        for i, each in enumerate(files_all):
            # Tells us where our segemnt L ended up in.
            # so for each raw frame in the unpadded, where did it
            # end in the padded...
            num_frame = num_frame_list[i]
            dataset_idx[loc_i:loc_i + num_frame] = np.arange(loc_d, loc_d + num_frame)
            loc_i += num_frame
            loc_d += num_frame + self._stride(config)

        # store dataset_idx in the save_name directory. Still in frame units --
        # nothing about the audio slab changes what an index means.
        np.save(f"{self.save_name}/idx/{split}/dataset_idx" + suffix, dataset_idx)
        del dataset_idx # delete to free memory

        ## Process the audio
        # The slab holds total_num_frame frames' worth of samples, plus fft_bins
        # // 2 of headroom at each end. That headroom is what lets the dataset
        # read an item as one flat slice: the samples a centred frame f needs
        # start at f * hop_sample - fft_bins // 2, which would run negative at
        # f = 0 without it. So sample position of frame f is f * hop + audio_pad,
        # and an item starting at frame f reads slab[f * hop : f * hop + length].
        print(f"Processing audio for {split}...")
        hop = config['feature']['hop_sample']
        audio_pad = config['feature']['fft_bins'] // 2
        stride_samples = self._stride(config) * hop

        dataset_audio = np.lib.format.open_memmap(
            f"{self.save_name}/audio/{split}/dataset_audio" + suffix,
            mode='w+', dtype=np.float32, shape=(total_num_frame * hop + 2 * audio_pad,))

        loc_s = config['input']['margin_b'] * hop + audio_pad
        for i, each in enumerate(files_all):
            num_frame = num_frame_list[i]

            # load the npz file and store the audio at the right location
            npz_file = np.load(each, allow_pickle=True)
            audio = npz_file['audio']

            # The layout is driven by the num_frames written at extraction, and
            # the slab check in scripts/verify_hft_slab_layout.py cannot catch a
            # wrong value there -- it rebuilds its reference from the same
            # layout, so both sides would move together. Check it here instead,
            # against the audio actually being written. A track of F frames is
            # always under F * hop samples long, so this also rules out one
            # track overwriting the next.
            # Raised rather than asserted: these guard against silent data
            # corruption, and `python -O` strips assert statements.
            if self._num_frames_hft(audio, config) != num_frame_audio_list[i]:
                raise ValueError(
                    f"{each.name}: {len(audio)} samples give "
                    f"{self._num_frames_hft(audio, config)} frames but num_frames "
                    f"says {num_frame_audio_list[i]}")
            if len(audio) >= num_frame * hop:
                raise ValueError(
                    f"{each.name}: {len(audio)} samples will not fit in "
                    f"{num_frame} frames")

            dataset_audio[loc_s:loc_s + len(audio)] = audio
            del npz_file # delete npz file to free memory
            loc_s += num_frame * hop + stride_samples

        dataset_audio.flush()
        del dataset_audio # delete to free memory

        # Silence between tracks needs no filling: the gaps are zeros, and
        # log(0 + log_offset) is exactly the log(log_offset) the old feature slab
        # was pre-filled with, so the padding reproduces itself for free.

        ## Process the labels
        print(f"Processing labels for {split}...")
        self._fill_label_slab(
            files_all, num_frame_list, config,
            f"{self.save_name}/label_frames/{split}/dataset_label_frames" + suffix,
            'label_frames', total_num_frame, bool)

        ## process the onsets
        print(f"Processing onsets for {split}...")
        self._fill_label_slab(
            files_all, num_frame_list, config,
            f"{self.save_name}/label_onset/{split}/dataset_label_onset" + suffix,
            'label_onset', total_num_frame, np.float32)

        ## process the offsets
        print(f"Processing offsets for {split}...")
        self._fill_label_slab(
            files_all, num_frame_list, config,
            f"{self.save_name}/label_offset/{split}/dataset_label_offset" + suffix,
            'label_offset', total_num_frame, np.float32)

        ## process the velocities
        print(f"Processing velocities for {split}...")
        self._fill_label_slab(
            files_all, num_frame_list, config,
            f"{self.save_name}/label_velocity/{split}/dataset_label_velocity" + suffix,
            'label_velocity', total_num_frame, np.int8)

        # Record the layout so the dataset can check its frame -> sample
        # arithmetic against what was actually written instead of assuming it.
        # An off-by-one in the * hop conversion is 16 ms of label drift, well
        # inside the 50 ms tolerance, so it would never raise on its own.
        meta = {
            'total_num_frame': int(total_num_frame),
            'num_tracks': len(files_all),
            'sr': config['feature']['sr'],
            'hop_sample': hop,
            'fft_bins': config['feature']['fft_bins'],
            'window_length': config['feature']['window_length'],
            'mel_bins': config['feature']['mel_bins'],
            'log_offset': config['feature']['log_offset'],
            'pad_mode': config['feature']['pad_mode'],
            'audio_pad': audio_pad,
            'margin_b': config['input']['margin_b'],
            'margin_f': config['input']['margin_f'],
            'num_frame': config['input']['num_frame'],
        }
        with open(f"{self.save_name}/meta/{split}/dataset_meta" + str(div).zfill(3) + ".json", "w") as f:
            json.dump(meta, f, indent=2)

        # delete the per-track npz files for train and val
        for i, each in enumerate(files_all):
            if (split == "train" or split == "val") and os.path.exists(each):
                    os.remove(each)
    
    def _extract_hft(self, config: dict):
        """
            Extract the features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.

            Args
            -----
                config (dict): Configuration dictionary containing the parameters   

            Returns
            --------
                None
        """
        split_files = self._get_splits_hft() # train, val, test
        split_str = ["train", "val", "test"]

        print(f"------ Dataset Split Statistics (hFT-Transformer) ------")
        print(f"Number of train files: {len(split_files[0])}")
        print(f"Number of val files: {len(split_files[1])}")
        print(f"Number of test files: {len(split_files[2])}")

        # make an audio directory if it does not exist
        Path(f"{self.save_name}/audio/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/audio/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/audio/test").mkdir(parents=True, exist_ok=True)

        # Also create paths for the labels, we will use them later on
        Path(f"{self.save_name}/label_frames/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_frames/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_frames/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_onset/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_onset/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_onset/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_offset/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_offset/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_offset/test").mkdir(parents=True, exist_ok=True)

        Path(f"{self.save_name}/label_velocity/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_velocity/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/label_velocity/test").mkdir(parents=True, exist_ok=True)

        # make a directory for the idx
        Path(f"{self.save_name}/idx/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/idx/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/idx/test").mkdir(parents=True, exist_ok=True)

        # and one for the per-division layout metadata
        Path(f"{self.save_name}/meta/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/meta/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.save_name}/meta/test").mkdir(parents=True, exist_ok=True)

        print(f"Extracting audio and labels for the hFT-Transformer model...")
        for i, split in tqdm(enumerate(split_files)):
            split_name = split_str[i]
            for (audio_file, midi_file) in tqdm(split, total=len(split), desc="Extracting files"):
                filename = audio_file.stem
                audio = self._get_audio_hft(str(audio_file), config)
                label = self._get_label_hft(str(midi_file), config)

                # num_frames is stored so collation can lay out the division
                # without loading every waveform just to measure it.
                audio_dict = {'audio': audio, 'num_frames': self._num_frames_hft(audio, config)}
                audio_dict.update(label)

                # Save the audio and label to a npz file
                if self.dataset_name != "slakh":
                    store_path = f"{self.save_name}/audio/{split_name}/{filename}.npz"
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"{self.save_name}/audio/{split_name}/{track_name}_{filename}.npz"

                np.savez(store_path, **audio_dict)
    
    def _get_audio_hft(self, audio_file: str, config: dict) -> np.ndarray:
        """
            Get the resampled mono waveform for the audio file.

            hFT used to store a log-mel here and slice it by frame index at
            training time, which left no waveform to augment. We store the audio
            instead and compute the mel per item in the dataset; see
            `scripts/verify_hft_frame_equivalence.py` for the proof that a
            per-item mel reproduces the whole-track one exactly.

            Args
            ------
                audio_file (str): Path to the audio file
                config (dict): Configuration dictionary containing the parameters

            Returns
            --------
                audio (np.ndarray): Mono waveform at config["feature"]["sr"]
        """
        # we use torchaudio to speed this up; librosa is too slow
        audio, sr = torchaudio.load(audio_file)
        audio = torch.mean(audio, dim=0)
        resample = torchaudio.transforms.Resample(sr, config["feature"]["sr"])
        audio = resample(audio)
        return audio.numpy().astype(np.float32)

    @staticmethod
    def _num_frames_hft(audio: np.ndarray, config: dict) -> int:
        """
            Frames a centred STFT produces for this waveform.

            torch.stft with center=True pads by n_fft // 2 on both sides, so the
            count is floor(len / hop) + 1. This has to match what the old stored
            feature had, because dataset_idx and every label array are laid out
            in these units.
        """
        return len(audio) // config["feature"]["hop_sample"] + 1

    def _extend_note_offsets(self, events, config: dict) -> list:
        """ 
            This method sort of mirrors _midi2note from hfTransformer implementation by Sony.
            However, the events represent a list of MIDI messages created using the Message class.
            The main purpose is to make the logic clearer. Extending note offsets can be simple
            yet tricky. Below is a detailed explanation of the logic.

            We have four types of events: 'note_on', 'note_off', 
            'control_change_on', 'control_change_off'.

            If the pedal is pressed (control_change_on), notes that are 
            ACTIVE should be extended.

            If the pedal is released (control_change_off), notes that are
            SUSTAINED and NOT ACTIVE should be ended. Because if they are
            ACTIVE, we should allow them continue...

            If it is a 'note_on' event, we mark the note as ACTIVE if it was
            neither ACTIVE nor SUSTAINED. Otherwise, it is a re-onset.

            If it is a 'note_off' event, we mark the note as not ACTIVE. If
            the pedal is pressed, we leave it as SUSTAINED.

            Args
            -----
                events (list): List of MIDI events sorted by time.
                config (dict): Configuration dictionary containing useful info.

            Returns
            --------
                notes (dict): Dictionary containing note information.
        """
        notes = [] # list containing note events ('onset', 'offset', 'pitch', 'velocity', 'reonset')
        active_notes = [False for _ in range(128)]
        sustained_notes = [False for _ in range(128)]
        reonset_notes = [False for _ in range(128)]
        onset_notes = [-1 for _ in range(128)]
        velocity_notes = [-1 for _ in range(128)]
        min_pitch = config['midi']['note_min']
        max_pitch = config['midi']['note_max']

        for i, event in enumerate(events):
            time = event.time.item()

            if event.type == 'control_change_off':
                # Sustain is off, so end all sustained notes that are not active
                for pitch in range(min_pitch, max_pitch + 1):
                    if sustained_notes[pitch] and not active_notes[pitch]:
                        notes.append({
                            'onset': onset_notes[pitch],
                            'offset': time,
                            'pitch': pitch,
                            'velocity': velocity_notes[pitch],
                            'reonset': reonset_notes[pitch]
                        })
                        onset_notes[pitch] = -1
                        velocity_notes[pitch] = -1
                        reonset_notes[pitch] = False
                sustained_notes = [False for _ in range(128)]
            elif event.type == 'control_change_on':
                # Sustain is on, so mark all active notes as sustained
                for pitch in range(min_pitch, max_pitch + 1):
                    if active_notes[pitch]:
                        sustained_notes[pitch] = True
            elif event.type == 'note_on':
                # Note on event
                # Check if it's a re-onset
                if active_notes[event.note] or sustained_notes[event.note]:
                    # Re-onset
                    notes.append({
                        'onset': onset_notes[event.note],
                        'offset': time,
                        'pitch': event.note,
                        'velocity': velocity_notes[event.note],
                        'reonset': reonset_notes[event.note]
                    })
                    reonset_notes[event.note] = True
                else:
                    reonset_notes[event.note] = False
                onset_notes[event.note] = time
                velocity_notes[event.note] = event.velocity
                active_notes[event.note] = True
                if sustained_notes[event.note]:
                    sustained_notes[event.note] = True
            elif event.type == 'note_off':
                # Note off event
                # End ONLY notes that are active and not sustained
                if active_notes[event.note] and not sustained_notes[event.note]:
                    notes.append({
                        'onset': onset_notes[event.note],
                        'offset': time,
                        'pitch': event.note,
                        'velocity': velocity_notes[event.note],
                        'reonset': reonset_notes[event.note]
                    })
                    onset_notes[event.note] = -1
                    velocity_notes[event.note] = -1
                    reonset_notes[event.note] = False
                active_notes[event.note] = False

        # Handle any remaining active or sustained notes at the end
        final_time = events[-1].time
        for pitch in range(min_pitch, max_pitch + 1):
            if active_notes[pitch] or sustained_notes[pitch]:
                notes.append({
                    'onset': onset_notes[pitch],
                    'offset': final_time,
                    'pitch': pitch,
                    'velocity': velocity_notes[pitch],
                    'reonset': reonset_notes[pitch]
                })

        # Perform sorting operations
        notes.sort(key=lambda x: x['pitch']) # sort by pitch
        notes.sort(key=lambda x: x['onset']) # sort by onset time
        return notes
    
    def _get_notes(self, midi_obj: pretty_midi.PrettyMIDI):
        """
            Retrieve notes from a PrettyMIDI object and
            returns as a list of note dictionaries.

            Args
            ------
                midi_obj (pretty_midi.PrettyMIDI): The PrettyMIDI object to extract notes from.

            Returns
            ---------
                notes (list): List of note dictionaries with keys 'onset', 'offset', 'pitch', 'velocity', 'reonset'.
        """
        notes = []
        for instrument in midi_obj.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append({
                        'onset': note.start,
                        'offset': note.end,
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'reonset': False  # Placeholder, reonset handling can be added if needed
                    })
        
        # Sort notes by pitch and onset time
        notes.sort(key=lambda x: x['pitch']) # sort by pitch
        notes.sort(key=lambda x: x['onset']) # sort by onset time
        return notes

    def _midi2note(self, config, f_midi, verbose_flag = False):
        import mido
        NUM_PITCH = 128
        # (1) read MIDI file
        midi_file = mido.MidiFile(f_midi)
        ticks_per_beat = midi_file.ticks_per_beat
        num_tracks = len(midi_file.tracks)

        # (2) tempo curve
        max_ticks_total = 0
        for it in range(len(midi_file.tracks)):
            ticks_total = 0
            for message in midi_file.tracks[it]:
                ticks_total += int(message.time)
            if max_ticks_total < ticks_total:
                max_ticks_total = ticks_total
        a_time_in_sec = [0.0 for i in range(max_ticks_total+1)]
        ticks_curr = 0
        ticks_prev = 0
        tempo_curr = 0
        tempo_prev = 0
        time_in_sec_prev = 0.0
        for im, message in enumerate(midi_file.tracks[0]):
            ticks_curr += message.time
            if 'set_tempo' in str(message):
                tempo_curr = int(message.tempo)
                for i in range(ticks_prev, ticks_curr):
                    a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
                if ticks_curr > 0:
                    time_in_sec_prev = time_in_sec_prev + ((ticks_curr-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
                tempo_prev = tempo_curr
                ticks_prev = ticks_curr
        for i in range(ticks_prev, max_ticks_total+1):
            a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_curr / 1e06)

        # (3) obtain MIDI message
        a_note = []
        a_onset = []
        a_velocity = []
        a_reonset = []
        a_push = []
        a_sustain = []
        for i in range(NUM_PITCH):
            a_onset.append(-1)
            a_velocity.append(-1)
            a_reonset.append(False)
            a_push.append(False)
            a_sustain.append(False)

        ticks = 0
        sustain_flag = False
        for message in midi_file.tracks[num_tracks-1]:
            ticks += message.time
            time_in_sec = a_time_in_sec[ticks]
            if ('control_change' in str(message)) and ('control=64' in str(message)):
                if message.value < 64:
                    # sustain off
                    for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                        if (a_push[i] is False) and (a_sustain[i] is True):
                            a_note.append({'onset': a_onset[i],
                                        'offset': time_in_sec,
                                        'pitch': i,
                                        'velocity': a_velocity[i],
                                        'reonset': a_reonset[i]})
                            a_onset[i] = -1
                            a_velocity[i] = -1
                            a_reonset[i] = False
                    sustain_flag = False
                    for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                        a_sustain[i] = False
                else:
                    # sustain on
                    sustain_flag = True
                    for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                        if a_push[i] is True:
                            a_sustain[i] = True
            elif ('note_on' in str(message)) and (int(message.velocity) > 0):
                # note on
                note = message.note
                velocity = message.velocity
                if (a_push[note] is True) or (a_sustain[note] is True):
                    # reonset
                    a_note.append({'onset': a_onset[note],
                                'offset': time_in_sec,
                                'pitch': note,
                                'velocity': a_velocity[note],
                                'reonset': a_reonset[note]})
                    a_reonset[note] = True
                else:
                    a_reonset[note] = False
                a_onset[note] = time_in_sec
                a_velocity[note] = velocity
                a_push[note] = True
                if sustain_flag is True:
                    a_sustain[note] = True
            elif (('note_off' in str(message)) or \
                (('note_on' in str(message)) and (int(message.velocity) == 0))):
                # note off
                note = message.note
                velocity = message.velocity
                if (a_push[note] is True) and (a_sustain[note] is False):
                    # offset
                    a_note.append({'onset': a_onset[note],
                                'offset': time_in_sec,
                                'pitch': note,
                                'velocity': a_velocity[note],
                                'reonset': a_reonset[note]})
                    a_onset[note] = -1
                    a_velocity[note] = -1
                    a_reonset[note] = False
                a_push[note] = False

        for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
            if (a_push[i] is True) or (a_sustain[i] is True):
                a_note.append({'onset': a_onset[i],
                            'offset': time_in_sec,
                            'pitch': i,
                            'velocity': a_velocity[i],
                            'reonset': a_reonset[i]})
        a_note_sort = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])

        return a_note_sort

    def _get_label_hft(self, midi_file: str, config: dict) -> dict:
        """ 
            Gets the labels for a given MIDI file.

            Args
            -----
                midi_file (str): midi file path
                config (dict): Configuration dictionary
            
            Returns
            -------
                label (dict): Dictionary containing note-level
                              events.
        """
        notes = []
        max_offset = 0
        pm_notes = self._midi2note(config, midi_file)

        # Extract the notes from the MIDI file
        for note in pm_notes:
            pitch = note["pitch"]
            start = note["onset"]
            end = note["offset"]
            velocity = note["velocity"]
            
            notes.append({
                'pitch': pitch,
                'onset': start,
                'offset': end,
                'velocity': velocity
            })

            if max_offset < end:
                max_offset = end

        # Sort notes by onset and pitch
        notes = sorted(sorted(notes, key=lambda x: x['pitch']), key=lambda x: x['onset'])

        # The hop is the distance between two consecutive frames
        hop_ms = 1000*config["feature"]["hop_sample"] / config["feature"]["sr"]

        # We use a 50ms window as a tolerance as to how many frames to the left or right
        # of the true onset we should consider as being close enough
        # we convert the tolerance to frames below
        onset_tolerance = int(50.0 / hop_ms + 0.5)
        offset_tolerance = int(50.0 / hop_ms + 0.5)

        # number of frames per second
        nframe_in_sec = config['feature']['sr'] / config['feature']['hop_sample']

        nframe = int(max_offset * nframe_in_sec + 0.5) + 1
        label_frames = np.zeros((nframe, config['midi']['num_note']), dtype=bool)
        label_onset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
        label_offset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
        label_velocity = np.zeros((nframe, config['midi']['num_note']), dtype=np.int8)

        for i in range(len(notes)):
            pitch = notes[i]['pitch'] - config['midi']['note_min']

            # Get onset time in frames
            onset_frame = int(notes[i]['onset'] * nframe_in_sec + 0.5)
            onset_ms = notes[i]['onset']*1000.0 # onset time in ms
            onset_sharpness = onset_tolerance

            # offset time in frames
            offset_frame = int(notes[i]['offset'] * nframe_in_sec + 0.5)
            offset_ms = notes[i]['offset']*1000.0 # offset time in ms
            offset_sharpness = offset_tolerance

            # velocity
            velocity = notes[i]['velocity']

            # onset
            for j in range(0, onset_sharpness+1):
                # Create a traingular soft label centred at the note's actual onset time
                # This is the idea for the Kong Model
                onset_ms_q = (onset_frame + j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame+j < nframe:
                    # There may be multiple notes whose tolerance windows overlap.
                    # We always keep the highest label value at each (frame, pitch).
                    label_onset[onset_frame+j][pitch] = max(label_onset[onset_frame+j][pitch], onset_val)
                    if (label_onset[onset_frame+j][pitch] >= 0.5):
                        #  If this frame is “close enough,” record velocity
                        # We only record the velocity if the corresponding onset value is greater than 0.5
                        # By doing this, we ensure the velocity is most-dependent on the onset responsible
                        label_velocity[onset_frame+j][pitch] = velocity
                
            for j in range(1, onset_sharpness+1):
                onset_ms_q = (onset_frame - j) * hop_ms
                onset_ms_diff = onset_ms_q - onset_ms
                onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                if onset_frame-j >= 0:
                    label_onset[onset_frame-j][pitch] = max(label_onset[onset_frame-j][pitch], onset_val)
                    if (label_onset[onset_frame-j][pitch] >= 0.5) and (label_velocity[onset_frame-j][pitch] == 0):
                        # Think about it, if the velocity is already set, we don't need to set it again
                        # This is because we have found a frame above closest to it.
                        # Drawing this out on paper might help
                        label_velocity[onset_frame-j][pitch] = velocity
            
            # mpe or frames label
            for j in range(onset_frame, offset_frame+1):
                label_frames[j][pitch] = 1
            
            # offset
            offset_flag = True
            for j in range(len(notes)):
                if notes[i]['pitch'] != notes[j]['pitch']:
                    continue
                if notes[i]['offset'] == notes[j]['onset']:
                    offset_flag = False
                    break
            
            if offset_flag is True:
                for j in range(0, offset_sharpness+1):
                    offset_ms_q = (offset_frame + j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame+j < nframe:
                        label_offset[offset_frame+j][pitch] = max(label_offset[offset_frame+j][pitch], offset_val)

                for j in range(1, offset_sharpness+1):
                    offset_ms_q = (offset_frame - j) * hop_ms
                    offset_ms_diff = offset_ms_q - offset_ms
                    offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    if offset_frame-j >= 0:
                        label_offset[offset_frame-j][pitch] = max(label_offset[offset_frame-j][pitch],  offset_val)
        
        label = {
            'label_frames': label_frames.tolist(),
            'label_onset': label_onset.tolist(),
            'label_offset': label_offset.tolist(),
            'label_velocity': label_velocity.tolist(),
            'notes': notes
        }
        return label

    def _get_splits_hft(self):
        """
            Extract the audio segments, features and labels from the audio and midi files
            specific to the hFT-Transformer model by Sony.
        """
        train_files = []
        val_files = []
        test_files = []

        if self.dataset_name == "maps":
            train_files, val_files, test_files = self._get_maps_train_val_test()
        elif self.dataset_name == "maestro":
            train_files, val_files, test_files = self._get_maestro_train_val_test()
        elif self.dataset_name == "goat":
            train_files, val_files, test_files = self._get_goat_train_val_test()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported for HFT mode!")
                    
        return train_files, val_files, test_files
