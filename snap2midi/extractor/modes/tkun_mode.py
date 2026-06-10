from .base_mode import _BaseMode
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pretty_midi
from scipy.signal import resample_poly
import shutil
import soundfile as sf
from collections import defaultdict
import pickle

# a local definition of the midi note object
class Note:
    def __init__(self, start, end, pitch, velocity, hasOnset=True, hasOffset=True):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.hasOnset = hasOnset 
        self.hasOffset = hasOffset

    def __repr__(self):
        return str(self.__dict__)

class _TranskunMode(_BaseMode):
    """ 
        _TranskunMode is a class for extracting
        the necessary data for further training.
    """
    def __init__(self, config) -> None:
        """ 
            Instantiate _TranskunMode.

            Args
            ----
                config (dict): Configuration parameters
        """
        super().__init__(config)
        self._transkun(config)
    
    def _transkun(self, config):
        """ 
            Performs extraction based on config.

            Args
            ----
                config (dict): Configuration parameters
        """
        # Check if the path is None
        if self.path is None or not Path(self.path).exists():
            raise ValueError(f"Path {self.path} does not exist!")
        
        if config["dataset_name"] not in ["maps", "maestro"]:
            raise ValueError(f"Dataset {config['dataset_name']} not supported for Transkun mode!")
        
        if config["sample_rate"] is None:
            raise ValueError("sample_rate must be set!")
        
        self.sample_rate = config["sample_rate"]
        self.extend_pedal=self.config["extend_pedal"] # If to extend note offsets
        self.resample = self.config["resample"]

        if self.resample:
            self.resample_dataset()

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

    def resample_dataset(self):
        """ 
            Resamples the entire dataset
            to given sample rate before 
            proceeding with extraction.
        """
        input_dir = Path(self.path)
        output_dir = Path(self.save_name) / "resampled"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_files = list(input_dir.rglob("*"))

        for in_path in tqdm(all_files, desc="Processing files"):
            # Skip directories
            if not in_path.is_file():
                continue

            # Mirror structure in output directory
            rel_path = in_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Process WAV files
            if in_path.suffix.lower() in [".wav", ".mp3"]:

                audio, fs = sf.read(in_path, always_2d=False)
                info = sf.info(in_path)
                subtype = info.subtype

                if fs != self.sample_rate:
                    # Resample along time axis
                    audio = resample_poly(
                        audio,
                        up=self.sample_rate,
                        down=fs,
                        axis=0,
                    )

                    # Restore original PCM dtype
                    if subtype == "PCM_16":
                        audio = np.clip(
                            audio * 32767,
                            -32768,
                            32767
                        ).astype(np.int16)

                    elif subtype == "PCM_24":
                        # Stored in int32 container
                        audio = np.clip(
                            audio * (2**23 - 1),
                            -(2**23),
                            (2**23 - 1)
                        ).astype(np.int32)

                    elif subtype == "PCM_32":
                        audio = np.clip(
                            audio * (2**31 - 1),
                            -(2**31),
                            (2**31 - 1)
                        ).astype(np.int32)

                    print(
                        f"Resampled: {rel_path} "
                        f"({fs} → {self.sample_rate})"
                    )

                else:
                    print(f"Copied audio unchanged: {rel_path}")

                # Save using original subtype
                sf.write(
                    out_path,
                    audio,
                    self.sample_rate,
                    subtype=subtype,
                )

            # Copy non-WAV files
            else:
                shutil.copy2(in_path, out_path)
                print(f"Copied file: {rel_path}")
        
        # Update the dataset path to the resampled folder
        self.path = output_dir
    
    def _extract(self, split_files: list, split:str):
        """ 
            Extract audio paths and the MIDI notes
            for each file. The audio will be loaded
            at the target sample rate.

            Args
            ----
                split_files (list): List of audio/MIDI pairs
                split (str): train, val or test split
        """
        # parse the (audio_file, midi_file) in split_files
        for audio_file, midi_file in tqdm(split_files, \
                        desc=f"Processing audio and MIDI files for {split} split..."):
                assert audio_file.exists(), f"{audio_file} does not exist"
                assert midi_file.exists(), f"{midi_file} does not exist"

                if self.dataset_name != "slakh":
                    store_path = f"./{self.save_name}/{split}/{str(audio_file.stem)}.pt" 
                else:
                    # For slakh, we need to get the track name
                    # from the audio file path
                    track_name = audio_file.parent.parent.stem
                    store_path = f"./{self.save_name}/{split}/{track_name}_{str(audio_file.stem)}.pt"

                # audio, fs = librosa.load(audio_file, sr=self.sample_rate, mono=False)
                midiObj: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI(midi_file)
                assert(len(midiObj.instruments) == 1)

                inst: pretty_midi.Instrument = midiObj.instruments[0]
                if len(midiObj.instruments)>1:
                    raise Exception("contains more than one track")
                events = self.parseEventAll(inst.notes, inst.control_changes, \
                                extendSustainPedal=self.extend_pedal)


                dump_data = {
                    "notes": events,
                    "audio_filename": str(audio_file),
                    "fs": self.sample_rate,
                    "duration": midiObj.get_end_time() 
                }

                with open(store_path, "wb") as f:
                    pickle.dump(dump_data, f)
                # np.savez(store_path, notes=events, audio=audio, \
                #     fs=fs, duration=midiObj.get_end_time())

    def parseControlChangeSwitch(self, ccSeq: list, controlNumber: int, \
            onThreshold: int = 64, endT: float|None = None):
        """ 
            Parse control changes.
            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            ----
                ccSeq (list): List of control changes
                controlNumber (int): The control change id
                onThreshold (int): Threshold for control change
                endT (float | None): Max time to consider
        """
        runningStatus = False
        seqEvent = []
        currentEvent = None
        currentStatus = False

        time = 0

        for c in ccSeq:
            if c.number == controlNumber:
                time = c.time
                if c.value>=onThreshold:
                    currentStatus = True
                else:
                    currentStatus = False
            
            if runningStatus != currentStatus:
                if currentStatus == True:
                    #use negative number as pitch for the control change event
                    # the velocity of a pedal is normalized to 0-1, where values smaller than off is cut off
                    currentEvent = Note(time, None, -controlNumber, 127)
                else:
                    currentEvent.end = time
                    seqEvent.append(currentEvent)
            runningStatus = currentStatus
        if runningStatus and endT is not None:
            # process the case where the state is not closed off at the end
            # print("Warning: running status {} not closed at the end".format(controlNumber));
            currentEvent.end = max(endT, time)
            if currentEvent.end > currentEvent.start:
                seqEvent.append(currentEvent)
        return seqEvent
    
    def extendPedal(self, note_events, pedal_events):
        """ 
            Extend note offsets based on pedal events.
            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            ----
                note_events (list): List of note events
                pedal_events (list): List of pedal events
            
            Returns
            -------
                ex_note_events (list): List of extended note events.
        """
        note_events.sort(key = lambda x: (x.start, x.end,x.pitch))
        pedal_events.sort(key = lambda x: (x.start, x.end,x.pitch))
        ex_note_events = []

        idx = 0     

        buffer_dict = {}
        nIn = len(note_events)        
        for note_event  in note_events:

            midi_note = note_event.pitch
            if midi_note in buffer_dict.keys():
                _idx = buffer_dict[midi_note]
                if ex_note_events[_idx].end > note_event.start:
                    ex_note_events[_idx].end = note_event.start


            for curPedal in pedal_events:
                if note_event.end< curPedal.end and note_event.end>curPedal.start:
                    note_event.end = curPedal.end

            
            buffer_dict[midi_note] = idx
            idx += 1
            ex_note_events.append(note_event)

        # print("haha")
        ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

        nOut = len(ex_note_events)
        assert(nOut == nIn)

        ex_note_events = self.resolveOverlapping(ex_note_events)
        self.validateNotes(ex_note_events)
        return ex_note_events

    def resolveOverlapping(self, note_events):
        """ 
            Deal with overlapping notes
            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            -----
                note_events (list): List of note events
            
            Returns
            -------
                ex_note_events (list): resolved list of note events
        """
        note_events.sort(key = lambda x: (x.start, x.end,x.pitch))
        ex_note_events = []
        idx = 0     
        buffer_dict = {}
        
        for note_event  in note_events:
            midi_note = note_event.pitch
            # note_event.end = max(note_event.start+1e-5, note_event.end)
            # note_event.end = max(note_event.start+1e-5, note_event.end)
            if midi_note in buffer_dict.keys():
                _idx = buffer_dict[midi_note]
                if ex_note_events[_idx].end > note_event.start:
                    ex_note_events[_idx].end = note_event.start

            buffer_dict[midi_note] = idx
            idx += 1

            ex_note_events.append(note_event)

        ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

        # else:
            # print("overlappingOnsetOffset", note_event)

        # remove all notes that has start == end
        n1 = len(ex_note_events)
        error_notes = [n for n in ex_note_events if not n.start<n.end]
        ex_note_events = [n for n in ex_note_events if n.start<n.end]
        n2 = len(ex_note_events)
        if n1!=n2:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(error_notes)

        self.validateNotes(ex_note_events)
        return ex_note_events

    def validateNotes(self, notes: list) -> None:
        """ 
            Validates notes

            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            ----
                notes (list): List of notes
            
            Returns
            -------
                None
        """
        pitches = defaultdict(list)
        for n in notes:
            if len(pitches[n.pitch])>0:
                nPrev = pitches[n.pitch][-1]
                assert n.start >= nPrev.end, str(n)+ str(nPrev)
            
            assert n.start < n.end, n

            pitches[n.pitch].append(n)
    
    def parseEventAll(self, notesList: list, ccList: list, supportedCC: list = [64, 66, 67], \
            extendSustainPedal : bool = True, pedal_ext_offset: float = 0.0):
        """ 
            Extract note events.

            `Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Data.py`

            Args
            ----
                notesList (list): list of note events
                ccList (list): list of control changes
                supportedCC (list): Supported control changes
                extendSustainPedal (bool): Extend note offsets
                pedal_ext_offset (float): Offset to pedal timing
            
            Returns
            -------
                events (list): List of note events.
        """
        # CC 64: sustain
        # CC 66: sostenuto
        # CC 67: una conda
        # normalize all velocity of notes
        notesList = [ Note(**n.__dict__) for n in notesList]
        notesList.sort(key = lambda x: (x.start, x.end,x.pitch))

        for n in notesList:
            assert n.start < n.end

        # get the ending time of the last note event for the missing off event at the boundary 
        lastT = max([n.end for n in notesList])        
        if extendSustainPedal:
            # currently ignore cc 66
            sustainEvents = self.parseControlChangeSwitch(ccList, controlNumber = 64, endT = lastT)
            sustainEvents.sort(key = lambda x: (x.start, x.end,x.pitch))

            if pedal_ext_offset != 0.0:
                for n in sustainEvents:
                    n.start += pedal_ext_offset
                    n.end += pedal_ext_offset

            notesList = self.extendPedal(notesList, sustainEvents)
        else:
        # remove overlappings, als remove n.start>=n.end
            notesList = self.resolveOverlapping(notesList)
        self.validateNotes(notesList)

        eventSeqs = [notesList]
        for ccNum in supportedCC:
            ccSeq = self.parseControlChangeSwitch(ccList, controlNumber = ccNum, endT=lastT)
            eventSeqs.append(ccSeq)
        
        events = sum(eventSeqs, [])

        # sort all events by the beginning
        events.sort(key = lambda x: (x.start, x.end,x.pitch))
        return events
