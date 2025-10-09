import pretty_midi
import numpy as np
from pathlib import Path
import jams
from tqdm import tqdm
import yaml
import csv

class _BaseMode:
    def __init__(self, config: dict):
        # init the main core attributes
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.ext_audio = config["ext_audio"]
        self.ext_midi = config["ext_midi"]
        self.path = config["path"]
        self.save_name = config["save_name"]

        # Just in case extensions are wrong
        self.set_extensions()

        # Extract datasets
        if self.dataset_name == "maestro":
            self.data = self._get_files_maestro(self.path)
        elif self.dataset_name == "guitarset":
            self.data = self._get_files_guitarset(self.path)
        elif self.dataset_name == "musicnet":
            self.data = self._get_files_musicnet(self.path)
        elif self.dataset_name == "slakh":
            self.data = self._get_files_slakh(self.path)
        elif self.dataset_name == "maps":
            self.data = self._get_files_maps(self.path)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        # Create the save directory if it does not exist
        Path(self.save_name).mkdir(parents=True, exist_ok=True)

        if self.dataset_name == "maps":
            # To prevent tick overflow in pretty_midi,
            # for the MAPS dataset
            pretty_midi.pretty_midi.MAX_TICK = 1e10
    
    def set_extensions(self):
        """
            Set the audio and midi file extensions.
        """
        audio_extensions = ["wav", "flac"]
        midi_extensions = ["midi", "mid"]
        
        for ext in audio_extensions:
            if sorted(Path(self.path).rglob(f"*.{ext}")):
                self.ext_audio = ext
                break
        
        for ext in midi_extensions:
            if sorted(Path(self.path).rglob(f"*.{ext}")):
                self.ext_midi = ext
                break
        
        print(f"Audio extension set to: {self.ext_audio}, MIDI extension set to: {self.ext_midi}")

    def _checker(self, audio_files: list[Path], midi_files: list[Path]) -> None:
        """
            Check if the audio and midi files are the same.
            This function should be used only on datasets
            with audio and midi files having the same name.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"
        
        # Generate a random number from 0 to len(audio_files)
        idx = np.random.randint(0, len(audio_files))
        assert audio_files[idx].stem == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"
    
    def _checker_guitarset_slakh(self, audio_files: list[Path], midi_files: list[Path], \
                                dataset: str="guitarset") -> None:
        """
            Check if the audio and midi files are the same
            for the GuitarSet dataset (assuming the audio is
            from the audio_mono-mic folder) or for Slakh.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"
        
        # Generate a random number from 0 to len(audio_files)
        idx = np.random.randint(0, len(audio_files))
        if dataset == "guitarset":
            assert audio_files[idx].stem[:-4] == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"
        else:
            # We assume its Slakh
            audio_track_name = audio_files[idx].parent.parent.stem
            midi_track_name = midi_files[idx].parent.parent.stem

            # Check the track names
            assert audio_track_name == midi_track_name, \
                f"Audio Track name: {audio_track_name} not the same as midi Track name: {midi_track_name}"
            
            # Check the file names
            assert audio_files[idx].stem == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"

    def _get_files_maestro(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the MAESTRO dataset.

            Args:
                path (str): Path to the MAESTRO dataset

            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        audio_files = sorted(Path(path).rglob(f"*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*.{self.ext_midi}"))

        # Since MAESTRO's audio and midi files have the same name,
        # we can use checker
        self._checker(audio_files, midi_files)
        return audio_files, midi_files

    def _get_files_guitarset(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the GuitarSet dataset.

            Args:
                path (str): Path to the GuitarSet dataset

            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        # check if the annotations-midi folder exists
        if not (Path(path)/"annotations-midi").exists():
            # Get all the jams in this path
            path_annot = Path(path)/"annotation"
            all_jams = sorted(path_annot.glob("*.jams")) 

            for _, jamPath in tqdm(enumerate(all_jams), total=len(all_jams)):
                jam_path = str(jamPath)
                jam = jams.load(jam_path)
                midi = self.jams_to_midi(jam, q=1)
                save_path = path_annot.parent / f"annotations-midi/{Path(jam_path).stem}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                midi.write(str(save_path) + f".{self.ext_midi}")

        # Get the list of audio and midi
        audio_files = sorted((Path(path)/"audio_mono-mic").rglob(f"*.{self.ext_audio}"))
        midi_files = sorted((Path(path)/"annotations-midi").rglob(f"*.{self.ext_midi}"))

        # Use the GuitarSet checker to check if the audio and midi files are okay
        self._checker_guitarset_slakh(audio_files, midi_files)

        return audio_files, midi_files

    def _get_files_musicnet(self, path: str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the MusicNet dataset. This function assumes that you are using 
        the MusicNet dataset alongside the musicnet_em labels. Also, note 
        that this function assumes that the musicnet_em labels are in a 
        folder called musicnet_em which should be in the same directory 
        with the test_data/, train_data/, etc. folders.

        Args:
            path (str): Path to the MusicNet dataset

        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        # Get the list of audio and midi files
        audio_files = []
        midi_files = sorted((Path(path)/"musicnet_em/").glob(f"*.{self.ext_midi}"))

        for i in range(len(midi_files)):
            midi_files[i] = Path(midi_files[i])
            temp = sorted(Path(path).rglob(f"{midi_files[i].stem}.{self.ext_audio}"))
            audio_files.append(temp[0])

        # Since MusicNet's audio and midi files have the same name,
        # we can use checker
        self._checker(audio_files, midi_files)

        return audio_files, midi_files

    def _get_files_slakh(self, path: str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the Slakh dataset.

        Args:
            path (str): Path to the Slakh dataset

        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        unwanted = ["Drums", "Percussive", "Sound Effects", "Sound effects", \
                    "Chromatic Percussion"]
        audio_files = []
        midi_files = []

        for each in ['train', 'validation', 'test']:
            base_path = Path(path)/f"{each}/"
            tracks = [folder for folder in base_path.iterdir() if folder.is_dir()]
            for track in tqdm(tracks):
                try:
                    metadata = track / "metadata.yaml"
                    with open(metadata, "r") as f:
                        yaml_data = yaml.safe_load(f)
        
                    for key, value in yaml_data["stems"].items():
                        if value["inst_class"] not in unwanted:
                            audio_file = track / "stems" / f"{key}.{self.ext_audio}"
                            midi_file = track / "MIDI" / f"{key}.{self.ext_midi}"

                            try:
                                assert audio_file.exists(), f"{audio_file} does not exist"
                                assert midi_file.exists(), f"{midi_file} does not exist"
                            except AssertionError as e:
                                continue
                            audio_files.append(audio_file)
                            midi_files.append(midi_file)
                except:
                    print(f"Error in {track}")
                    continue
        
        # Check if the audio and midi files are the same
        # for the Slakh dataset
        self._checker_guitarset_slakh(audio_files, midi_files, dataset="slakh")

        return audio_files, midi_files

    def _get_files_maps(self, path:str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the MAPS dataset. Note that MAPS is organized into
        several categories: ISOL, RAND, UCHO and MUS.

        ISOL: Isolated notes and monophonic sounds
        RAND: Randomly chords
        UCHO: Usual chords
        MUS: Pieces of music

        This function extracts the audio and midi files
        for the MUS category only.

        Args:
            path (str): Path to the MAPS dataset

        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        # Get the list of audio and midi files
        audio_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_midi}"))
        self._checker(audio_files, midi_files)
        return audio_files, midi_files
    
    def _get_maps_train_val_test(self):
        train_files = []
        val_files = []
        test_files = []

        # collect all tunes for test first from the ENSTDkAm and ENSTDkCl subsets
        # After that, collect the rest of the tunes for train and val
        tunes = []
        for i, each in enumerate(self.data[0]):
            tmp = str(each).replace(f"{self.path}", "").replace(\
                f".{self.ext_audio}", "").rstrip('\n').split('/')
            code = tmp[1] # folder name
            content = tmp[2] # category name (MUS, ISOL, etc.)
            tune = tmp[-1].rstrip(code).lstrip('MAPS_'+content+'-') # tune name

            if (code == 'ENSTDkAm' or code == 'ENSTDkCl'):
                # append tune name to the tunes list
                test_files.append((each, self.data[1][i]))
                if tune not in tunes:
                    tunes.append(tune)
        
        for i, each in enumerate(self.data[0]):
            tmp = str(each).replace(f"{self.path}", "").replace(\
                f".{self.ext_audio}", "").rstrip('\n').split('/')
            code = tmp[1] # folder name
            content = tmp[2] # category name (MUS, ISOL, etc.)
            tune = tmp[-1].rstrip(code).lstrip('MAPS_'+content+'-') # tune name

            if (code != 'ENSTDkAm' and code != 'ENSTDkCl'):
                if tune not in tunes:
                    train_files.append((each, self.data[1][i]))
                else:
                    val_files.append((each, self.data[1][i]))
        
        return train_files, val_files, test_files
    
    def _get_maestro_train_val_test(self):
        train_files = []
        val_files = []
        test_files = []

        # metadata_csv is structured as follows:
        # canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration
        # read the csv file
        metadata_csv = Path(self.path) / "maestro-v3.0.0.csv"
        assert metadata_csv.exists(), f"{metadata_csv} does not exist"
        
        with open(metadata_csv, 'r') as f:
            content = csv.reader(f, delimiter=',', quotechar='"')

            base_path = Path(self.path)
            next(content)  # skip the header

            for i, each in enumerate(content):
                if each[2] == 'train':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{self.ext_midi}", f".{self.ext_audio}")
                    audio_path = Path(audio_path)
                    train_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'validation':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{self.ext_midi}", f".{self.ext_audio}")
                    audio_path = Path(audio_path)
                    val_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'test':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{self.ext_midi}", f".{self.ext_audio}")
                    audio_path = Path(audio_path)
                    test_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                else:
                    raise ValueError(f"Split {each[2]} not supported")
        return train_files, val_files, test_files


    def jams_to_midi(self, jam: jams.JAMS, q: int = 1) -> pretty_midi.PrettyMIDI:
        """
            Convert jams to midi using pretty_midi.
            Gotten from the `marl repo`_.
            .. _marl repo: https://github.com/marl/GuitarSet/blob/master/visualize/interpreter.py

            Args:
                jam (jams.JAMS): Jams object
                q (int): 1: with pitch bend. q = 0: without pitch bend.
            
            Returns:
                midi: PrettyMIDI object
        """
        # q = 1: with pitch bend. q = 0: without pitch bend.
        midi = pretty_midi.PrettyMIDI()
        annos = jam.search(namespace='note_midi')
        if len(annos) == 0:
            annos = jam.search(namespace='pitch_midi')
        for anno in annos:
            midi_ch = pretty_midi.Instrument(program=25)
            for note in anno:
                pitch = int(round(note.value))
                bend_amount = int(round((note.value - pitch) * 4096))
                st = note.time
                dur = note.duration
                n = pretty_midi.Note(
                    velocity=100 + np.random.choice(range(-5, 5)),
                    pitch=pitch, start=st,
                    end=st + dur
                )
                pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
                midi_ch.notes.append(n)
                midi_ch.pitch_bends.append(pb)
            if len(midi_ch.notes) != 0:
                midi.instruments.append(midi_ch)
        return midi
