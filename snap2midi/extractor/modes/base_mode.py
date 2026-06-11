import pretty_midi
import numpy as np
from pathlib import Path
import jams
from tqdm import tqdm
import yaml
import csv

SUPPORTED_DATASETS = [
    "oaf_maps",
    "maps",
    "maestro",
    "guitarset",
    "musicnet",
    "slakh",
    "goat"
]

GOAT_DI_AUDIO_COLUMNS = ["di_audio_path"]
GOAT_REAMPED_AUDIO_COLUMNS = [
    "amp_audio_path_1",
    "amp_audio_path_2",
    "amp_audio_path_3",
    "amp_audio_path_4",
    "amp_audio_path_5",
]
GOAT_AUDIO_COLUMNS = GOAT_DI_AUDIO_COLUMNS + GOAT_REAMPED_AUDIO_COLUMNS
GOAT_MIDI_COLUMNS = ["finealigned_midi_path", "unaligned_midi_path"]

class _BaseMode:
    """ 
        Different AMT models might have weird pre-processing 
        pipelines (e.g hFT-Transformer takes context, Onsets and
        Frames has its own methodology for chunking segments etc).
        It is possible two models might have the same pre-processing
        methods; in that case, we can make a wrapper for that. To
        prevent issues with regards to this, we create different
        modes for each model. However, the base mode has functionalities
        that can be used by all the modes.
    """
    def __init__(self, config: dict):
        """ 
            Take the config file and init
            the BaseMode. 

            Args
            ------
                config (dict):
                    Config containing dataset_name,
                    audio extension etc.
        """
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
        elif self.dataset_name == "goat":
            self.data = self._get_files_goat(self.path)
        elif self.dataset_name == "maps" or self.dataset_name == "oaf_maps":
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

            Args
            ----
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns
            -------
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

            Args
            -----
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns
            --------
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

            Args
            -----
                path (str): Path to the MAESTRO dataset

            Returns
            -------
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

            Args
            -----
                path (str): Path to the GuitarSet dataset

            Returns
            --------
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

        Args
        -----
            path (str): Path to the MusicNet dataset

        Returns
        --------
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

        Args
        -----
            path (str): Path to the Slakh dataset

        Returns
        --------
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

        Args
        -----
            path (str): Path to the MAPS dataset

        Returns
        -------
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        # Get the list of audio and midi files
        audio_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*/MUS/*.{self.ext_midi}"))
        self._checker(audio_files, midi_files)
        return audio_files, midi_files
    
    def _get_maps_train_val_test(self):
        """ 
            Get the training, validation and
            test set for MAPS.

            Returns
            -------
            train_files (list): 
                Training files for MAPS
            val_files (list): 
                Validation files for MAPS
            test_files (list): 
                Test files for MAPS
        """
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
        """ 
            Get the training, validation and
            test set for MAESTRO.

            Returns
            -------
            train_files (list): 
                Training files for MAESTRO
            val_files (list): 
                Validation files for MAESTRO
            test_files (list): 
                Test files for MAESTRO
        """
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

    def _get_goat_root(self) -> Path:
        """Return the GOAT folder that contains metadata.csv and data/."""
        base_path = Path(self.path)
        if (base_path / "metadata.csv").exists():
            return base_path

        nested_path = base_path / "GOAT"
        if (nested_path / "metadata.csv").exists():
            return nested_path

        raise FileNotFoundError(
            f"Could not find GOAT metadata.csv in {base_path} or {nested_path}"
        )

    def _resolve_goat_path(self, raw_path: str) -> Path:
        """Resolve GOAT metadata-relative paths to files on disk."""
        if raw_path is None or raw_path == "":
            return Path()

        path = Path(raw_path)
        if path.is_absolute():
            return path

        root = self._get_goat_root()
        parts = path.parts
        if parts and parts[0] == "GOAT":
            parts = parts[1:]

        if parts and parts[0] == "data":
            return root / Path(*parts)

        return root / "data" / Path(*parts)

    def _get_goat_audio_columns(self) -> list[str]:
        """Select GOAT audio columns while intentionally excluding GP audio/files."""
        columns = self.config.get("goat_audio_columns")

        if columns is None or columns == "all" or columns == "di+reamped":
            selected = GOAT_AUDIO_COLUMNS
        elif columns == "reamped":
            selected = GOAT_REAMPED_AUDIO_COLUMNS
        elif columns == "di":
            selected = GOAT_DI_AUDIO_COLUMNS
        elif isinstance(columns, str):
            selected = [columns]
        else:
            selected = list(columns)

        invalid = [column for column in selected if column not in GOAT_AUDIO_COLUMNS]
        if invalid:
            raise ValueError(
                "GOAT only supports DI/reamped audio columns. "
                f"Invalid columns: {invalid}. Allowed columns: {GOAT_AUDIO_COLUMNS}"
            )

        return selected

    def _get_files_goat(self, path: str) -> tuple[list[Path], list[Path]]:
        """Get all GOAT audio/MIDI pairs described by metadata.csv."""
        train_files, val_files, test_files = self._get_goat_train_val_test()
        file_pairs = train_files + val_files + test_files
        audio_files = [audio_file for audio_file, _ in file_pairs]
        midi_files = [midi_file for _, midi_file in file_pairs]
        return audio_files, midi_files

    def _get_goat_rows(self) -> tuple[list[tuple[dict, Path]], int]:
        """Load valid GOAT metadata rows and their best available MIDI path."""
        metadata_csv = self._get_goat_root() / "metadata.csv"
        rows = []
        skipped = 0

        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                midi_path = Path()
                for midi_column in GOAT_MIDI_COLUMNS:
                    midi_value = row.get(midi_column, "")
                    if midi_value:
                        midi_path = self._resolve_goat_path(midi_value)
                        if midi_path.exists():
                            break

                if not midi_path.exists():
                    skipped += 1
                    continue

                rows.append((row, midi_path))

        return rows, skipped

    def _split_goat_rows_from_metadata(self, rows: list[tuple[dict, Path]]) -> dict[str, list[tuple[dict, Path]]]:
        split_rows = {"train": [], "val": [], "test": []}

        for row, midi_path in rows:
            split = row["split"].strip().lower()
            if split in ["valid", "validation"]:
                split = "val"
            if split not in split_rows:
                raise ValueError(f"Unknown GOAT split: {row['split']}")
            split_rows[split].append((row, midi_path))

        val_fraction = float(self.config.get("goat_val_fraction", 0.0) or 0.0)
        if not split_rows["val"] and val_fraction > 0:
            if val_fraction <= 0 or val_fraction >= 1:
                raise ValueError("goat_val_fraction must be greater than 0 and less than 1")

            rng = np.random.default_rng(int(self.config.get("goat_split_seed", 1234)))
            train_rows = split_rows["train"]
            val_count = max(1, int(round(len(train_rows) * val_fraction)))
            val_indices = set(rng.choice(len(train_rows), size=val_count, replace=False).tolist())
            split_rows["train"] = [item for idx, item in enumerate(train_rows) if idx not in val_indices]
            split_rows["val"] = [item for idx, item in enumerate(train_rows) if idx in val_indices]

        return split_rows

    def _split_goat_rows_by_fraction(self, rows: list[tuple[dict, Path]]) -> dict[str, list[tuple[dict, Path]]]:
        test_fraction = self.config.get("goat_test_fraction")
        val_fraction = float(self.config.get("goat_val_fraction", 0.0) or 0.0)

        if test_fraction is None:
            return self._split_goat_rows_from_metadata(rows)

        test_fraction = float(test_fraction)
        if test_fraction <= 0 or test_fraction >= 1:
            raise ValueError("goat_test_fraction must be greater than 0 and less than 1")
        if val_fraction < 0 or val_fraction >= 1:
            raise ValueError("goat_val_fraction must be greater than or equal to 0 and less than 1")
        if test_fraction + val_fraction >= 1:
            raise ValueError("goat_test_fraction + goat_val_fraction must be less than 1")

        rng = np.random.default_rng(int(self.config.get("goat_split_seed", 1234)))
        indices = rng.permutation(len(rows)).tolist()
        test_count = max(1, int(round(len(rows) * test_fraction)))
        val_count = int(round(len(rows) * val_fraction)) if val_fraction > 0 else 0
        if val_fraction > 0:
            val_count = max(1, val_count)

        test_indices = set(indices[:test_count])
        val_indices = set(indices[test_count:test_count + val_count])

        split_rows = {"train": [], "val": [], "test": []}
        for idx, item in enumerate(rows):
            if idx in test_indices:
                split_rows["test"].append(item)
            elif idx in val_indices:
                split_rows["val"].append(item)
            else:
                split_rows["train"].append(item)

        return split_rows

    def _get_goat_train_val_test(self) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        """Build GOAT splits from metadata rows, expanding audio variants inside each split."""
        audio_columns = self._get_goat_audio_columns()
        rows, skipped = self._get_goat_rows()
        split_rows = self._split_goat_rows_by_fraction(rows)

        split_files = {"train": [], "val": [], "test": []}
        for split, row_items in split_rows.items():
            for row, midi_path in row_items:
                for audio_column in audio_columns:
                    audio_value = row.get(audio_column, "")
                    if not audio_value:
                        skipped += 1
                        continue

                    audio_path = self._resolve_goat_path(audio_value)
                    if not audio_path.exists():
                        skipped += 1
                        continue

                    split_files[split].append((audio_path, midi_path))

        split_source = "metadata"
        if self.config.get("goat_test_fraction") is not None:
            split_source = (
                f"custom item-level split "
                f"(test={self.config.get('goat_test_fraction')}, "
                f"val={self.config.get('goat_val_fraction', 0.0)})"
            )

        print(
            "GOAT metadata extraction using audio columns "
            f"{audio_columns} and {split_source}. "
            f"Rows train/val/test: {len(split_rows['train'])}/"
            f"{len(split_rows['val'])}/{len(split_rows['test'])}. "
            f"Pairs train/val/test: {len(split_files['train'])}/"
            f"{len(split_files['val'])}/{len(split_files['test'])}. "
            f"Skipped missing audio/MIDI entries: {skipped}"
        )
        return split_files["train"], split_files["val"], split_files["test"]

    def jams_to_midi(self, jam: jams.JAMS, q: int = 1) -> pretty_midi.PrettyMIDI:
        """
            Convert jams to midi using pretty_midi.
            Gotten from the `marl repo`_.
            .. _marl repo: https://github.com/marl/GuitarSet/blob/master/visualize/interpreter.py

            Args
            ----
                jam (jams.JAMS): Jams object
                q (int): 1: with pitch bend. q = 0: without pitch bend.
            
            Returns
            -------
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
