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
    "goat",
    # Held-out evaluation sets. None of them is ever trained on here: they exist
    # so a MAESTRO-trained model can be scored out of domain, so their split
    # functions put every item in test.
    "smd",
    "mds",
    "gaps",
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


def drop_out_of_range_notes(midi_obj, min_pitch, max_pitch):
    """Remove notes outside [min_pitch, max_pitch], returning how many went.

    The label roll has exactly `max_pitch - min_pitch + 1` columns, so a note
    outside that range indexes past the end and raises. MDS has exactly one such
    note in 343,562 (pitch 112 against an 88-key roll) -- negligible for any
    score, fatal for the extraction of that file. A note *below* the range is
    worse than fatal: `note.pitch - min_pitch` is negative there, and numpy
    reads that as an index from the far end, so the note would be written
    silently onto the wrong pitch instead of raising.

    A module-level function rather than only a method, because the OAF extractor
    frames its events in a class that is not a `_BaseMode`.
    """
    dropped = 0
    for instrument in midi_obj.instruments:
        kept = [n for n in instrument.notes if min_pitch <= n.pitch <= max_pitch]
        dropped += len(instrument.notes) - len(kept)
        instrument.notes = kept
    return dropped


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
        elif self.dataset_name == "smd":
            self.data = self._get_files_smd(self.path)
        elif self.dataset_name == "mds":
            self.data = self._get_files_mds(self.path)
        elif self.dataset_name == "gaps":
            self.data = self._get_files_gaps(self.path)
        elif self.dataset_name == "gaps_ft":
            self.data = self._get_files_gaps_ft(self.path)
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
        # mp3 is here for GAPS, whose audio is YouTube-sourced. Order matters:
        # the first extension that matches anything under the root wins.
        audio_extensions = ["wav", "flac", "mp3"]
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

    def _get_files_smd(self, path: str) -> tuple[list[Path], list[Path]]:
        """Saarland Music Data, 50 real Disklavier piano recordings.

        Pairs `midi/<stem>.mid` with `wav_44100_stereo/<stem>.wav`. The 44.1 kHz
        stereo files are the actual recordings. Do NOT use
        `midi_wav_22050_mono/`: those are *synthesised* renderings of the same
        MIDI, so scoring against them would be circular. `wav_22050_mono/` is a
        plain downsample and would only lose bandwidth we resample away anyway.
        """
        audio_files, midi_files = [], []
        for midi in sorted((Path(path) / "midi").glob(f"*.{self.ext_midi}")):
            audio = Path(path) / "wav_44100_stereo" / f"{midi.stem}.{self.ext_audio}"
            if audio.exists():
                audio_files.append(audio)
                midi_files.append(midi)
        self._checker(audio_files, midi_files)
        return audio_files, midi_files

    def _get_files_mds(self, path: str) -> tuple[list[Path], list[Path]]:
        """MDS piano set: `Audio/<part>_<render>/` against `MIDI/<part>/`.

        Part `3_maetest` ships two renderings of the same 20 MIDI files -- `_dk`
        (Disklavier) and `_mae` -- so those MIDI files legitimately appear twice,
        paired with different audio. That is a matched A/B on rendering and is
        kept rather than de-duplicated.
        """
        audio_files, midi_files = [], []
        for adir in sorted((Path(path) / "Audio").iterdir()):
            if not adir.is_dir():
                continue
            part, _, _render = adir.name.rpartition("_")
            mdir = Path(path) / "MIDI" / part
            if not mdir.is_dir():
                continue
            for audio in sorted(adir.glob(f"*.{self.ext_audio}")):
                midi = mdir / f"{audio.stem}.{self.ext_midi}"
                if midi.exists():
                    audio_files.append(audio)
                    midi_files.append(midi)
        return audio_files, midi_files

    def _get_files_gaps(self, path: str) -> tuple[list[Path], list[Path]]:
        """GAPS guitar set: one directory per track under `audio_tracks/`.

        Audio is mp3 and the fine-aligned MIDI sits beside it. Track ids are
        CASE-SENSITIVE -- `Bc1wc` and `bc1wc` are different tracks -- so never
        fold case when matching them against the split file.
        """
        audio_files, midi_files = [], []
        for item in sorted((Path(path) / "audio_tracks").iterdir()):
            if not item.is_dir():
                continue
            audio = next(iter(sorted(item.glob(f"*.{self.ext_audio}"))), None)
            midi = next(iter(sorted(item.glob("*-fine-aligned.mid"))), None)
            if audio and midi:
                audio_files.append(audio)
                midi_files.append(midi)
        return audio_files, midi_files

    def _get_files_gaps_ft(self, path: str) -> tuple[list[Path], list[Path]]:
        """GAPS as prepared for fine-tuning: `<root>/{train,val,test}/<id>.wav`
        with the MIDI beside it.

        This is the tree experiments/data/prepare.py writes (44.1 kHz mono
        WAV, the fine-aligned MIDI renamed to `.midi`, val carved out of the
        official train split by a fixed seed). Reading the split from the
        directory layout, rather than from gaps_v1_splits.csv as `gaps` does,
        is what makes every model fine-tune on exactly the same 240/27/30
        tracks -- the study compares architectures, so the data must not vary.

        MIDI is matched by stem and accepts `.midi` or `.mid`, whichever sits
        next to the audio; set_extensions() picks one extension for the whole
        root and would otherwise miss a mixed tree silently.
        """
        audio_files, midi_files = [], []
        root = Path(path)
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(
                    f"{split_dir} missing -- gaps_ft expects the prepared "
                    "<root>/{train,val,test}/ layout from experiments/data/prepare.py")
            for audio in sorted(split_dir.glob(f"*.{self.ext_audio}")):
                midi = next((audio.with_suffix(f".{ext}")
                             for ext in ("midi", "mid")
                             if audio.with_suffix(f".{ext}").exists()), None)
                if midi is None:
                    raise FileNotFoundError(f"no MIDI beside {audio}")
                audio_files.append(audio)
                midi_files.append(midi)
        return audio_files, midi_files

    def _get_gaps_ft_train_val_test(self):
        """Split by containing directory: the layout IS the split."""
        splits = {"train": [], "val": [], "test": []}
        for audio, midi in zip(self.data[0], self.data[1]):
            splits[audio.parent.name].append((audio, midi))
        for name, pairs in splits.items():
            print(f"gaps_ft: {len(pairs)} items in {name}")
        assert splits["train"] and splits["val"], \
            "gaps_ft: empty train or val split -- wrong root?"
        return splits["train"], splits["val"], splits["test"]

    def _get_heldout_train_val_test(self):
        """Everything into test, nothing into train or val.

        For SMD, MDS and GuitarSet, which are used here only to score a model
        trained elsewhere. Returning empty train/val is deliberate: it makes an
        accidental training run on one of these fail loudly instead of silently
        fitting the set it is supposed to be held out from.
        """
        pairs = list(zip(self.data[0], self.data[1]))
        print(f"{self.dataset_name}: {len(pairs)} items, all assigned to test")
        return [], [], pairs

    def _get_gaps_train_val_test(self):
        """GAPS ships an official split; use it rather than inventing one."""
        splits_file = Path(self.path) / "gaps_v1_splits.csv"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"{splits_file} not found -- GAPS needs its official split file. "
                "Do not fall back to an invented split; the published numbers "
                "for this dataset are on the official test set.")
        wanted = set()
        with open(splits_file, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"].strip().lower() == "test":
                    wanted.add(Path(row["filename"].strip()).stem)

        pairs, found = [], set()
        for audio, midi in zip(self.data[0], self.data[1]):
            # The track id is the containing directory, not the file stem.
            if audio.parent.name in wanted:
                pairs.append((audio, midi))
                found.add(audio.parent.name)
        missing = sorted(wanted - found)
        if missing:
            print(f"gaps: {len(missing)} of {len(wanted)} official test ids are "
                  f"absent from this copy: {', '.join(missing)}")
        print(f"gaps: {len(pairs)} items assigned to test")
        return [], [], pairs

    def _to_int16(self, audio):
        """Scale float audio to int16, clipping the overshoot resampling leaves.

        Requiring |x| <= 1 rejects ordinary audio: librosa's resampler rings
        slightly on near-full-scale material and lossy formats overshoot more.
        Measured across these evaluation sets -- GAPS mp3 up to 1.071, GOAT
        1.010, SMD 1.003 -- so the bound is wrong, not the audio.

        The overshoot still cannot be ignored, because
        `(audio * 32767).astype(np.int16)` WRAPS on overflow: a peak of 1.07
        becomes a large negative sample and the corruption is silent. So clip
        to [-1, 1] first, and keep a loose guard for input that was never
        normalised at all (int16-valued floats would land in the thousands).
        """
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        assert peak <= 8.0, (
            f"audio peak {peak:.3f} does not look like normalised float audio; "
            "refusing to scale it to int16")
        return (np.clip(audio, -1.0, 1.0) * 32767.).astype(np.int16)

    def _store_stem(self, audio_file) -> str:
        """Filename for this item in the store. Must be unique across the dataset.

        MDS renders the same MIDI twice -- `Audio/3_maetest_dk/X.wav` and
        `Audio/3_maetest_mae/X.wav` have identical stems -- so keying the store
        on the stem alone silently drops 20 of 212 items and takes the _dk/_mae
        A/B with them. Qualify by the containing directory there. Every other
        dataset here has unique stems and is left alone.
        """
        if self.dataset_name == "mds":
            return f"{audio_file.parent.name}_{audio_file.stem}"
        return audio_file.stem

    def _resolve_split(self):
        """Pick the train/val/test split, honouring `eval_only`.

        `eval_only` forces every item into test whatever the dataset's own split
        logic would do. The cross-dataset benchmark needs that: GOAT ships a
        *fractional* splitter meant for training on GOAT, which quietly holds
        back three quarters of the data from an evaluation that wants all of it.
        """
        # eval_only puts EVERY item into test, including for datasets that ship
        # their own split. That is deliberate for this benchmark: none of these
        # models has seen any of these datasets (they trained on MAESTRO), so
        # there is no leakage, and the shipped splits exist to support *training
        # on that dataset* rather than to hold anything back from us. They are
        # also small -- GOAT designates 7 of 172 items as test, GAPS 31 of ~398 --
        # so honouring them would throw away most of the data and leave intervals
        # too wide to read.
        #
        # The per-piece JSONs keep every item id, so an official-split subset can
        # still be sliced out afterwards for comparability without re-running
        # anything. Do that at report time, not here.
        if self.config.get("eval_only"):
            return self._get_heldout_train_val_test()
        if self.dataset_name in ("maps", "oaf_maps"):
            return self._get_maps_train_val_test()
        if self.dataset_name == "maestro":
            return self._get_maestro_train_val_test()
        if self.dataset_name == "goat":
            return self._get_goat_train_val_test()
        if self.dataset_name == "gaps":
            return self._get_gaps_train_val_test()
        if self.dataset_name == "gaps_ft":
            return self._get_gaps_ft_train_val_test()
        if self.dataset_name in ("smd", "mds", "guitarset"):
            return self._get_heldout_train_val_test()
        raise ValueError(
            f"No split rule for dataset {self.dataset_name}")

    def _drop_out_of_range_notes(self, midi_obj):
        """`drop_out_of_range_notes` against this mode's own pitch range."""
        return drop_out_of_range_notes(midi_obj, self.min_pitch, self.max_pitch)

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
