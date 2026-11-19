"""Helpers shared by the model functions of models.py, and the test sets of
the two models evaluated from audio and MIDI files (test_items)."""

import csv
import os
from pathlib import Path

import numpy as np
import pretty_midi
import torch

os.environ.setdefault("NNAUDIO_DISABLE_CITATION_REMINDER", "1")

# The library's loaders (load_oafv2, load_hpp, load_kong) call torch.load with
# no map_location, so checkpoints saved on a GPU cannot be read on a machine
# without one. Default it to the CPU there; with a GPU nothing changes.
if not torch.cuda.is_available():
    _torch_load = torch.load
    torch.load = lambda *a, **k: _torch_load(*a, **{"map_location": "cpu", **k})

PITCH_OFFSET = 21   # MIDI pitch of the first of the 88 piano keys
EXPERIMENTS = Path(__file__).resolve().parents[1]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hz(midi_pitches):
    return 440 * (2 ** ((midi_pitches - 69) / 12))


def notes_from_rolls(note_extract, onset, frame, velocity, frame_rate, threshold=None,
                     vel_scale=None):
    """Notes decoded from onset/frame/velocity rolls by the library's
    note_extract: the predictions at the model's threshold, the labels at
    note_extract's default thresholds. The re-onset rule (a note ends when
    its pitch is struck again, Magenta's decoder) is passed explicitly rather
    than left to a library default. Returns (intervals, pitches in Hz,
    velocities)."""
    kwargs = dict(restart_on_reonset=True)
    if threshold is not None:
        kwargs.update(onset_thresh=threshold, frame_thresh=threshold)
    pitches, intervals, velocities = note_extract(onset, frame, velocity, **kwargs)
    if vel_scale:
        velocities = vel_scale(velocities)
    # No notes comes back as shape (0,); mir_eval wants n x 2 even when empty.
    intervals = (intervals / frame_rate).reshape(-1, 2)
    return intervals, hz(pitches + PITCH_OFFSET), velocities


# ---------------------------------------------------------------------------
# Reading a MIDI file into notes, as TransKun's evaluation reads it
# (transkun.computeMetrics): same-pitch notes that overlap are cut where the next one starts,
# notes left with no duration are dropped, and on the piano sets the sustain
# pedal (CC64) first extends the notes it holds.
#
# Ported from TransKun (https://github.com/Yujia-Yan/Transkun,
# transkun/Data.py at commit c5cb837), MIT License,
# Copyright (c) 2021 Yujia Yan.

class _Note:
    def __init__(self, start, end, pitch, velocity):
        self.start, self.end, self.pitch, self.velocity = start, end, pitch, velocity


def _order(notes):
    notes.sort(key=lambda n: (n.start, n.end, n.pitch))
    return notes


def _sustain_spans(control_changes, end_time, number=64, on_threshold=64):
    """Pedal-down spans of one controller (TransKun's parseControlChangeSwitch)."""
    spans, down, current, time = [], False, None, 0
    for c in control_changes:
        if c.number == number:
            time = c.time
            state = c.value >= on_threshold
        else:
            state = down
        if state != down:
            if state:
                current = _Note(time, None, -number, 127)
            else:
                current.end = time
                spans.append(current)
        down = state
    if down:
        current.end = max(end_time, time)
        if current.end > current.start:
            spans.append(current)
    return spans


def _resolve_overlapping(notes):
    """Cut a note where the next note of the same pitch starts, then drop the
    notes left with no duration (a duplicate onset becomes one note)."""
    _order(notes)
    last = {}
    for i, n in enumerate(notes):
        if n.pitch in last and notes[last[n.pitch]].end > n.start:
            notes[last[n.pitch]].end = n.start
        last[n.pitch] = i
    return [n for n in _order(notes) if n.start < n.end]


def _extend_pedal(notes, spans):
    """A note released while the pedal is down lasts until the pedal is lifted;
    same-pitch overlaps are then resolved as above."""
    _order(notes)
    spans = _order(spans)
    last = {}
    for i, n in enumerate(notes):
        if n.pitch in last and notes[last[n.pitch]].end > n.start:
            notes[last[n.pitch]].end = n.start
        for s in spans:
            if s.start < n.end < s.end:
                n.end = s.end
        last[n.pitch] = i
    return _resolve_overlapping(_order(notes))


def merge_overlaps(notes):
    """The same rule applied to decoded notes (intervals, pitches in Hz,
    velocities): same-pitch overlaps cut, notes left with no duration dropped,
    so estimate and reference are read alike."""
    intervals, pitches, velocities = notes
    if len(pitches) == 0:
        return notes
    midi = np.round(69 + 12 * np.log2(np.asarray(pitches) / 440)).astype(int)
    kept = _resolve_overlapping([_Note(float(s), float(e), int(p), v)
                                 for (s, e), p, v in zip(intervals, midi, velocities)])
    if not kept:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (np.array([[n.start, n.end] for n in kept]),
            np.array([2 ** ((n.pitch - 69) / 12) * 440 for n in kept]),
            np.array([n.velocity for n in kept]))


def read_midi_notes(path, extend_pedal: bool):
    """(intervals, pitches in Hz, velocities) of a single-track MIDI file. A
    file with no notes (an empty estimate) gives empty arrays."""
    pretty_midi.pretty_midi.MAX_TICK = 1e10   # some MAPS files exceed the default
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    if midi.instruments:
        if len(midi.instruments) != 1:
            raise ValueError(f"{path}: {len(midi.instruments)} tracks, expected one")
        inst = midi.instruments[0]
        notes = _order([_Note(n.start, n.end, n.pitch, n.velocity) for n in inst.notes])
    if notes:
        if any(not n.start < n.end for n in notes):
            raise ValueError(f"{path}: a note with no duration")
        if extend_pedal:
            end_time = max(n.end for n in notes)
            notes = _extend_pedal(notes, _sustain_spans(inst.control_changes, end_time))
        else:
            notes = _resolve_overlapping(notes)
    if not notes:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (np.array([[n.start, n.end] for n in notes]),
            np.array([2 ** ((n.pitch - 69) / 12) * 440 for n in notes]),
            np.array([n.velocity for n in notes]))


# ---------------------------------------------------------------------------
# Test sets of hFT and TransKun, which are evaluated from the audio and the
# reference MIDI directly: each set is the list of (item_id, audio_path,
# midi_path) its builder below finds in the datasets of config.yaml. The
# other four models read their feature stores instead.
#
# Every list is checked against the study's list of the set in
# experiments/data/splits/ (GAPS: gaps_split.csv; GOAT DI: goat_di_items.txt;
# the others: <set>_items.txt), so a copy of a dataset that would score a
# different set of pieces stops here instead.

_SPLITS = EXPERIMENTS / "data" / "splits"


def study_pieces(name):
    """The ids of the pieces the study scores on one test set."""
    if name.startswith("gaps"):
        split = {"gaps": None, "gaps_test": "test", "gaps_val": "val"}[name]
        with open(_SPLITS / "gaps_split.csv") as fh:
            return {r["stem"] for r in csv.DictReader(fh) if split in (None, r["split"])}
    if name == "goat_di":
        return set((_SPLITS / "goat_di_items.txt").read_text().split())
    return set((_SPLITS / f"{name}_items.txt").read_text().split())


def _builders(cfg):
    D, P = cfg["datasets"], cfg["prepared"]
    splits = EXPERIMENTS / "data" / "splits"

    def rows_smd():
        smd = D["smd"]
        return [(m.stem, smd / "wav_44100_stereo" / f"{m.stem}.wav", m)
                for m in sorted((smd / "midi").glob("*.mid"))
                if (smd / "wav_44100_stereo" / f"{m.stem}.wav").exists()]

    def rows_mds():
        mds, out = D["mds"], []
        for adir in sorted((mds / "Audio").iterdir()):
            part, _, render = adir.name.rpartition("_")
            for audio in sorted(adir.glob("*.wav")):
                midi = mds / "MIDI" / part / f"{audio.stem}.mid"
                if midi.exists():
                    out.append((f"{part}_{render}_{audio.stem}", audio, midi))
        return out

    def rows_guitarset():
        """Reference MIDI is the single-track copy prepare.py wrote,
        since TransKun's parser asserts one instrument."""
        merged, out = P["guitarset"] / "test", []
        for audio in sorted((D["guitarset"] / "audio_mono-mic").glob("*_mic.wav")):
            midi = merged / f"{audio.stem[:-4]}.midi"
            if midi.exists():
                out.append((audio.stem, audio, midi))     # the id keeps _mic, as Snap2MIDI's does
        return out

    def rows_gaps():
        """Audio decoded by prepare.py; reference is the release's fine-aligned MIDI."""
        out = []
        for wav in sorted(P["gaps"].glob("*/*.wav")):
            midi = D["gaps"] / "audio_tracks" / wav.stem / f"{wav.stem}-fine-aligned.mid"
            if midi.exists():
                out.append((wav.stem, wav, midi))
        return out

    def rows_gaps_test():
        """The 30 test tracks of rows_gaps()."""
        return [r for r in rows_gaps() if r[1].parent.name == "test"]

    def rows_gaps_val():
        """The 27 validation tracks used for checkpoint selection and calibration."""
        out = []
        for a in sorted((P["gaps"] / "val").glob("*.wav")):
            midi = next((a.with_suffix(f".{e}") for e in ("midi", "mid")
                         if a.with_suffix(f".{e}").exists()), None)
            if midi is None:
                raise SystemExit(f"no MIDI beside {a}")
            out.append((a.stem, a, midi))
        return out

    def rows_goat():
        root = D["goat"]

        def resolve(rel):
            return root / "data" / Path(*Path(rel.strip()).parts[1:])
        out = []
        with open(root / "metadata.csv") as fh:
            for r in csv.DictReader(fh):
                # 20 of 172 items have no fine-aligned MIDI; Snap2MIDI's GOAT
                # reader falls back to the unaligned MIDI for those, and so does this.
                col = next(c for c in ("finealigned_midi_path", "unaligned_midi_path") if r[c].strip())
                midi = resolve(r[col])
                cols = [("di_audio_path", r["item"])] + \
                       [(f"amp_audio_path_{k}", f"{r['item']}_amp_{k}") for k in range(1, 6)]
                for c, item_id in cols:
                    audio = resolve(r[c])
                    if audio.exists() and midi.exists():
                        out.append((item_id, audio, midi))
        return out

    def rows_goat_di():
        """The 170 direct-input performances of splits/goat_di_items.txt: the DI
        recording of each take, minus item_96 and item_110, whose fine-aligned
        MIDI runs about twice the length of the audio."""
        keep = set((splits / "goat_di_items.txt").read_text().split())
        return [r for r in rows_goat() if r[0] in keep]

    def rows_studio():
        out = []
        for wav in sorted(P["studio_maestro"].glob("*/*.wav")):
            midi = wav.with_suffix(".midi")
            if midi.exists():
                out.append((wav.stem, wav.resolve(), midi.resolve()))
        return out

    def rows_maestro():
        m, out = D["maestro"], []
        with open(m / "maestro-v3.0.0.csv") as fh:
            for r in csv.DictReader(fh):
                if r["split"] == "test":
                    audio, midi = m / r["audio_filename"], m / r["midi_filename"]
                    out.append((audio.stem, audio, midi))
        return sorted(out)

    def rows_maps_full():
        """All 270 MUS pieces of the nine MAPS pianos; Table 1 keeps the 60 of
        the two real Disklaviers (ENSTDkCl, ENSTDkAm)."""
        return [(w.stem, w, w.with_suffix(".mid")) for w in sorted(D["maps"].glob("*/MUS/*.wav"))]

    return {
        "smd": rows_smd, "mds": rows_mds, "guitarset": rows_guitarset,
        "gaps": rows_gaps, "gaps_test": rows_gaps_test, "gaps_val": rows_gaps_val,
        "goat": rows_goat, "goat_di": rows_goat_di,
        "studio_maestro": rows_studio, "maestro": rows_maestro, "maps_full": rows_maps_full,
    }


TEST_SETS = ("smd", "mds", "guitarset", "gaps", "gaps_test", "gaps_val", "goat", "goat_di",
             "studio_maestro", "maestro", "maps_full")
# Store names of extract.py whose split is a test set under another name.
_STORE_SETS = {("gaps_ft", "test"): "gaps_test", ("gaps_ft", "val"): "gaps_val",
               ("gaps_ft_val", "test"): "gaps_val"}


def set_name(dataset, split):
    """The test set of utils.test_items that a store's split holds, or None
    when it holds none (a training split)."""
    if (dataset, split) in _STORE_SETS:
        return _STORE_SETS[(dataset, split)]
    if dataset in ("gaps_test", "gaps_val"):   # names that are a split themselves
        return dataset
    return dataset if split == "test" and dataset in TEST_SETS else None


def is_piano(name):
    """Piano sets carry the sustain pedal (CC64), extended into the reference notes."""
    return not name.startswith(("guitarset", "gaps", "goat"))


def midi_reference(name, cfg):
    """piece -> its reference notes, read from the set's MIDI files with the
    rule above, the reference every model is scored against."""
    midi = {item: m for item, _, m in test_items(name, cfg)}
    piano = is_piano(name)

    def reference(piece):
        if piece not in midi:
            raise SystemExit(f"{name}: no reference MIDI for piece {piece!r}")
        return read_midi_notes(midi[piece], piano)
    return reference


def test_items(name, cfg):
    """The (item_id, audio_path, midi_path) rows of one test set, sorted by id,
    checked against the study's list (study_pieces)."""
    build = _builders(cfg)
    if name not in build:
        raise SystemExit(f"unknown test set {name!r}; one of {', '.join(TEST_SETS)}")
    rows = sorted(build[name](), key=lambda r: r[0])
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name}: duplicate ids")
    ref = study_pieces(name)
    if set(ids) != ref:
        raise SystemExit(f"{name}: {len(ids)} items, but the study scores {len(ref)}: "
                         f"missing {sorted(ref - set(ids))[:5]} extra {sorted(set(ids) - ref)[:5]}")
    for _, a, m in rows:
        if not (a.exists() and m.exists()):
            raise SystemExit(f"{name}: missing file {a if not a.exists() else m}")
    return rows
