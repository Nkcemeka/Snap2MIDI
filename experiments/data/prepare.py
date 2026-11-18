#!/usr/bin/env python
"""Check every dataset of the study, and prepare the ones that need it.

Every dataset is first checked against what the study used (files present,
pieces counted), so an incomplete or different copy is reported here rather
than as a wrong number hours later. Five are then read by Snap2MIDI exactly as
released; three need a step first:

  maestro         check only
  maps            check only (the 270 MUS pieces of the nine pianos)
  smd             check only
  mds             check only
  goat            check only (the 170 direct-input takes of splits/goat_di_items.txt)
  gaps            decode to <prepared.gaps>/{train,val,test}/<id>.wav + <id>.midi,
                  with the study's split from splits/gaps_split.csv
  guitarset       convert the JAMS annotations to MIDI, then write single-track
                  copies for TransKun to <prepared.guitarset>/test/
  studio_maestro  pair the 177 recordings of MAESTRO's test split with their
                  MAESTRO MIDI in <prepared.studio_maestro>/<year>/
  (all)           every one of the above

GAPS. The release ships one directory per track (<id>.mp3, mostly 48 kHz,
and <id>-fine-aligned.mid) and an official train/test split. The study holds
27 train tracks out for validation (240 / 27 / 30, splits/gaps_split.csv; the
copy used holds 267 of the 270 official train tracks). Each mp3 is decoded
once with ffmpeg to 44.1 kHz mono 16-bit WAV (TransKun neither resamples nor
checks the rate) and the MIDI is copied as <id>.midi, the extension TransKun
globs for.
The split list is also written as <out>/fmeasure.csv, with each track's
score-alignment quality, where TransKun's createDatasetGAPS.py reads it.
Resumable: an existing WAV is kept.

GuitarSet. The release annotates in JAMS. They are converted to
<guitarset>/annotations-midi/<id>.mid, one instrument per string, by the same
library function the Snap2MIDI reader would otherwise call on first use. That
function draws each note's velocity at random between 95 and 104, so the seed
is fixed here; onsets, offsets and pitches do not depend on it. An existing
annotations-midi/ is kept as it is. TransKun asserts a single instrument, so
for it the strings are merged into <prepared.guitarset>/test/<id>.midi, with
the audio (already 44.1 kHz mono) symlinked beside it.

Studio MAESTRO (Edwards et al. 2024) re-records the 1276 MAESTRO performances
and ships audio only, named like MAESTRO's. The study scores the 177 of
MAESTRO's test split (splits/studio_maestro_items.txt); each is paired with
the original performance's MIDI as symlinks, which Snap2MIDI's MAESTRO reader
and the evaluator's test lists both pair by name. The downloaded datasets are not touched.

Usage:
    python experiments/data/prepare.py --dataset all [--config experiments/config.yaml]
"""

import argparse
import csv
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
GAPS_SPLIT = HERE / "splits/gaps_split.csv"
GOAT_DI = HERE / "splits/goat_di_items.txt"
FS = 44100
GUITARSET_SEED = 0


class Problem(Exception):
    """The copy of a dataset is not the one the study used."""


def expect(name, found, wanted):
    if found != wanted:
        raise Problem(f"{name}: found {found}, the study used {wanted}")


# ------------------------------------------------------------------ checks only

def maestro(cfg, args):
    root = cfg["datasets"]["maestro"]
    rows = list(csv.DictReader(open(root / "maestro-v3.0.0.csv")))
    missing = [r["midi_filename"] for r in rows
               if not ((root / r["audio_filename"]).exists() and (root / r["midi_filename"]).exists())]
    expect("MAESTRO v3 performances", len(rows), 1276)
    expect("MAESTRO performances without audio or MIDI", len(missing), 0)
    print("maestro: 1276 performances (177 test), read as released")


def maps(cfg, args):
    root = cfg["datasets"]["maps"]
    wavs = sorted(root.glob("*/MUS/*.wav"))
    expect("MAPS MUS pieces", len(wavs), 270)
    expect("MAPS MUS pieces without MIDI", sum(not w.with_suffix(".mid").exists() for w in wavs), 0)
    print("maps: 270 MUS pieces of nine pianos, read as released")


def smd(cfg, args):
    root = cfg["datasets"]["smd"]
    pairs = [m for m in (root / "midi").glob("*.mid")
             if (root / "wav_44100_stereo" / f"{m.stem}.wav").exists()]
    expect("SMD recordings with MIDI", len(pairs), 50)
    print("smd: 50 recordings, read as released")


def mds(cfg, args):
    root = cfg["datasets"]["mds"]
    n = 0
    for adir in (root / "Audio").iterdir():
        part = adir.name.rpartition("_")[0]
        n += sum((root / "MIDI" / part / f"{a.stem}.mid").exists() for a in adir.glob("*.wav"))
    expect("MDS recordings with MIDI", n, 212)
    print("mds: 212 recordings, read as released")


def goat(cfg, args):
    root = cfg["datasets"]["goat"]
    rows = {r["item"]: r for r in csv.DictReader(open(root / "metadata.csv"))}
    expect("GOAT performances", len(rows), 172)
    di = GOAT_DI.read_text().split()
    resolve = lambda rel: root / "data" / Path(*Path(rel.strip()).parts[1:])
    missing = [i for i in di if i not in rows or not resolve(rows[i]["di_audio_path"]).exists()]
    expect("GOAT direct-input takes of the study missing", len(missing), 0)
    print(f"goat: 172 performances, the {len(di)} DI takes of the study present, read as released")


# ------------------------------------------------------------------ preparation

def _decode(job):
    src, dst = job
    if dst.exists() and dst.stat().st_size > 0:
        return dst, "skip"
    tmp = dst.with_suffix(".wav.part")
    r = subprocess.run(
        # -f wav: the temp name ends in .part, so ffmpeg cannot infer the container.
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le", "-f", "wav", str(tmp)],
        capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return dst, f"FAIL {r.stderr.strip()[:120]}"
    tmp.rename(dst)
    return dst, "ok"


def gaps(cfg, args):
    G, out = cfg["datasets"]["gaps"], cfg["prepared"]["gaps"]
    rows = list(csv.DictReader(open(GAPS_SPLIT)))
    if args.only:
        only = set(args.only.split(","))
        rows = [r for r in rows if r["stem"] in only]
    jobs, missing = [], []
    for r in rows:
        sid = r["stem"]
        mp3 = G / "audio_tracks" / sid / f"{sid}.mp3"
        mid = G / "audio_tracks" / sid / f"{sid}-fine-aligned.mid"
        if not (mp3.exists() and mid.exists()):
            missing.append(sid)
            continue
        d = out / r["split"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mid, d / f"{sid}.midi")
        jobs.append((mp3, d / f"{sid}.wav"))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GAPS_SPLIT, out / "fmeasure.csv")

    print(f"gaps: decoding {len(jobs)} tracks to mono {FS} Hz WAV ...")
    fails = []
    with ProcessPoolExecutor(args.workers) as ex:
        for done, (dst, status) in enumerate(ex.map(_decode, jobs), 1):
            if status.startswith("FAIL"):
                fails.append(f"{dst.name}: {status}")
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    if fails:
        raise Problem(f"gaps: {len(fails)} decode failures, e.g. {fails[0]}")
    counts = {}
    for r in rows:
        if r["stem"] not in missing:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"gaps: {counts} in {out}")
    if missing:
        raise Problem(f"gaps: {len(missing)} track(s) of the study's split are not in "
                      f"{G / 'audio_tracks'}: {', '.join(missing)}; the other tracks are "
                      "prepared, but results on this copy will not match the paper exactly")


def guitarset(cfg, args):
    import jams
    import numpy as np
    import pretty_midi
    from snap2midi.extractor.modes.base_mode import _BaseMode

    G = cfg["datasets"]["guitarset"]
    audio_dir, midi_dir = G / "audio_mono-mic", G / "annotations-midi"
    expect("GuitarSet mono-mic recordings", len(list(audio_dir.glob("*_mic.wav"))), 360)
    if not midi_dir.is_dir():
        all_jams = sorted((G / "annotation").glob("*.jams"))
        expect("GuitarSet JAMS annotations", len(all_jams), 360)
        np.random.seed(GUITARSET_SEED)
        midi_dir.mkdir()
        for jam_path in all_jams:
            # The library's own conversion (no extractor state is involved).
            _BaseMode.jams_to_midi(None, jams.load(str(jam_path)), q=1).write(
                str(midi_dir / f"{jam_path.stem}.mid"))
        print(f"guitarset: converted {len(all_jams)} JAMS annotations to {midi_dir}")
    else:
        print(f"guitarset: keeping the existing {midi_dir}")

    out = cfg["prepared"]["guitarset"] / "test"
    out.mkdir(parents=True, exist_ok=True)
    pretty_midi.pretty_midi.MAX_TICK = 1e10
    written, merged = 0, 0
    for wav in sorted(audio_dir.glob("*.wav")):
        stem = wav.stem[:-4] if wav.stem.endswith("_mic") else wav.stem
        mid = midi_dir / f"{stem}.mid"
        if not mid.exists():
            raise Problem(f"guitarset: no annotation for {wav.name}")
        with wave.open(str(wav)) as f:
            if f.getframerate() != FS:
                raise Problem(f"guitarset: {wav.name} is {f.getframerate()} Hz, expected {FS}")
        pm = pretty_midi.PrettyMIDI(str(mid))
        notes = sorted((n for inst in pm.instruments for n in inst.notes),
                       key=lambda n: (n.start, n.pitch))
        merged += len(pm.instruments) > 1
        flat = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program("Acoustic Guitar (steel)"))
        inst.notes = notes
        flat.instruments.append(inst)
        flat.write(str(out / f"{stem}.midi"))
        link = out / f"{stem}.wav"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(wav.resolve())
        written += 1
    print(f"guitarset: {written} single-track copies for TransKun in {out} "
          f"({merged} merged from several strings)")


def studio_maestro(cfg, args):
    studio, mae = cfg["datasets"]["studio_maestro"], cfg["datasets"]["maestro"]
    out = cfg["prepared"]["studio_maestro"]
    wavs = sorted(studio.glob("*/*.wav"))
    expect("Studio MAESTRO recordings", len(wavs), 1276)
    keep = set((HERE / "splits/studio_maestro_items.txt").read_text().split())
    missing = []
    for wav in wavs:
        midi = mae / wav.parent.name / f"{wav.stem}.midi"
        if not midi.exists():
            missing.append(wav.name)
            continue
        if wav.stem not in keep:
            for at in (out / wav.parent.name / wav.name, out / wav.parent.name / midi.name):
                if at.is_symlink():
                    at.unlink()
            continue
        d = out / wav.parent.name
        d.mkdir(parents=True, exist_ok=True)
        for target, at in ((wav, d / wav.name), (midi, d / midi.name)):
            if at.is_symlink() or at.exists():
                at.unlink()
            at.symlink_to(target.resolve())
    expect("Studio MAESTRO recordings without MAESTRO MIDI", len(missing), 0)
    print(f"studio_maestro: {len(keep)} test-split recordings paired with their MAESTRO MIDI in {out}")


STEPS = {"maestro": maestro, "maps": maps, "smd": smd, "mds": mds, "goat": goat,
         "gaps": gaps, "guitarset": guitarset, "studio_maestro": studio_maestro}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=config.DEFAULT)
    ap.add_argument("--dataset", required=True, choices=sorted(STEPS) + ["all"])
    ap.add_argument("--workers", type=int, default=8, help="parallel GAPS decodes")
    ap.add_argument("--only", help="gaps: comma-separated track ids, for testing (use --only=...)")
    args = ap.parse_args()
    cfg = config.load(args.config)

    problems = []
    for name in STEPS if args.dataset == "all" else [args.dataset]:
        try:
            STEPS[name](cfg, args)
        except Problem as p:
            problems.append(str(p))
            print(f"PROBLEM {p}", file=sys.stderr)
    if problems:
        sys.exit(f"\n{len(problems)} dataset(s) differ from the study's copies; see above.")
    print("\nall checked datasets match the study's copies")


if __name__ == "__main__":
    main()
