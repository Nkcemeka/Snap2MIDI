"""The six models, one function each: what each produces for every test piece.

Every function runs its model on the test set and yields (piece, reference,
estimate), where reference and estimate are notes as (intervals in seconds,
pitches in Hz, velocities). score.py scores every model's pieces with the
same note metrics (metrics.py); nothing here computes a score.

Each function's docstring says how its model is run.
"""

import glob
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import PITCH_OFFSET, device, is_piano, notes_from_rolls, read_midi_notes

# The paper's decoding settings (Sec. 2.5), each model's own. Kong's threshold
# applies to its onset, offset and frame heads alike, hFT's to its onset,
# offset and frame thresholds; TransKun has a note bias instead (0: none).
THRESHOLD = {"oaf": 0.5, "oafv2": 0.5, "hpp": 0.4, "kong": 0.3, "hft": 0.5}


# hFT's and Kong's decoding at many thresholds (grids) is pure-Python work on
# CPU arrays, the same for every threshold: with parallel=True it is spread over
# the cores this process may use (GRID_WORKERS to override). The workers are
# forked after the forward pass and only decode, so the results are those of
# the serial loop, in its order. OAF, OAFv2 and HPP decode CUDA tensors, which a
# forked process cannot use: they stay serial.
_DECODE = None


def _decode_at(t):
    return _DECODE(t)


def _workers(n):
    return max(1, min(n, int(os.environ.get("GRID_WORKERS", len(os.sched_getaffinity(0))))))


def at_thresholds(threshold, decode, parallel=False):
    """decode(t) at one threshold, or {t: decode(t)} for a list of them: one
    forward pass decoded at every threshold (experiments/select/grid.py), in
    forked processes if parallel (decode must not touch the GPU)."""
    global _DECODE
    if not isinstance(threshold, (list, tuple)):
        return decode(threshold)
    workers = _workers(len(threshold)) if parallel else 1
    if workers == 1:
        return {t: decode(t) for t in threshold}
    _DECODE = decode
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            return dict(zip(threshold, pool.map(_decode_at, threshold, chunksize=1)))
    finally:
        _DECODE = None


def _one(threshold):
    """A single value for the settings a loader reads (decoding uses each)."""
    return threshold[0] if isinstance(threshold, (list, tuple)) else threshold


# ---------------------------------------------------------------- OAF (2018)

@torch.no_grad()
def oaf(checkpoint, items, threshold, limit=None, centre_onsets=False):
    """Reads the log-mel the extractor wrote (OAF has no in-model front end).
    A training split holds 5-20 s excerpts <piece>_<k>.npz, an evaluation set
    one whole-file window <piece>_0.npz; a piece split into several excerpts
    is yielded once per excerpt.

    centre_onsets: OAF's onset labels are floored to the frame (the frame that
    contains the onset), so an onset frame k stands for the interval
    [k, k+1) / frame_rate; its estimate is then read at the centre, (k + 1/2) /
    frame_rate. Offsets are left at the frame boundary (shifting them by the
    same rule would leave a one-frame note with no duration)."""
    from snap2midi.models.oaf.oaf import OnsetsAndFrames
    from snap2midi.utils.eval_mir import note_extract

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = state.get("hyper_parameters", {}).get("params")
    if not params:
        raise SystemExit(f"{checkpoint} carries no hyper_parameters.params: the "
                         "architecture cannot be rebuilt without guessing.")
    dev = device()
    model = OnsetsAndFrames(dict(params))
    model.load_state_dict(state["state_dict"])
    model.to(dev).eval()
    frame_rate = float(params.get("frame_rate", 31.25))

    by_piece = {}
    for path in sorted(Path(items).glob("*.npz")):
        by_piece.setdefault(re.sub(r"_\d+$", "", path.stem), []).append(path)
    pieces = sorted(by_piece)[:limit]
    for piece, path in tqdm([(q, p) for q in pieces for p in by_piece[q]], desc="oaf"):
        item = np.load(path)
        feature = torch.from_numpy(item["feature"].astype(np.float32)).to(dev).unsqueeze(0)
        labels = [torch.from_numpy(item[k].astype(np.float32)).to(dev)
                  for k in ("label_onsets", "label_frames", "label_velocities")]
        on, _, frame, vel = model(feature)

        def decode(t):
            # Velocities as the library's evaluator maps them.
            est = notes_from_rolls(note_extract, torch.sigmoid(on)[0], torch.sigmoid(frame)[0],
                                   vel[0], frame_rate, t,
                                   vel_scale=lambda v: 80 * np.clip(v, 0, 1) + 10)
            if centre_onsets:
                est[0][:, 0] += 0.5 / frame_rate
            return est
        yield piece, notes_from_rolls(note_extract, *labels, frame_rate), at_thresholds(threshold, decode)


# ---------------------------------------------------------------- OAFv2

def _oafv2_forward(model, mel, audio, hop_length):
    """Whole-piece forward (the BiLSTMs need the whole context); on GPU
    out-of-memory, 16384-frame chunks with 512 frames of context each side."""
    def run(segment):
        spec = mel(segment)
        spec = torch.log(torch.clamp(spec, min=1e-5)).transpose(-1, -2)
        on, off, _, frame, vel = model(spec)
        return on[0], frame[0], vel[0]

    try:
        return run(audio)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    chunk_frames, pad = 16384, 512
    outs = []
    n_frames = audio.shape[-1] // hop_length
    for start in range(0, n_frames, chunk_frames):
        end = min(start + chunk_frames, n_frames)
        lo, hi = max(0, start - pad), min(n_frames, end + pad)
        on, frame, vel = run(audio[..., lo * hop_length:hi * hop_length])
        a, b = start - lo, (start - lo) + (end - start)
        outs.append((on[a:b], frame[a:b], vel[a:b]))
        torch.cuda.empty_cache()
    return tuple(torch.cat([o[i] for o in outs]) for i in range(3))


@torch.no_grad()
def oafv2(checkpoint, items, threshold, limit=None):
    """Rebuilds the log-mel from the stored audio with the trained config."""
    from nnAudio2.features.mel import MelSpectrogram
    from snap2midi.models.oafv2.dataset_oafv2 import OAFV2Dataset
    from snap2midi.models.oafv2.inference import load_oafv2
    from snap2midi.utils.eval_mir import note_extract

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained = state.get("hyper_parameters", {}).get("config")
    if not trained:
        raise SystemExit(f"{checkpoint} carries no hyper_parameters.config: the "
                         "architecture cannot be rebuilt without guessing.")
    config = dict(trained)
    config.update({
        "checkpoint_path": str(checkpoint),
        "test_path": str(items),
        "threshold": _one(threshold),
        "pitch_offset": PITCH_OFFSET,
        "frame_rate": config["sample_rate"] / config["hop_length"],
        "sequence_length": None,   # whole tracks, not training excerpts
        "augment": False,
    })
    frame_rate, hop_length = config["frame_rate"], config["hop_length"]

    dataset = OAFV2Dataset(config, [str(items)])
    dataset.data = dataset.data[:limit]
    dev = device()
    model = load_oafv2(config)
    mel = MelSpectrogram(
        sr=config["sample_rate"], n_fft=config["n_fft"], n_mels=config["n_mels"],
        hop_length=hop_length, htk=config["htk"], fmin=config["fmin"],
        fmax=config["fmax"], pad_mode=config["pad_mode"], center=config["center"],
        window=config["window"],
    ).to(dev)
    for index in tqdm(range(len(dataset)), desc="oafv2"):
        item = dataset[index]
        audio = torch.from_numpy(item["audio"]).to(dev).unsqueeze(0)
        labels = [torch.from_numpy(item[k]).to(dev) for k in ("onset", "frame", "velocity")]
        rolls = _oafv2_forward(model, mel, audio, hop_length)
        est = at_thresholds(threshold,
                            lambda t: notes_from_rolls(note_extract, *rolls, frame_rate, t))
        yield dataset.data[index].stem, notes_from_rolls(note_extract, *labels, frame_rate), est
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- HPP

@torch.no_grad()
def _hpp_predict(model, audio, frame_num, hop_length, clip_len=10240):
    """Whole-track forward, in clips when it does not fit (HPPNet's own rule)."""
    if frame_num <= clip_len:
        torch.cuda.empty_cache()
        model.frame_num = frame_num
        return model(audio)
    clip_list = [clip_len] * (frame_num // clip_len)
    res = frame_num % clip_len
    if res != 0:
        clip_list[-1] -= (clip_len - res) // 2
        clip_list += [res + (clip_len - res) // 2]
    begin, predictions = 0, {}
    for clip in clip_list:
        end = begin + clip
        audio_i = audio[:, hop_length * begin:hop_length * end]
        torch.cuda.empty_cache()
        model.frame_num = clip
        for key, item in model(audio_i).items():
            item = item.squeeze()
            predictions[key] = torch.cat([predictions[key], item], dim=0) \
                if key in predictions else item
        begin += clip
    return predictions


@torch.no_grad()
def hpp(checkpoint, items, threshold, limit=None):
    """HPPNet-sp, the configuration of Evaluator.evaluate_hpp."""
    from torch.utils.data import DataLoader
    from snap2midi.models.hpp.dataset_hpp import HPPDataset
    from snap2midi.models.hpp.inference import load_hpp
    from snap2midi.utils.eval_mir import note_extract

    sample_rate, hop_length = 16000, 320
    config = {
        "checkpoint_path": str(checkpoint), "test_path": str(items),
        "sample_rate": sample_rate, "hop_length": hop_length,
        "frame_rate": sample_rate / hop_length, "bins_per_semitone": 4,
        "threshold": _one(threshold), "pitch_offset": PITCH_OFFSET,
        "sequence_length": None,   # whole tracks
        "SUBNETS_TO_TRAIN": ["onset_subnet", "frame_subnet"],
        "onset_subnet_heads": ["onset"],
        "frame_subnet_heads": ["frame", "offset", "velocity"],
        "fixed_dilation": 24, "model_size": 128,
    }
    dataset = HPPDataset(config, [str(items)])
    dataset.data = dataset.data[:limit]
    names = [p.stem for p in dataset.data]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    dev = device()
    model = load_hpp(config)
    for name, data in tqdm(zip(names, loader), total=len(names), desc="hpp"):
        labels = [data[k].to(dev)[0] for k in ("onset", "frame", "velocity")]
        predictions = _hpp_predict(model, data["audio"].to(dev), labels[1].size(-2), hop_length)
        rolls = [predictions[k].squeeze() for k in ("onset", "frame", "velocity")]
        est = at_thresholds(threshold, lambda t: notes_from_rolls(
            note_extract, *rolls, config["frame_rate"], t))
        yield name, notes_from_rolls(note_extract, *labels, config["frame_rate"]), est
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- Kong

def _kong_clean(events, pitch_is_relative, min_pitch, max_pitch):
    """mir_eval rejects zero-length notes, and a reference pitch outside the
    88 keys (GAPS has 109, MDS 112) cannot be predicted: both dropped."""
    if len(events) == 0:
        return events
    pitch = events[:, 2] + (min_pitch if pitch_is_relative else 0)
    keep = (events[:, 1] > events[:, 0]) & (pitch >= min_pitch) & (pitch <= max_pitch)
    return events[keep]


def _kong_decode(result, t, frame_rate, min_pitch, max_pitch):
    """Kong's note decoding (get_note_events) of one piece at threshold t; CPU only."""
    from mir_eval.util import midi_to_hz
    from snap2midi.models.kong.utilities import get_note_events
    est = get_note_events(result, t, t, t, frame_rate)
    est = _kong_clean(np.zeros((0, 4)) if est is None or len(est) == 0 else est, True,
                      min_pitch, max_pitch)
    return est[:, :2], midi_to_hz(est[:, 2] + min_pitch), est[:, 3] * 128


@torch.no_grad()
def kong(checkpoint, items, threshold, limit=None):
    """10 s windows at 50% overlap, stitched, decoded by get_note_events, as in
    snap2midi.models.kong.evaluate.evaluate. The grid comes from the store's
    extraction_config.h5. The forward pass runs here; the decoding, which is
    CPU-bound, runs in forked processes: across the thresholds of one piece for
    a grid, across pieces for a single threshold (yielded in piece order)."""
    from snap2midi.models.kong.utilities import load_extract_config, load_kong

    ext_config_path = f"{Path(items).parent}/extraction_config.h5"
    extraction_config = load_extract_config(ext_config_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained = state.get("hyper_parameters", {}).get("config", {})
    # Architecture defaults of Evaluator.evaluate_kong, for checkpoints that
    # carry no hyper_parameters block.
    defaults = {"cmp": 48, "factors": [16, 32, 32], "momentum": 0.01}
    config = {k: trained.get(k, v) for k, v in defaults.items()}
    config.update(checkpoint_note_path=str(checkpoint), test_path=str(items),
                  ext_config_path=ext_config_path, onset_threshold=_one(threshold),
                  offset_threshold=_one(threshold), frame_threshold=_one(threshold))

    frame_rate = extraction_config["frame_rate"]
    sample_rate = extraction_config["sample_rate"]
    min_pitch, max_pitch = extraction_config["min_pitch"], extraction_config["max_pitch"]
    window_samples = int(extraction_config["window_size"] * sample_rate)
    hop_samples = window_samples // 2
    dev = device()
    model = load_kong(config)

    single = not isinstance(threshold, (list, tuple))
    workers = _workers(len(os.sched_getaffinity(0))) if single else 1
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    pending = []   # (name, reference, async decode) of pieces not yet yielded

    def drain(keep):
        while len(pending) > keep:
            name, ref_notes, job = pending.pop(0)
            yield name, ref_notes, job.get()

    try:
        yield from _kong_pieces(model, dev, items, threshold, limit, extraction_config,
                                window_samples, hop_samples, frame_rate, sample_rate,
                                min_pitch, max_pitch, pool, pending, drain, workers)
        yield from drain(0)
    finally:
        if pool is not None:
            pool.terminate()


@torch.no_grad()
def _kong_pieces(model, dev, items, threshold, limit, extraction_config, window_samples,
                 hop_samples, frame_rate, sample_rate, min_pitch, max_pitch, pool, pending,
                 drain, workers):
    import h5py
    import pretty_midi
    from mir_eval.util import midi_to_hz
    from snap2midi.models.kong.evaluate import get_midi_note_events
    from snap2midi.models.kong.utilities import extend_pedal, stitch

    for file in tqdm(sorted(glob.glob(f"{items}/*.h5"))[:limit], desc="kong"):
        with h5py.File(file, "r") as hf:
            audio = (hf["audio"][:] / 32767.0).astype(np.float32)
            midi_path = hf.attrs["midi_path"]
        midi = pretty_midi.PrettyMIDI(midi_path)
        if extraction_config["extend_pedal"]:
            midi = extend_pedal(midi)

        result = {}
        for start in range(0, len(audio), hop_samples):
            segment = audio[start:start + window_samples]
            if len(segment) < window_samples:
                segment = np.pad(segment, (0, window_samples - len(segment)))
            for key, value in model(torch.tensor(segment.reshape(1, -1)).to(dev)).items():
                result.setdefault(key, []).append(value.cpu().detach().numpy())
        for key in result:
            result[key] = stitch(np.concatenate(result[key]))[:len(audio)]

        ref = _kong_clean(get_midi_note_events(midi, 0, len(audio) / sample_rate), False,
                          min_pitch, max_pitch)
        ref_notes = (ref[:, :2], midi_to_hz(ref[:, 2]), ref[:, 3])
        if pool is not None:
            pending.append((Path(file).stem, ref_notes, pool.apply_async(
                _kong_decode, (result, threshold, frame_rate, min_pitch, max_pitch))))
            yield from drain(2 * workers)
        else:
            yield (Path(file).stem, ref_notes, at_thresholds(
                threshold, lambda t: _kong_decode(result, t, frame_rate, min_pitch, max_pitch),
                parallel=True))
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- hFT, TransKun

# hFT: Inference.inference_hft's settings, the configuration it was trained with.
HFT_CONFIG = dict(
    margin_b=32, margin_f=32, n_bins=256, fft_bins=2048, window_length=2048,
    n_slice=16, frame_threshold=0.5, onset_threshold=0.5, offset_threshold=0.5,
    num_frame=128, epochs=50, frame_rate=100, shift=32, num_velocity=128,
    note_min=21, note_max=108, num_note=108 - 21 + 1, hop_sample=256,
    mel_bins=256, sr=16000, cnn_channel=4, cnn_kernel=5, d=256, pff_dim=512,
    enc_layer=3, dropout=0.1, dec_layer=3, enc_head=4, dec_head=4,
    weight_A=1.0, weight_B=1.0, log_offset=1e-8, pad_mode="constant",
)


def _transcribed(model, checkpoint, settings, rows, dataset, estimates, load, limit):
    """Transcribe every item of `rows` (utils.test_items) not yet in `estimates` (one MIDI per item,
    written to .part first so an interrupted run leaves no truncated file),
    then yield each item's reference and estimated notes. The folder records
    the checkpoint and settings it was made with and is never reused for others.
    The sustain pedal is extended on the piano sets only."""
    estimates = Path(estimates)
    estimates.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "checkpoint": str(Path(checkpoint).resolve()), **settings}
    record_path = estimates / "settings.json"
    if record_path.exists():
        if json.loads(record_path.read_text()) != record:
            raise SystemExit(f"{estimates} holds transcriptions made with "
                             f"{record_path.read_text()}, not {record}: use another folder")
    elif any(estimates.glob("*.mid")):
        raise SystemExit(f"{estimates} holds transcriptions of unknown origin: use another folder")
    else:
        record_path.write_text(json.dumps(record, indent=1))

    rows = rows[:limit]
    todo = [r for r in rows if not (estimates / f"{r[0]}.mid").exists()]
    print(f"{model} x {dataset}: {len(rows)} items, {len(todo)} to transcribe -> {estimates}")
    if todo:
        run = load()
        for item, audio, _ in tqdm(todo, desc=f"{model} transcription"):
            dst = estimates / f"{item}.mid"
            tmp = dst.with_suffix(".mid.part")
            run(audio).write(str(tmp))
            tmp.rename(dst)

    piano = is_piano(dataset)
    for item, _, midi in rows:
        yield item, read_midi_notes(midi, piano), read_midi_notes(estimates / f"{item}.mid", piano)


def _hft_runner(checkpoint, keep_zero_velocity):
    """(forward, decode): the library's hFT forward pass (log-mel front end,
    mono 16 kHz, half-stride inference) and its frames_to_note decoding at a
    threshold, at Inference.inference_hft's settings. The threshold enters at
    decoding only, so one forward pass serves every threshold."""
    from snap2midi.models.hft.inference import get_feature_hft, load_hft
    from snap2midi.models.hft.utilities import frames_to_note, half_stride, notes_to_midi
    cfg = dict(HFT_CONFIG, checkpoint_path=str(checkpoint), filename=None,
               keep_zero_velocity_notes=keep_zero_velocity)
    model = load_hft(cfg)

    @torch.no_grad()
    def forward(audio_path):
        c = dict(cfg, audio_path=str(audio_path))
        return half_stride(model, get_feature_hft(str(audio_path), c), c["shift"], c)

    def decode(output, threshold):
        c = dict(cfg, onset_threshold=threshold, offset_threshold=threshold,
                 frame_threshold=threshold)
        return notes_to_midi(frames_to_note(output[-4], output[-3], output[-2], output[-1], c))
    return forward, decode


def hft(checkpoint, rows, dataset, estimates, threshold, keep_zero_velocity=False,
        limit=None):
    """The library's hFT (_hft_runner) at one threshold. keep_zero_velocity
    keeps the notes decoded with velocity 0 (the adapted checkpoints' decoder)."""
    def load():
        forward, decode = _hft_runner(checkpoint, keep_zero_velocity)
        return lambda audio_path: decode(forward(audio_path), threshold)

    settings = {"threshold": threshold, "keep_zero_velocity": keep_zero_velocity}
    return _transcribed("hft", checkpoint, settings, rows, dataset, estimates, load, limit)


def _midi_notes(midi, piano):
    """Notes of an estimate as score.py reads a stored transcription: written
    to a MIDI file and read back (utils.read_midi_notes), so a grid scores
    exactly what a test run would."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "est.mid"
        midi.write(str(path))
        return read_midi_notes(path, piano)


def hft_grid(checkpoint, rows, dataset, thresholds, keep_zero_velocity=False, limit=None):
    """hFT at every threshold from one forward pass per item: yields (item,
    reference, {threshold: estimate})."""
    forward, decode = _hft_runner(checkpoint, keep_zero_velocity)
    piano = is_piano(dataset)
    for item, audio, midi in tqdm(rows[:limit], desc="hft grid"):
        output = forward(audio)
        yield (item, read_midi_notes(midi, piano),
               at_thresholds(list(thresholds), lambda t: _midi_notes(decode(output, t), piano),
                             parallel=True))


def _transkun_model(checkpoint, transkun_repo, state, dev):
    """TransKun 2.0 from its repository with the checkpoint's chosen state."""
    sys.path.insert(0, str(transkun_repo))
    import moduleconf
    cm = moduleconf.parseFromFile(str(Path(transkun_repo) / "transkun/pretrained/2.0.conf"))
    TransKun, conf = cm["Model"].module.TransKun, cm["Model"].config
    ck = torch.load(checkpoint, map_location=dev, weights_only=False)
    which = {"best": "best_state_dict", "last": "state_dict"}.get(
        state, "best_state_dict" if "best_state_dict" in ck else "state_dict")
    if which not in ck:
        raise SystemExit(f"{checkpoint} has no {which}; keys: {list(ck)}")
    model = TransKun(conf=conf).to(dev)
    model.load_state_dict(ck[which], strict=True)
    model.eval()
    return model


def _transkun_audio(audio_path, fs_model, dev):
    """The audio as TransKun reads it: float32, channels kept, soxr to its rate."""
    import soundfile as sf
    import soxr
    y, fs = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if fs != fs_model:
        y = soxr.resample(y, fs, fs_model)
    return torch.from_numpy(np.ascontiguousarray(y)).to(dev)


def transkun(checkpoint, rows, dataset, estimates, transkun_repo, note_bias=0.0,
             state="auto", limit=None):
    """TransKun from its official repository (transkun_repo, with the paper's
    patch applied): the 2.0 configuration, audio resampled to the model's rate
    with soxr, and its own semi-CRF decoding. note_bias is added to every
    interval score before decoding (0: none). A training checkpoint holds two
    states: `state` picks best_state_dict or state_dict (auto: best if present)."""
    def load():
        dev = device()
        model = _transkun_model(checkpoint, transkun_repo, state, dev)
        from transkun.Data import writeMidi
        if note_bias != 0.0:
            from transkun.CRF.NeuralSemiCRFInterval import (NeuralSemiCRFInterval,
                                                            viterbi, viterbiBackward)

            def decode_biased(self, forcedStartPos=None, forward=False):
                f = viterbi if forward else viterbiBackward
                return f(self.score + note_bias, self.noiseScore, forcedStartPos)
            NeuralSemiCRFInterval.decode = decode_biased

        @torch.no_grad()
        def run(audio_path):
            notes = model.transcribe(_transkun_audio(audio_path, model.fs, dev),
                                     discardSecondHalf=False)
            return writeMidi(notes)
        return run

    settings = {"note_bias": note_bias, "state": state}
    return _transcribed("transkun", checkpoint, settings, rows, dataset, estimates, load, limit)


def transkun_grid(checkpoint, rows, dataset, biases, transkun_repo, state="auto", limit=None):
    """TransKun at every note bias from one forward pass per segment
    (transkun_biases.transcribe_all_biases): yields (item, reference,
    {bias: estimate})."""
    dev = device()
    model = _transkun_model(checkpoint, transkun_repo, state, dev)
    from transkun.Data import writeMidi
    from transkun_biases import transcribe_all_biases
    piano = is_piano(dataset)
    for item, audio, midi in tqdm(rows[:limit], desc="transkun grid"):
        with torch.no_grad():
            notes = transcribe_all_biases(model, _transkun_audio(audio, model.fs, dev), biases)
        yield (item, read_midi_notes(midi, piano),
               {b: _midi_notes(writeMidi(notes[b]), piano) for b in biases})
