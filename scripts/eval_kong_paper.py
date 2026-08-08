"""Score a Kong run per piece, so checkpoints can be ranked on note F1 with CIs.

Runs the same decode as snap2midi.models.kong.evaluate.evaluate -- same 10 s
windows at 50% overlap, same stitch, same get_note_events at the config's
thresholds -- but keeps every piece's score instead of collapsing to a mean.
The library function returns only the averages, so no confidence interval can
be recovered from its output and two checkpoints half a point apart cannot be
told from a tie.

Frame F1 here thresholds the frame head directly (>= --frame-threshold), which
is what Kong et al. report. That is NOT what scripts/eval_hpp_paper.py's
frame_f1 measures -- that one rasterises frames back from decoded notes. Compare
each model's frame row to its own paper, never to the other model's.

Usage:
    .venv/bin/python scripts/eval_kong_paper.py --dataset maestro-val \
        --checkpoint save_dir/kong_paper_aug/kong-step=step=193868-...ckpt
    .venv/bin/python scripts/eval_kong_paper.py --dataset maestro
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pretty_midi
import torch
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel
from mir_eval.util import midi_to_hz
from sklearn.metrics import precision_recall_fscore_support as prfs
from tqdm import tqdm

from snap2midi.models.kong.evaluate import get_frames, get_midi_note_events
from snap2midi.models.kong.utilities import (extend_pedal, get_note_events,
                                             load_extract_config, load_kong,
                                             stitch)

DATASETS = {
    "maestro": "data/kong_maestro/test",
    "maps": "data/kong_maps/test",
    # The split to select a checkpoint on. Ranking candidates on test would be
    # choosing the number being reported.
    "maestro-val": "data/kong_maestro/val",
}

# Architecture defaults, matching Evaluator.evaluate_kong. Only used when the
# checkpoint carries no hyper_parameters block.
DEFAULTS = {"cmp": 48, "factors": [16, 32, 32], "momentum": 0.01}


def resolve_best_checkpoint(last_ckpt: str) -> str:
    """The checkpoint ModelCheckpoint ranked best, i.e. min valid_total_loss.

    Only a fallback: that ranking is the one this whole exercise exists to
    replace, so a sweep should pass --checkpoint explicitly.
    """
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    cb = [v for k, v in state["callbacks"].items() if "ModelCheckpoint" in k][0]
    best = cb.get("best_model_path")
    if not best:
        raise SystemExit(f"{last_ckpt} records no best_model_path -- pass "
                         "--checkpoint explicitly.")
    return best


def build_config(checkpoint: str, test_path: str, ext_config_path: str,
                 thresholds: dict) -> dict:
    """Architecture from the checkpoint's own hparams, thresholds from the CLI."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained = state.get("hyper_parameters", {}).get("config", {})
    config = {k: trained.get(k, v) for k, v in DEFAULTS.items()}
    config.update({
        "checkpoint_note_path": checkpoint,
        "test_path": test_path,
        "ext_config_path": ext_config_path,
        **thresholds,
    })
    return config


def step_of(checkpoint: str) -> str:
    """'step193868' from 'kong-step=step=193868-loss=valid_total_loss=0.7016'."""
    match = re.search(r"step=(\d+)", Path(checkpoint).stem)
    return f"step{match.group(1)}" if match else Path(checkpoint).stem.replace("=", "")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to the best checkpoint named in --last-ckpt.")
    parser.add_argument("--last-ckpt", default="save_dir/kong_paper_aug/last.ckpt")
    parser.add_argument("--onset-threshold", type=float, default=0.3)
    parser.add_argument("--offset-threshold", type=float, default=0.3)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N pieces (sorted, so the "
                             "subset is identical across checkpoints). For "
                             "ranking a sweep cheaply before confirming the "
                             "leaders on the full split.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or resolve_best_checkpoint(args.last_ckpt)
    test_path = DATASETS[args.dataset]
    if not Path(test_path).is_dir():
        raise SystemExit(f"{test_path} does not exist -- extract it first.")
    # The run's own name, so the A/B arms and the paper run land in separate
    # files without a flag.
    run = Path(checkpoint).parent.name

    if args.out:
        out_path = Path(args.out)
    elif args.checkpoint or args.limit:
        # A sweep run: name the file after the candidate, or each iteration
        # overwrites the last.
        suffix = f"_first{args.limit}" if args.limit else ""
        out_path = Path(f"results/sweep/kong_{args.dataset}{suffix}_{step_of(checkpoint)}.json")
    else:
        out_path = Path(f"results/{run}_{args.dataset}_per_piece.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext_config_path = f"{Path(test_path).parent}/extraction_config.h5"
    extraction_config = load_extract_config(ext_config_path)
    thresholds = {"onset_threshold": args.onset_threshold,
                  "offset_threshold": args.offset_threshold,
                  "frame_threshold": args.frame_threshold}
    config = build_config(checkpoint, test_path, ext_config_path, thresholds)

    frame_rate = extraction_config["frame_rate"]
    sample_rate = extraction_config["sample_rate"]
    min_pitch = extraction_config["min_pitch"]
    max_pitch = extraction_config["max_pitch"]
    window_samples = int(extraction_config["window_size"] * sample_rate)
    hop_samples = window_samples // 2

    # Sorted, unlike the library's bare glob: --limit has to hand every
    # candidate the same pieces for the comparison to be paired.
    test_files = sorted(glob.glob(f"{test_path}/*.h5"))
    if args.limit:
        test_files = test_files[:args.limit]
    names = [Path(f).stem for f in test_files]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_kong(config)

    print(f"checkpoint : {checkpoint}")
    print(f"test set   : {test_path} ({len(test_files)} pieces)")
    print(f"thresholds : onset {args.onset_threshold} offset "
          f"{args.offset_threshold} frame {args.frame_threshold}")

    scores = defaultdict(list)
    for file in tqdm(test_files, desc=f"kong/{args.dataset}"):
        with h5py.File(file, 'r') as hf:
            audio = hf["audio"][:]
            midi_path = hf.attrs["midi_path"]
            audio = (audio / 32767.0).astype(np.float32)

        midi = pretty_midi.PrettyMIDI(midi_path)
        if extraction_config["extend_pedal"]:
            midi = extend_pedal(midi)

        result_dict: dict = defaultdict(list)
        for start in range(0, len(audio), hop_samples):
            audio_segment = audio[start:start + window_samples]
            if len(audio_segment) < window_samples:
                audio_segment = np.pad(audio_segment,
                                       (0, window_samples - len(audio_segment)))
            output_dict = model(torch.tensor(audio_segment.reshape(1, -1)).to(device))
            for key in output_dict:
                result_dict[key].append(output_dict[key].cpu().detach().numpy())

        for key in result_dict:
            result_dict[key] = stitch(np.concatenate(result_dict[key]))[:len(audio)]

        duration = len(audio) / sample_rate
        ref_note_events = get_midi_note_events(midi, 0, duration)
        est_note_events = get_note_events(result_dict, args.onset_threshold,
                                          args.offset_threshold,
                                          args.frame_threshold, frame_rate)

        # A checkpoint early enough to decode nothing would crash the library's
        # evaluate; here it scores 0 and the sweep carries on.
        if est_note_events is None or len(est_note_events) == 0:
            est_note_events = np.zeros((0, 4))

        # Drop the events whose offset precedes their onset -- mir_eval rejects
        # them outright, so the library discards them too.
        locs = np.where(est_note_events[:, 1] < est_note_events[:, 0])[0]
        if len(locs) > 0:
            est_note_events = np.delete(est_note_events, locs, axis=0)

        est_ints = est_note_events[:, :2]
        est_notes = midi_to_hz(est_note_events[:, 2] + min_pitch)
        est_vels = est_note_events[:, 3] * 128
        ref_ints = ref_note_events[:, :2]
        ref_notes = midi_to_hz(ref_note_events[:, 2])
        ref_vels = ref_note_events[:, 3]

        p, r, f, _ = prf(ref_ints, ref_notes, est_ints, est_notes, offset_ratio=None)
        scores["note_no_offset_precision"].append(p)
        scores["note_no_offset_recall"].append(r)
        scores["note_no_offset_f1"].append(f)

        p, r, f, _ = prf(ref_ints, ref_notes, est_ints, est_notes)
        scores["note_precision"].append(p)
        scores["note_recall"].append(r)
        scores["note_f1"].append(f)

        p, r, f, _ = prf_vel(ref_ints, ref_notes, ref_vels, est_ints, est_notes, est_vels)
        scores["note_vel_precision"].append(p)
        scores["note_vel_recall"].append(r)
        scores["note_vel_f1"].append(f)

        ref_frame_roll = get_frames(midi, 0, duration, frame_rate, max_pitch, min_pitch)
        est_frame_roll = (result_dict["frame_roll"] >= args.frame_threshold).astype(int)
        est_frame_roll = est_frame_roll[:ref_frame_roll.shape[0], :]
        ref_frame_roll = ref_frame_roll[:est_frame_roll.shape[0], :]

        # labels=[0, 1] pins the positive class to index 1 even for a piece
        # where one of the two rolls is entirely empty.
        frame = prfs(ref_frame_roll.flatten(), est_frame_roll.flatten(),
                     labels=[0, 1], zero_division=0)
        scores["frame_precision"].append(frame[0][1])
        scores["frame_recall"].append(frame[1][1])
        scores["frame_f1"].append(frame[2][1])

        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {k: float(np.mean(v)) for k, v in scores.items()}
    out_path.write_text(json.dumps({
        "checkpoint": checkpoint,
        "test_path": test_path,
        "thresholds": thresholds,
        "pieces": names,
        "summary": summary,
        "scores": {k: [float(x) for x in v] for k, v in scores.items()},
    }, indent=1))

    print(f"\n=== {args.dataset} ({len(names)} pieces) ===")
    for key in sorted(summary):
        print(f"  {key:28s} {100 * summary[key]:6.2f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
