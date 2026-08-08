"""Score the augmented HPPNet-sp run against Wei et al. (ISMIR 2022).

Runs the same decode as snap2midi.models.hpp.evaluate.evaluate_test -- same
chunked forward, same note_extract at threshold 0.4, same 50 fps grid -- but
keeps every piece's score instead of only the mean, and computes an F1 per
piece rather than reporting bare Precision/Recall.

That last part is the reason this is a script and not a call into
Evaluator.evaluate_hpp: the library's frame block stores only
{'Precision', 'Recall'} (mir_eval.multipitch returns no F1), so the frame F1
the paper's Tables 3 and 4 report cannot be recovered from its output. F1 of
the averaged P and R is not the average of the per-piece F1s, and it is the
latter the paper quotes.

Usage:
    .venv/bin/python scripts/eval_hpp_paper.py --dataset maestro
    .venv/bin/python scripts/eval_hpp_paper.py --dataset maps
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NNAUDIO_DISABLE_CITATION_REMINDER", "1")

import numpy as np
import torch
from mir_eval.transcription import precision_recall_f1_overlap as prf
from mir_eval.transcription_velocity import precision_recall_f1_overlap as prf_vel
from torch.utils.data import DataLoader
from tqdm import tqdm

from snap2midi.models.hpp.dataset_hpp import HPPDataset
from snap2midi.models.hpp.inference import load_hpp
from snap2midi.utils.eval_mir import multipitch_metrics, note_extract, notes_to_frames

DATASETS = {
    "maestro": "data/hpp_maestro/test",
    "maps": "data/hpp_maps/test",
    # The split to select a checkpoint on. Wei et al. pick their model by
    # validation performance; ranking candidates on test would be choosing on
    # the number being reported.
    "maestro-val": "data/hpp_maestro/val",
}


def resolve_best_checkpoint(last_ckpt: str) -> str:
    """The best-scoring checkpoint, as ModelCheckpoint recorded it.

    Read from the callback state rather than guessed from the filenames: the
    template is 'hpp-{step}', so the file names carry no loss and the highest
    step is not the best model -- here it is step 247748, with two later
    checkpoints scoring worse.
    """
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    cb = [v for k, v in state["callbacks"].items() if "ModelCheckpoint" in k][0]
    best = cb.get("best_model_path")
    if not best:
        raise SystemExit(
            f"{last_ckpt} records no best_model_path -- it was written by an "
            "unranked periodic checkpointer, which is the point: pass "
            "--checkpoint explicitly and select on validation note F1.")
    return best


def build_config(checkpoint_path: str, test_path: str, threshold: float) -> dict:
    """The sp configuration, matching Evaluator.evaluate_hpp's sp branch."""
    sample_rate, hop_length = 16000, 320
    return {
        "checkpoint_path": checkpoint_path,
        "test_path": test_path,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "frame_rate": sample_rate / hop_length,
        "bins_per_semitone": 4,
        "threshold": threshold,
        "pitch_offset": 21,
        # None means the dataset hands back whole tracks, which is what the
        # chunked forward below expects.
        "sequence_length": None,
        "SUBNETS_TO_TRAIN": ["onset_subnet", "frame_subnet"],
        "onset_subnet_heads": ["onset"],
        "frame_subnet_heads": ["frame", "offset", "velocity"],
        "fixed_dilation": 24,
        "model_size": 128,
    }


@torch.no_grad()
def predict(model, audio, frame_num, hop_length, clip_len=10240):
    """Forward the whole track, in clips when it does not fit."""
    if frame_num <= clip_len:
        torch.cuda.empty_cache()
        model.frame_num = frame_num
        return model(audio)

    clip_list = [clip_len] * (frame_num // clip_len)
    res = frame_num % clip_len
    if res != 0:
        clip_list[-1] -= (clip_len - res) // 2
        clip_list += [res + (clip_len - res) // 2]

    begin = 0
    predictions = {}
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


def f1(precision: float, recall: float) -> float:
    """Harmonic mean, 0 when a piece scores nothing -- mir_eval's convention."""
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to the best checkpoint named in --last-ckpt.")
    parser.add_argument("--last-ckpt", default="save_dir/hpp_augmented/last.ckpt")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Onset and frame threshold. 0.4 is the paper's.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N pieces (sorted, so the "
                             "subset is identical across checkpoints). For "
                             "ranking a sweep cheaply before confirming the "
                             "leaders on the full split.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or resolve_best_checkpoint(args.last_ckpt)
    test_path = DATASETS[args.dataset]
    if args.out:
        out_path = Path(args.out)
    elif args.checkpoint or args.limit:
        # A sweep run: name the file after the candidate, or each iteration
        # overwrites the last. Kept out of results/ proper so the headline
        # per-piece files stay the ones compare_hpp_to_paper.py reads.
        tag = Path(checkpoint).stem.replace("=", "")
        suffix = f"_first{args.limit}" if args.limit else ""
        out_path = Path(f"results/sweep/{args.dataset}{suffix}_{tag}.json")
    else:
        out_path = Path(f"results/hpp_augmented_{args.dataset}_per_piece.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_config(checkpoint, test_path, args.threshold)
    frame_rate = config["frame_rate"]
    pitch_offset = config["pitch_offset"]

    dataset = HPPDataset(config, [test_path])
    if args.limit:
        # dataset.data is sorted, so the head is the same subset for every
        # candidate -- the comparison is paired even though it is partial.
        dataset.data = dataset.data[:args.limit]
    names = [p.stem for p in dataset.data]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_hpp(config)

    print(f"checkpoint : {checkpoint}")
    print(f"test set   : {test_path} ({len(dataset)} pieces)")
    print(f"threshold  : {args.threshold}")

    scores = defaultdict(list)
    for data in tqdm(loader, total=len(loader), desc=f"hpp/{args.dataset}"):
        y_onset = data["onset"].to(device)[0]
        y_frame = data["frame"].to(device)[0]
        y_velocity = data["velocity"].to(device)[0]

        predictions = predict(model, data["audio"].to(device), y_frame.size(-2),
                              config["hop_length"])
        on_preds = predictions["onset"].squeeze()
        frame_preds = predictions["frame"].squeeze()
        vel_preds = predictions["velocity"].squeeze()

        note_preds, int_preds, vels = note_extract(
            on_preds, frame_preds, vel_preds,
            onset_thresh=args.threshold, frame_thresh=args.threshold)
        int_preds = int_preds / frame_rate
        note_preds += pitch_offset

        note_gt, int_gt, vel_gt = note_extract(y_onset, y_frame, y_velocity)
        int_gt = int_gt / frame_rate
        note_gt += pitch_offset

        # Frames are rasterised back from the decoded notes, as the library's
        # evaluator does. It is not the paper's frame-head thresholding, so the
        # frame row of the comparison is the looser of the two.
        frame_pred = notes_to_frames(note_preds - pitch_offset,
                                     (int_preds * frame_rate).astype(int),
                                     on_preds.shape)
        frame_gt = notes_to_frames(note_gt - pitch_offset,
                                   (int_gt * frame_rate).astype(int), y_frame.shape)
        frame_pred = frame_pred[:frame_gt.shape[0], :]

        note_gt_hz = 440 * (2 ** ((note_gt - 69) / 12))
        note_preds_hz = 440 * (2 ** ((note_preds - 69) / 12))

        p, r, f, _ = prf(ref_intervals=int_gt, ref_pitches=note_gt_hz,
                         est_intervals=int_preds, est_pitches=note_preds_hz,
                         offset_ratio=None)
        scores["note_no_offset_precision"].append(p)
        scores["note_no_offset_recall"].append(r)
        scores["note_no_offset_f1"].append(f)

        p, r, f, _ = prf(ref_intervals=int_gt, ref_pitches=note_gt_hz,
                         est_intervals=int_preds, est_pitches=note_preds_hz)
        scores["note_precision"].append(p)
        scores["note_recall"].append(r)
        scores["note_f1"].append(f)

        p, r, f, _ = prf_vel(ref_intervals=int_gt, ref_pitches=note_gt_hz,
                             ref_velocities=vel_gt, est_intervals=int_preds,
                             est_pitches=note_preds_hz, est_velocities=vels,
                             velocity_tolerance=0.1)
        scores["note_vel_precision"].append(p)
        scores["note_vel_recall"].append(r)
        scores["note_vel_f1"].append(f)

        frame = multipitch_metrics(frame_gt, frame_pred, frame_rate)
        scores["frame_precision"].append(frame["Precision"])
        scores["frame_recall"].append(frame["Recall"])
        scores["frame_f1"].append(f1(frame["Precision"], frame["Recall"]))
        scores["frame_accuracy"].append(frame["Accuracy"])

        torch.cuda.empty_cache()

    summary = {k: float(np.mean(v)) for k, v in scores.items()}
    out_path.write_text(json.dumps({
        "checkpoint": checkpoint,
        "test_path": test_path,
        "threshold": args.threshold,
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
