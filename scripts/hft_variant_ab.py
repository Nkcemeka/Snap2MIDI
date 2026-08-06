#!/usr/bin/env python
"""Does a faster kernel change hFT's answers? Asked on a trained checkpoint, on real audio.

scripts/hft_profile.py prices the speedups. This one decides whether they are
allowed, and it is the cheap substitute for retraining MAPS to find out.

It runs the *evaluation* path -- load_hft, half_stride, frames_to_note, the same
functions Evaluator.evaluate_hft calls, with the same defaults -- once per
variant on the same pieces, and reports two things per variant:

    max |dp|   the largest difference in any predicted probability, against the
               stock model. This is the sharp number: it sees a change of 1e-7
               that F1 would round away.

    note F1    per piece, and the delta. This is the number that ends up in the
               paper, so it is the one that decides.

Forward pass only, which is the point: it is the half of the change that a
trained model exposes and a fresh one hides. A random model outputs ~0.5
everywhere and would show nothing.

    python scripts/hft_variant_ab.py \
        --checkpoint save_dir/hft-epoch=00-valid_total_loss=0.1754.ckpt \
        --test-path data/hft_maestro/audio/test \
        --pieces 8 --variants sdpa,bf16,sdpa+bf16

Reading the result: identical F1 with max |dp| around 1e-6 is floating-point
noise and the variant is safe. A visibly different F1, or max |dp| in the 1e-2
range, means the variant changes the model's answers and has to be declared as
a deviation -- or dropped.

`--mode grad` covers the other half. A kernel can be exact forward and still
train differently, because gradients feed back into the weights and compound.
That mode loads the same checkpoint, takes one real unaugmented batch, and
compares the gradients themselves -- a far sharper signal than watching two
loss curves for 2000 steps, and it runs in seconds:

    python scripts/hft_variant_ab.py --mode grad \
        --checkpoint save_dir/hft-epoch=00-valid_total_loss=0.1754.ckpt \
        --base-path data/hft_maestro --variants sdpa,sdpa+compile
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from hft_profile import autocast_forward, loss_without_sync, sdpa_forward

import snap2midi.models.hft.hft as hft_module
from snap2midi.models.hft.evaluate import frames_to_note, half_stride, transcription_metrics
from snap2midi.models.hft.inference import load_hft, mel_from_audio

STOCK_ATTENTION = hft_module.MultiHeadAttention.forward


def eval_config(checkpoint, test_path):
    """Exactly Evaluator.evaluate_hft's defaults.

    Copied rather than imported because evaluate_hft goes straight on to run
    the whole 177-piece split, and the point here is a handful of pieces run
    several times. Any drift from those defaults makes the F1 printed below
    incomparable to the 0.880 the cluster checkpoint already scored.
    """
    note_min, note_max = 21, 108
    return dict(
        test_path=test_path, checkpoint_path=checkpoint,
        margin_b=32, margin_f=32, n_bins=256, n_slice=16,
        frame_threshold=0.5, onset_threshold=0.5, offset_threshold=0.5,
        num_frame=128, frame_rate=100, num_velocity=128,
        note_min=note_min, note_max=note_max, num_note=note_max - note_min + 1,
        hop_sample=256, sr=16000, fft_bins=2048, window_length=2048,
        mel_bins=256, log_offset=1e-8, pad_mode="constant",
        cnn_channel=4, cnn_kernel=5, d=256, pff_dim=512,
        enc_layer=3, dec_layer=3, enc_head=4, dec_head=4,
        dropout=0.1, weight_A=1.0, weight_B=1.0,
    )


def build(config, variant):
    """A model with `variant` applied. Weights are identical in every case."""
    # "fp32strict" is the control, not a candidate. It turns TF32 off, leaving
    # everything else alone -- so it measures how far a gradient moves under a
    # precision change the run *already makes deliberately*
    # (run_hft_paper.py sets matmul precision to "high", and argues there that
    # doing so is what matches the authors' torch 1.10 A100 run). Any candidate
    # whose gradient difference is at or below this one is inside a band the
    # experiment has already accepted as paper-faithful. Without it, a raw
    # percentage means nothing.
    torch.set_float32_matmul_precision(
        "highest" if "fp32strict" in variant else "high")

    hft_module.MultiHeadAttention.forward = (
        sdpa_forward if "sdpa" in variant else STOCK_ATTENTION)

    model = load_hft(config)  # already .eval(), so dropout is off everywhere

    if "bf16" in variant:
        model.forward = autocast_forward(model, torch.bfloat16)
    if "compile" in variant:
        model.hft_encoder = torch.compile(model.hft_encoder)
        model.hft_decoder = torch.compile(model.hft_decoder)
    return model


@torch.no_grad()
def transcribe(model, file, config):
    """One piece through the real inference path. Returns probabilities and F1."""
    data = np.load(file, allow_pickle=True)
    feature = mel_from_audio(torch.from_numpy(data["audio"]), config).numpy()

    output = half_stride(model, feature, shift=32, config=config)
    onset, offset, frames, velocity = output[-4], output[-3], output[-2], output[-1]

    notes = frames_to_note(onset, offset, frames, velocity, config)
    # "F-measure_no_offset" is onset+pitch only, which is the note F1 the hFT
    # paper reports and the 0.880 the cluster checkpoint already scored.
    metrics = transcription_metrics(notes, data["notes"])["F-measure_no_offset"]
    # The three sigmoid heads are what a kernel change moves; velocity is a
    # 128-way argmax downstream and would hide a small shift.
    probs = np.concatenate([np.asarray(onset).ravel(),
                            np.asarray(offset).ravel(),
                            np.asarray(frames).ravel()])
    return probs, metrics


def run_grad_ab(args, variants):
    """Compare the gradient itself, which is what actually trains the model.

    The note-F1 mode above only exercises the forward pass. A kernel can be
    exact forward and still train differently, because gradients feed back into
    the weights and compound. This asks the question directly: same checkpoint,
    same real batch, how far apart are the gradients?

    Dropout is off (model.eval()) on purpose. With it on, SDPA consumes the
    random stream differently from nn.Dropout on an explicit attention matrix,
    and that difference would swamp the kernel difference being measured. Off,
    the only thing left between the two paths is arithmetic.

    Read the relative L2 the way you would a tolerance: 1e-6 is float32 noise
    and the change is safe; 1e-2 means the two versions are computing
    materially different updates and the trial is not going to rescue it.
    """
    from torch.utils.data import DataLoader

    from hft_profile import hft_config
    from snap2midi.models.hft.hft_dataset import HFTDataset

    class Args:  # hft_config reads these five attributes
        base_path = args.base_path
        loader_batch = args.batch
        workers = 0
        augment = False
        augment_asset_root = None

    data_config = hft_config(Args())
    # Validation split: never augmented, so the batch is identical every time.
    dataset = HFTDataset(data_config, split="val")
    batch = next(iter(DataLoader(dataset, batch_size=args.batch, shuffle=False)))
    batch = [t.cuda() for t in batch]

    eval_cfg = eval_config(args.checkpoint, args.test_path)
    print(f"checkpoint : {Path(args.checkpoint).name}")
    print(f"batch      : {args.batch} unaugmented val excerpts from {args.base_path}\n")

    grads = {}
    for variant in variants:
        model = build(eval_cfg, variant)
        model.eval()  # dropout off -- see docstring
        model.zero_grad(set_to_none=True)
        loss = loss_without_sync(model, batch)
        loss.backward()
        grads[variant] = (
            torch.cat([p.grad.reshape(-1) for p in model.parameters()
                       if p.grad is not None]).double(),
            loss.item())
        del model
        torch.cuda.empty_cache()

    base_g, base_loss = grads["stock"]
    print(f"{'variant':<16} {'loss':>14} {'loss delta':>12} "
          f"{'grad rel L2':>12} {'max rel elem':>13}")
    print("-" * 72)
    print(f"{'stock':<16} {base_loss:>14.8f} {'-':>12} {'-':>12} {'-':>13}")
    for variant in variants[1:]:
        g, loss = grads[variant]
        rel = ((g - base_g).norm() / base_g.norm()).item()
        # Per-element, on the parameters that carry real gradient signal --
        # dividing by a near-zero gradient manufactures a huge ratio that says
        # nothing about the update the optimizer would actually apply.
        big = base_g.abs() > base_g.abs().max() * 1e-3
        max_rel = ((g - base_g).abs()[big] / base_g.abs()[big]).max().item()
        print(f"{variant:<16} {loss:>14.8f} {loss-base_loss:>+12.2e} "
              f"{rel:>12.2e} {max_rel:>13.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="notes", choices=["notes", "grad"],
                   help="'notes' compares decoded note F1 (forward only); "
                        "'grad' compares the gradients (the backward path)")
    p.add_argument("--base-path", default="data/hft_maestro",
                   help="extracted store, for --mode grad")
    p.add_argument("--batch", type=int, default=8, help="batch size for --mode grad")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-path", default="data/hft_maestro/audio/test")
    p.add_argument("--pieces", type=int, default=8,
                   help="how many test pieces; they are the same for every variant")
    p.add_argument("--variants", default="sdpa,bf16,sdpa+bf16",
                   help="comma separated, each a '+' joined set of sdpa/bf16/compile")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    variants = ["stock"] + [v for v in args.variants.split(",") if v]

    if args.mode == "grad":
        run_grad_ab(args, variants)
        return

    config = eval_config(args.checkpoint, args.test_path)
    files = sorted(Path(args.test_path).glob("*.npz"))[: args.pieces]
    if not files:
        raise SystemExit(f"no .npz test pieces under {args.test_path}")

    print(f"checkpoint : {Path(args.checkpoint).name}")
    print(f"pieces     : {len(files)} from {args.test_path}\n")

    # Per piece, per variant, so a piece that fails to decode can be dropped
    # from *every* variant rather than from one -- otherwise the comparison
    # silently stops being paired.
    #
    # frames_to_note ends in a bare `assert False` when a decoded note has zero
    # duration (utilities.py:289). On a fully trained model that never fires;
    # on a part-trained checkpoint it does, and it takes the whole evaluation
    # with it. That is a real bug worth fixing in the library, but not a reason
    # this comparison cannot run today.
    raw, timings = {}, {}
    for variant in variants:
        model = build(config, variant)
        started = time.perf_counter()
        raw[variant] = {}
        for f in files:
            try:
                raw[variant][f] = transcribe(model, f, config)
            except AssertionError as exc:
                raw[variant][f] = None
                print(f"  [{variant}] {f.stem[:40]}: decode failed -- "
                      f"{str(exc).splitlines()[0][:80]}")
        timings[variant] = time.perf_counter() - started
        del model
        torch.cuda.empty_cache()

    usable = [f for f in files if all(raw[v][f] is not None for v in variants)]
    dropped = len(files) - len(usable)
    if not usable:
        raise SystemExit("every piece failed to decode; nothing to compare.")
    if dropped:
        print(f"\n{dropped} of {len(files)} pieces dropped (decode failure in at "
              f"least one variant); comparing on the {len(usable)} that all "
              f"variants transcribed.\n")
    files = usable

    results = {
        v: (np.concatenate([raw[v][f][0] for f in files]),
            np.array([raw[v][f][1] for f in files]),
            timings[v])
        for v in variants
    }
    base_probs, base_f1, base_t = results["stock"]
    print(f"{'variant':<14} {'mean note F1':>13} {'delta F1':>10} "
          f"{'max |dp|':>10} {'pieces changed':>15} {'secs':>7}")
    print("-" * 74)
    print(f"{'stock':<14} {base_f1.mean():>13.6f} {'-':>10} {'-':>10} "
          f"{'-':>15} {base_t:>7.1f}")

    for variant in variants[1:]:
        probs, f1, elapsed = results[variant]
        dp = np.abs(probs - base_probs).max()
        changed = int((np.abs(f1 - base_f1) > 5e-4).sum())
        print(f"{variant:<14} {f1.mean():>13.6f} {f1.mean()-base_f1.mean():>+10.6f} "
              f"{dp:>10.2e} {changed:>15d} {elapsed:>7.1f}")

    print("\nper piece note F1")
    print(f"{'piece':<44} " + " ".join(f"{v:>11}" for v in variants))
    for i, f in enumerate(files):
        name = f.stem[:42]
        print(f"{name:<44} " + " ".join(f"{results[v][1][i]:>11.6f}" for v in variants))


if __name__ == "__main__":
    main()
