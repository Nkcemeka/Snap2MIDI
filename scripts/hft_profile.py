#!/usr/bin/env python
"""Measure where hFT's 22.6 h/epoch actually goes, and what each fix is worth.

Nothing here changes the library. It builds the *same* HFT module the training
builds, runs the *same* training_step, and times it -- so a number this script
prints is a number the real run would get.

Three questions, three modes:

    gpu     What does one optimizer step cost, at what batch size, in what
            precision, and how much of the card is actually working? Sweeps
            batch sizes until OOM, reports peak memory and achieved TFLOP/s
            next to the projected h/epoch and days-for-20-epochs.

    loader  What can the input pipeline deliver, in excerpts/s, with the real
            slab and the real augmentation? This is the ceiling the GPU work
            cannot beat. Worth knowing *before* speeding the GPU up 3x and
            discovering the dataloader was the wall at 2x.

    both    gpu then loader, and says which one binds.

The variant flags (--precision, --sdpa, --compile, --step nolog) monkeypatch
this process only. They exist so every proposed optimisation can be priced on
the cluster's own hardware before anyone edits snap2midi/.

Typical use on Pirineus, inside an interactive allocation on one H100:

    srun -p res_gpu -A upf105_b --gres=gpu:1 --cpus-per-task=32 --time=1:00:00 \
         --pty python scripts/hft_profile.py both \
         --base-path /data/upf105/resh000979/hft_maestro \
         --augment-asset-root /data/upf105/resh000979/mtg_ir_datasets

    # then price the fixes:
    python scripts/hft_profile.py gpu --batch 8,16,32,64 --precision bf16 --sdpa

The baseline to beat, measured on job 3100103: 3.9 it/s train and 14.7 it/s
validation at batch 8, i.e. 22.6 h/epoch, 19 days for 20 epochs.
"""

import argparse

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from snap2midi.models.hft.hft import HFT, MultiHeadAttention

# Measured on the trial job, at batch 8. Everything the script projects is
# anchored to these so a new number is directly comparable to the 19 days.
BASELINE_TRAIN_IT_S = 3.9
BASELINE_VAL_IT_S = 14.7
BASELINE_BATCH = 8

# Excerpts in one pass of each split, at n_slice=16 over MAESTRO v3. Fixed by
# the store, not by the batch size: raising the batch cuts the *step* count,
# never the work. Overridden from the real idx files when --base-path is given.
TRAIN_EXCERPTS = 2_245_272
VAL_EXCERPTS = 274_184
VALS_PER_EPOCH = 4  # val_check_interval=0.25


def hft_config(args) -> dict:
    """The config run_hft_paper.py produces, as far as HFT and HFTDataset read it.

    Kept as one literal rather than importing the Trainer wrapper, because the
    wrapper's job is to call pl.Trainer and that is exactly what this script
    must not do. Any key here that disagrees with run_hft_paper.py makes the
    measurement a measurement of something else -- keep the two in step.
    """
    return dict(
        project_name="snap2midi",
        experiment_name="HFT",
        base_path=args.base_path,
        batch_size=args.loader_batch,
        n_div_train=1,
        n_div_val=1,
        margin_b=32,
        margin_f=32,
        n_bins=256,
        num_note=88,
        num_velocity=128,
        num_frame=128,
        n_slice=16,
        epochs=20,
        frame_rate=100,
        lr=1e-4,
        dropout=0.1,
        clip_gradient_norm=1.0,
        seed=1234,
        cnn_channel=4,
        cnn_kernel=5,
        d=256,
        pff_dim=512,
        enc_layer=3,
        dec_layer=3,
        enc_head=4,
        dec_head=4,
        weight_A=1.0,
        weight_B=1.0,
        num_workers=args.workers,
        augment=args.augment,
        augment_asset_root=args.augment_asset_root,
        augment_manifest_dir=None,
        feature_source="audio",
        reverb_level="rms",
    )


# --------------------------------------------------------------------------
# variants: each one is a thing we might change in snap2midi/, priced here first
# --------------------------------------------------------------------------

def sdpa_forward(self, query, key, value):
    """MultiHeadAttention.forward via scaled_dot_product_attention.

    The stock implementation materialises the full (B, heads, Lq, Lk) softmax
    and hands it to dropout, so *both* tensors are kept for backward. In the
    encoder that is (8*128, 4, 256, 256) fp32 = 1.07 GB per layer per copy --
    which is why batch 8 is where the memory goes, and why the arithmetic units
    spend their time waiting on HBM.

    SDPA computes the same function without ever writing that matrix. Its
    default scale is 1/sqrt(head_dim), which is the division the stock code
    does by hand, and dropout_p applies to the same attention weights -- so
    this is the same maths, not an approximation. The flash kernel needs fp16
    or bf16; in fp32 SDPA falls back to the memory-efficient backend, still a
    win on memory but a smaller one on time. Measure both.

    The attention weights are returned by every caller and consumed by exactly
    nobody -- training_step names the decoder's `attention` and drops it, and
    the inference path never asks for it. Returning an empty tensor with the
    right rank keeps the decoder's reshape valid at zero cost, and is the
    signal that if this becomes a library change, the return value should go.
    """
    batch_size = query.size(0)
    q = self.fc_query(query).view(batch_size, -1, self.num_heads, self.dh).transpose(1, 2)
    k = self.fc_key(key).view(batch_size, -1, self.num_heads, self.dh).transpose(1, 2)
    v = self.fc_value(value).view(batch_size, -1, self.num_heads, self.dh).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=self.dropout.p if self.training else 0.0)

    out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d)
    empty_attn = q.new_empty((batch_size, self.num_heads, q.size(2), 0))
    return self.fc_out(out), empty_attn


def autocast_forward(model, dtype):
    """Run the network in `dtype`, hand the losses fp32 tensors.

    Not optional tidiness: nn.BCELoss raises under autocast, by design --
    a sigmoid computed in bf16 can land on exactly 1.0, and BCE's gradient
    there is 1/p. Casting back before the loss is the minimum safe form of
    mixed precision for this model. The real fix is for the decoder to emit
    logits and the loss to be BCEWithLogits (identical maths, computed
    stably), with the sigmoid moved to inference -- a library change, so it is
    not done here. Treat a bf16 number from this script as an upper bound that
    still needs that change to be trusted for a 19-day run.

    Index 4 of the output is the attention map, which no loss touches; casting
    it would allocate 370 MB per step to no purpose.
    """
    inner = model.forward

    def forward(spectrogram):
        with torch.autocast("cuda", dtype=dtype):
            outs = inner(spectrogram)
        return tuple(o if i == 4 else o.float() for i, o in enumerate(outs))

    return forward


def loss_without_sync(model, batch):
    """training_step's maths, without the nine .item() calls.

    HFT.training_step builds its log dict from `loss.item()` x9. Every one of
    those is a device sync: the CPU stops and waits for the GPU to drain before
    it can queue the next step's kernels, so the launch pipeline empties once
    per step. Lightning can log the tensors themselves and read them later.

    This replica exists to price that. It is verified against the real
    training_step at startup rather than assumed equal -- see check_replica.
    """
    spec, label_onset, label_offset, label_frames, label_velocity = batch
    (on1, off1, fr1, vel1, _attn, on2, off2, fr2, vel2) = model.forward(spec)

    n_vel = vel1.shape[-1]
    label_onset = label_onset.reshape(-1)
    label_offset = label_offset.reshape(-1)
    label_frames = label_frames.reshape(-1)
    label_velocity = label_velocity.reshape(-1)

    loss_1st = (model.bce(on1.reshape(-1), label_onset)
                + model.bce(off1.reshape(-1), label_offset)
                + model.bce(fr1.reshape(-1), label_frames)
                + model.ce(vel1.reshape(-1, n_vel), label_velocity))
    loss_2nd = (model.bce(on2.reshape(-1), label_onset)
                + model.bce(off2.reshape(-1), label_offset)
                + model.bce(fr2.reshape(-1), label_frames)
                + model.ce(vel2.reshape(-1, n_vel), label_velocity))
    return model.config["weight_A"] * loss_1st + model.config["weight_B"] * loss_2nd


def check_replica(model, batch):
    """Fail loudly if loss_without_sync has drifted from HFT.training_step."""
    model.eval()  # dropout off, so the two paths are comparable at all
    with torch.no_grad():
        real = model.training_step(batch, 0).item()
        mine = loss_without_sync(model, batch).item()
    model.train()
    if abs(real - mine) > 1e-4 * max(1.0, abs(real)):
        raise SystemExit(
            f"loss_without_sync no longer matches HFT.training_step "
            f"({mine} vs {real}). training_step changed; fix the replica "
            f"before trusting any number from --step nolog.")


# --------------------------------------------------------------------------
# gpu mode
# --------------------------------------------------------------------------

def synthetic_batch(config, batch_size, device):
    """One batch shaped exactly as HFTDataset.__getitem__ collates.

    spec is [n_bins, margin_b + num_frame + margin_f] per item -- the encoder
    unfolds dim 2, so the time axis is last. Values are random; the step time
    of a dense network does not depend on them, and using synthetic input is
    what separates the GPU question from the dataloader question.
    """
    n_time = config["margin_b"] + config["num_frame"] + config["margin_f"]
    g = torch.Generator(device=device).manual_seed(0)
    shape = (batch_size, config["num_frame"], config["num_note"])
    return (
        torch.randn(batch_size, config["n_bins"], n_time, generator=g, device=device),
        torch.randint(0, 2, shape, generator=g, device=device).float(),
        torch.randint(0, 2, shape, generator=g, device=device).float(),
        torch.randint(0, 2, shape, generator=g, device=device).float(),
        torch.randint(0, config["num_velocity"], shape, generator=g, device=device),
    )


def count_flops(model, batch, train: bool) -> float | None:
    """Matmul/conv FLOPs of one step, so 'is the GPU busy' has a real answer.

    nvidia-smi's utilisation is the fraction of time *any* kernel was resident.
    A model that spends its life in layernorms and softmaxes reads 100% there
    while using a few per cent of the arithmetic. FLOP/s divided by the card's
    peak is the number that tells you whether a bigger batch or a fused kernel
    has anything to win.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    counter = FlopCounterMode(display=False)
    with counter:
        if train:
            loss_without_sync(model, batch).backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                loss_without_sync(model, batch)
    return float(counter.get_total_flops())


def time_steps(fn, steps, warmup):
    """Median seconds per call, with the CUDA queue drained around each timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def build_model(config, args, device):
    torch.manual_seed(config["seed"])
    model = HFT(config).to(device)
    # training_step logs; there is no Trainer here to log into.
    model.log_dict = lambda *a, **k: None
    model.log = lambda *a, **k: None

    if args.sdpa:
        MultiHeadAttention.forward = sdpa_forward
    if args.compile:
        model.hft_encoder = torch.compile(model.hft_encoder)
        model.hft_decoder = torch.compile(model.hft_decoder)
    if args.precision == "bf16":
        model.forward = autocast_forward(model, torch.bfloat16)
    elif args.precision == "fp16":
        model.forward = autocast_forward(model, torch.float16)
    return model


def run_gpu(args, config, excerpts):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision(args.matmul)

    print(f"device      : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)")
    print(f"variant     : precision={args.precision} matmul={args.matmul} "
          f"sdpa={args.sdpa} compile={args.compile} step={args.step}")
    print(f"epoch       : {excerpts['train']:,} train excerpts, "
          f"{VALS_PER_EPOCH} x {excerpts['val']:,} validation\n")

    header = (f"{'batch':>6} {'train it/s':>11} {'excerpt/s':>10} {'val it/s':>9} "
              f"{'peak GiB':>9} {'TFLOP/s':>8} {'h/epoch':>8} {'days x20':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for batch_size in args.batch:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(config, args, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            batch = synthetic_batch(config, batch_size, device)

            if batch_size == args.batch[0] and args.step == "nolog":
                check_replica(model, batch)

            def train_step():
                optimizer.zero_grad(set_to_none=True)
                if args.step == "real":
                    loss = model.training_step(batch, 0)
                else:
                    loss = loss_without_sync(model, batch)
                loss.backward()
                optimizer.step()

            def val_step():
                with torch.no_grad():
                    loss_without_sync(model, batch)

            train_s = time_steps(train_step, args.steps, args.warmup)
            model.eval()
            val_s = time_steps(val_step, args.steps, args.warmup)
            model.train()

            peak = torch.cuda.max_memory_allocated() / 2**30
            flops = count_flops(model, batch, train=True)
            tflops = flops / train_s / 1e12 if flops else float("nan")

            train_h = excerpts["train"] * train_s / batch_size / 3600
            val_h = VALS_PER_EPOCH * excerpts["val"] * val_s / batch_size / 3600
            hours = train_h + val_h

            print(f"{batch_size:>6} {1/train_s:>11.2f} {batch_size/train_s:>10.1f} "
                  f"{1/val_s:>9.2f} {peak:>9.1f} {tflops:>8.1f} "
                  f"{hours:>8.1f} {hours*args.epochs/24:>9.1f}")
            rows.append(dict(batch=batch_size, train_it_s=1/train_s, val_it_s=1/val_s,
                             excerpts_s=batch_size/train_s, peak_gib=peak,
                             tflops=tflops, hours_per_epoch=hours,
                             train_hours=train_h, val_hours=val_h,
                             days=hours*args.epochs/24))
            del model, optimizer, batch
        except torch.cuda.OutOfMemoryError:
            print(f"{batch_size:>6} {'OOM':>11}")
            torch.cuda.empty_cache()

    base_hours = (excerpts["train"] / (BASELINE_TRAIN_IT_S * BASELINE_BATCH)
                  + VALS_PER_EPOCH * excerpts["val"] / (BASELINE_VAL_IT_S * BASELINE_BATCH)) / 3600
    print(f"\nmeasured baseline (job 3100103, H100 PCIe, fp32/tf32, batch 8): "
          f"{base_hours:.1f} h/epoch, {base_hours*args.epochs/24:.1f} days")
    if rows:
        best = min(rows, key=lambda r: r["hours_per_epoch"])
        print(f"best here: batch {best['batch']} at {best['hours_per_epoch']:.1f} h/epoch "
              f"({best['days']:.1f} days) = {base_hours/best['hours_per_epoch']:.2f}x baseline")
        print("note: 2 GPUs with DDP roughly halves the wall clock again, and "
              "doubles the effective batch.")
    return rows


# --------------------------------------------------------------------------
# loader mode
# --------------------------------------------------------------------------

def run_loader(args, config):
    """Excerpts/s the real input pipeline sustains, GPU untouched.

    Compare against the excerpts/s column above. If the loader number is the
    smaller one, no amount of kernel work shortens the run.
    """
    from snap2midi.models.hft.train_hft import HFTDataModule

    if not Path(config["base_path"]).exists():
        raise SystemExit(f"--base-path {config['base_path']} does not exist; "
                         f"loader mode needs the real store.")

    print(f"\nloader      : batch={config['batch_size']} workers={config['num_workers']} "
          f"augment={config['augment']}")
    dm = HFTDataModule(config)
    dm.setup("fit")
    loader = dm.train_dataloader()
    print(f"              {len(dm.train_dataset):,} train excerpts, "
          f"{len(loader):,} batches/epoch")

    iterator = iter(loader)
    for _ in range(args.warmup):  # worker startup and first page faults
        next(iterator)

    start = time.perf_counter()
    seen = 0
    for _ in range(args.batches):
        batch = next(iterator)
        seen += batch[0].shape[0]
    elapsed = time.perf_counter() - start

    rate = seen / elapsed
    print(f"              {rate:.1f} excerpts/s  ({rate/config['batch_size']:.2f} batches/s)")
    print(f"              per worker: {rate/max(1, config['num_workers']):.1f} excerpts/s")
    print(f"              an epoch of input alone: "
          f"{len(dm.train_dataset)/rate/3600:.1f} h")
    return rate


# --------------------------------------------------------------------------

def read_excerpt_counts(base_path, n_slice):
    """Excerpts per pass from the store itself, so projections are not folklore."""
    counts = {}
    for split, default in (("train", TRAIN_EXCERPTS), ("val", VAL_EXCERPTS)):
        files = sorted(Path(base_path, "idx", split).glob("dataset_idx*.npy"))
        if not files:
            counts[split] = default
            continue
        total = 0
        for f in files:
            n = len(np.load(f, mmap_mode="r"))
            total += (n // n_slice) if n_slice > 1 else n
        counts[split] = total
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["gpu", "loader", "both"])
    p.add_argument("--base-path", default="/data/upf105/resh000979/hft_maestro",
                   help="extracted store; read for excerpt counts, required by loader mode")
    p.add_argument("--batch", default="8,16,32,64",
                   help="batch sizes to sweep in gpu mode")
    p.add_argument("--loader-batch", type=int, default=8,
                   help="batch size for loader mode (does not affect its excerpts/s much)")
    p.add_argument("--workers", type=int, default=16, help="dataloader workers")
    p.add_argument("--steps", type=int, default=20, help="timed steps per configuration")
    p.add_argument("--warmup", type=int, default=5,
                   help="untimed steps first; raise to 20+ with --compile")
    p.add_argument("--batches", type=int, default=50, help="batches to time in loader mode")
    p.add_argument("--epochs", type=int, default=20, help="epochs the projection assumes")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="bf16/fp16 autocast the network, losses stay fp32 (see autocast_forward)")
    p.add_argument("--matmul", default="high", choices=["highest", "high", "medium"],
                   help="torch.set_float32_matmul_precision; 'high' is what run_hft_paper.py sets")
    p.add_argument("--sdpa", action="store_true",
                   help="attention via scaled_dot_product_attention instead of an explicit softmax")
    p.add_argument("--compile", action="store_true", help="torch.compile encoder and decoder")
    p.add_argument("--step", default="real", choices=["real", "nolog"],
                   help="'real' calls HFT.training_step; 'nolog' drops its nine .item() syncs")
    p.add_argument("--deterministic", action="store_true",
                   help="what pl.Trainer(deterministic=True) does to the process. The real "
                        "run sets it, and it is the thing most likely to reject a faster "
                        "kernel outright -- measure every variant with and without it")
    p.add_argument("--augment", action="store_true", default=True,
                   help="loader mode: augment as the paper run does (default on)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--augment-asset-root",
                   default="/data/upf105/resh000979/mtg_ir_datasets")
    p.add_argument("--json", help="also write the results here")
    args = p.parse_args()
    args.batch = [int(b) for b in args.batch.split(",")]

    if args.deterministic:
        # Both halves of what Lightning does for deterministic=True. The cuBLAS
        # variable has to be set before the first CUDA context or cuBLAS raises,
        # which is why this sits above every other import-time GPU touch.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    config = hft_config(args)
    excerpts = read_excerpt_counts(config["base_path"], config["n_slice"])

    results = {"args": vars(args), "excerpts": excerpts}
    if args.mode in ("gpu", "both"):
        if not torch.cuda.is_available():
            raise SystemExit("gpu mode needs a GPU.")
        results["gpu"] = run_gpu(args, config, excerpts)
    if args.mode in ("loader", "both"):
        results["loader_excerpts_s"] = run_loader(args, config)

    if args.mode == "both" and results.get("gpu"):
        gpu_rate = max(r["excerpts_s"] for r in results["gpu"])
        loader_rate = results["loader_excerpts_s"]
        binds = "the dataloader" if loader_rate < gpu_rate else "the GPU"
        print(f"\nbinding constraint: {binds} "
              f"(gpu {gpu_rate:.0f} excerpts/s vs loader {loader_rate:.0f} excerpts/s)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
