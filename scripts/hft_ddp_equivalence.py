#!/usr/bin/env python
"""Does 2 ranks at batch N/2 really produce the gradient 1 rank at batch N does?

run_hft_trial.py asserts that it does, and argues it on paper:

    every loss term reduces with mean(), each rank holds the same element
    count, and DDP averages across ranks -- so mean(mean(A), mean(B)) =
    mean(A + B). Effective batch and step count are unchanged, which is why
    this costs no fidelity.

The argument is sound -- all eight terms are nn.BCELoss/nn.CrossEntropyLoss at
their default reduction="mean", and hFT normalises with LayerNorm, which is
per-sample, so no statistic depends on how the batch is split. But it had never
been run. A 7-day training commitment should not rest on an unexecuted
derivation, and the failure mode if it is wrong is the quiet kind: the run
completes, the loss curve looks plausible, and the model is simply worse than
the paper's for a reason nothing reports.

What this measures
------------------
Three gradients over the same weights and the same excerpts:

    full    one process, the whole batch at once -- the ground truth
    halves  one process, the mean of the per-slice gradients, computed by hand
    ddp     `ranks` processes at batch/ranks each, reduced by DDP

and three comparisons, which is the point of computing all three rather than
just the obvious pair:

    halves vs full    the mean-of-means algebra, with no distributed code in
                      the picture. Fails only if the claim above is wrong.
    ddp    vs halves  the distributed machinery, holding the algebra fixed.
                      Fails if DDP is reducing something other than the plain
                      average -- a gradient-as-sum bucket, an uneven shard, a
                      parameter excluded from the reduction.
    ddp    vs full    the headline. What the trial actually relies on.

A single number could only say "different"; the decomposition says which half
of the claim broke, and those two have entirely different fixes.

Reading the result
------------------
Exact equality is the wrong expectation. Summing eight floats in one order and
in two halves gives answers that differ in the last bits, so the honest target
is "at float32 noise", not zero. A bare 1e-7 means nothing on its own, so the
run also prints a reference scale: the gradient from a *different* batch of the
same size. That is what a genuinely different update looks like. The
equivalence is credible when the three comparisons sit orders of magnitude
below that reference, and the verdict below checks exactly that rather than a
hardcoded tolerance alone.

Why CPU by default
------------------
The claim is about reduction order and is device-independent, so CPU tests it
just as well while keeping TF32 and cuDNN algorithm selection -- both of which
vary run to run -- out of a measurement that is looking for small differences.
It also means this never contends for the GPU with a training job.

--device cuda is available and uses gloo there too. NCCL is deliberately not an
option: two ranks sharing one device is a documented way to deadlock it, and
the point here is arithmetic, not collective throughput.

Dropout is off (model.eval()), as in scripts/hft_variant_ab.py --mode grad. On,
each rank would draw its own masks and the comparison would measure the RNG
rather than the reduction.

Usage
-----
    scripts/hft_ddp_equivalence.py --checkpoint save_dir/some.ckpt

    # the production shape: 2x4 against 1x8, fast attention and compile on
    scripts/hft_ddp_equivalence.py --checkpoint save_dir/some.ckpt \
        --batch 8 --ranks 2 --fast-attention --compile

--compile is the reason to run this a second time. scripts/hft_variant_ab.py
measures compile in one process, and this script without --compile measures DDP
in eager mode; the run does both at once, and that is its own code path --
Dynamo splits a compiled graph at gradient-bucket boundaries so the all-reduce
can overlap with compute, which cannot happen when either half is absent.
Neither of the two single-factor results implies it.

Exit status is 0 when the equivalence holds and 1 when it does not, so this can
gate a launch script.
"""

import argparse
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# scripts/ is sys.path[0] when this file is run directly, which is how
# hft_variant_ab.py reaches the same helpers.
from hft_profile import hft_config, loss_without_sync

from snap2midi.models.hft.hft import HFT
from snap2midi.models.hft.hft_dataset import HFTDataset


def free_port() -> str:
    """A port the rendezvous can have. Bound and released, so it is racy in
    principle; in practice nothing else on this box is grabbing ports between
    the release and init_process_group."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def to_device_dtype(tensors, device: str, dtype: torch.dtype):
    """Move a batch, promoting only what is real-valued.

    label_velocity is a class index consumed by CrossEntropyLoss and has to
    stay integral; casting it to the compute dtype makes the loss raise.
    """
    return [t.to(device=device, dtype=dtype) if t.is_floating_point()
            else t.to(device=device) for t in tensors]


def build_model(checkpoint: str, fast_attention: bool, device: str,
                dtype: torch.dtype = torch.float32,
                compile_model: bool = False) -> HFT:
    """The checkpoint's own model, on `device`, with dropout off.

    Params come from the checkpoint rather than from a literal copied out of
    run_hft_paper.py. A literal can drift from the file it was copied from and
    would then quietly build a model the weights do not belong to; the
    checkpoint cannot disagree with itself. fast_attention is the one key
    overridden, because it is a property of the run being tested rather than of
    the weights -- use_sdpa holds no parameters, so the same state_dict loads
    either way.
    """
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = dict(stored["hyper_parameters"]["params"])
    params["fast_attention"] = fast_attention

    model = HFT.load_from_checkpoint(
        checkpoint, params=params, map_location=device)
    model.to(device=device, dtype=dtype)
    model.eval()          # dropout off -- see module docstring

    # Exactly train_hft.py's form: Module.compile() on the two submodules, not
    # torch.compile(module). The wrapper form renames every state_dict key to
    # _orig_mod.*, and compiling HFT itself is silently skipped because
    # HFT.forward reaches its submodules as self.hft_encoder(...). Testing a
    # different compile shape than the run uses would prove nothing about the
    # run.
    if compile_model:
        model.hft_encoder.compile()
        model.hft_decoder.compile()
    return model


def loss_from_outputs(module: HFT, outputs, labels) -> torch.Tensor:
    """loss_without_sync's arithmetic, over outputs already computed.

    Kept separate because DDP has to be entered through its __call__ -- that is
    where it prepares the backward reduction -- while loss_without_sync reaches
    past it with model.forward(). Calling the wrapper and then reducing here is
    the only way to have both the real DDP path and the same loss.

    `check_equivalent_to_replica` below asserts this against loss_without_sync,
    which hft_profile.check_replica in turn holds against HFT.training_step.
    """
    label_onset, label_offset, label_frames, label_velocity = labels
    (on1, off1, fr1, vel1, _attn, on2, off2, fr2, vel2) = outputs

    n_vel = vel1.shape[-1]
    label_onset = label_onset.reshape(-1)
    label_offset = label_offset.reshape(-1)
    label_frames = label_frames.reshape(-1)
    label_velocity = label_velocity.reshape(-1)

    loss_1st = (module.bce(on1.reshape(-1), label_onset)
                + module.bce(off1.reshape(-1), label_offset)
                + module.bce(fr1.reshape(-1), label_frames)
                + module.ce(vel1.reshape(-1, n_vel), label_velocity))
    loss_2nd = (module.bce(on2.reshape(-1), label_onset)
                + module.bce(off2.reshape(-1), label_offset)
                + module.bce(fr2.reshape(-1), label_frames)
                + module.ce(vel2.reshape(-1, n_vel), label_velocity))
    return (module.config["weight_A"] * loss_1st
            + module.config["weight_B"] * loss_2nd)


def loss_of(callable_model, module: HFT, batch) -> torch.Tensor:
    """Forward through `callable_model`, reduce with `module`'s loss objects."""
    spec, *labels = batch
    return loss_from_outputs(module, callable_model(spec), labels)


def check_equivalent_to_replica(module: HFT, batch) -> None:
    """Fail loudly if loss_from_outputs has drifted from loss_without_sync.

    Same guard hft_profile.check_replica applies one level up. Without it a
    silent divergence here would make every number below a measurement of the
    wrong quantity, and it would still look like a clean PASS.
    """
    with torch.no_grad():
        mine = loss_of(module, module, batch).item()
        theirs = loss_without_sync(module, batch).item()
    if abs(mine - theirs) > 1e-6 * max(1.0, abs(theirs)):
        raise SystemExit(
            f"[hft_ddp_equivalence] loss_from_outputs disagrees with "
            f"loss_without_sync ({mine} vs {theirs}). One of the two has "
            f"drifted from HFT.training_step; fix that before trusting this.")


def grad_vector(module: HFT) -> torch.Tensor:
    """Every gradient as one float64 vector, in a stable order.

    Sorted by name, not by parameters() order: DDP wraps the module and the
    single-process path does not, and nothing guarantees the two iterate
    identically. A parameter that never received a gradient contributes zeros
    rather than being dropped, so the vectors stay aligned and a parameter that
    is live on one path and dead on the other shows up as a difference instead
    of a silent length mismatch.
    """
    parts = []
    for _name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.detach().reshape(-1).double())
    return torch.cat(parts)


def single_process_grads(batch, checkpoint, fast_attention, device, ranks,
                         dtype=torch.float32, compile_model=False):
    """`full`, `halves`, and the different-batch reference, in one process."""
    module = build_model(checkpoint, fast_attention, device, dtype,
                         compile_model)
    check_equivalent_to_replica(module, batch)

    # full: the whole batch in one backward.
    module.zero_grad(set_to_none=True)
    full_loss = loss_of(module, module, batch)
    full_loss.backward()
    full = grad_vector(module)

    # halves: the same batch, sliced the way DDP would slice it, each slice
    # backwarded on its own and the results averaged by hand. No distributed
    # code anywhere -- this isolates the algebra from the machinery.
    per_rank = batch[0].shape[0] // ranks
    accum = torch.zeros_like(full)
    slice_losses = []
    for r in range(ranks):
        sl = slice(r * per_rank, (r + 1) * per_rank)
        module.zero_grad(set_to_none=True)
        loss = loss_of(module, module, [t[sl] for t in batch])
        loss.backward()
        accum += grad_vector(module)
        slice_losses.append(loss.item())
    halves = accum / ranks

    return module, full, halves, full_loss.item(), slice_losses


def ddp_worker(rank: int, world: int, cfg: dict) -> None:
    """One rank: its own slice, wrapped in DDP, one backward. Rank 0 saves."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = cfg["port"]
    # Each rank otherwise spawns a full thread pool and they fight for cores,
    # which slows the run without changing a single number.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // world))

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        device = cfg["device"]
        dtype = getattr(torch, cfg["dtype"])
        module = build_model(cfg["checkpoint"], cfg["fast_attention"],
                             device, dtype, cfg["compile"])

        # broadcast_buffers=False: hFT has no buffers to keep in step (LayerNorm
        # only, no BatchNorm running stats), and broadcasting none of them each
        # forward is pure overhead.
        # find_unused_parameters=True: the decoder returns an attention tensor
        # that the loss drops, and under fast_attention it is a placeholder. If
        # that leaves any parameter out of the graph, the default would raise
        # instead of reducing. It costs a traversal and changes no arithmetic.
        wrapped = DDP(
            module,
            device_ids=[torch.device(device).index] if device.startswith("cuda") else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        batch = to_device_dtype(
            torch.load(cfg["batch_path"], weights_only=False), device, dtype)
        per_rank = batch[0].shape[0] // world
        sl = slice(rank * per_rank, (rank + 1) * per_rank)

        wrapped.zero_grad(set_to_none=True)
        loss = loss_of(wrapped, module, [t[sl] for t in batch])
        loss.backward()          # DDP all-reduces (mean) inside this call

        if rank == 0:
            torch.save({"grad": grad_vector(module), "loss": loss.item()},
                       cfg["out_path"])
    finally:
        dist.destroy_process_group()


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Relative L2, and the worst per-element relative difference.

    The per-element figure is taken only over parameters carrying real signal:
    dividing by a gradient that is already ~0 manufactures a huge ratio that
    says nothing about the update the optimizer would apply. Same threshold
    scripts/hft_variant_ab.py uses, so the two scripts' numbers can be read
    against each other.
    """
    rel = ((a - b).norm() / b.norm()).item()
    big = b.abs() > b.abs().max() * 1e-3
    max_rel = ((a - b).abs()[big] / b.abs()[big]).max().item() if big.any() else 0.0
    return rel, max_rel


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="weights to test with; any hFT checkpoint will do, "
                         "the question is about reduction, not quality")
    ap.add_argument("--base-path", default="data/hft_maestro",
                    help="extracted store the val batch comes from")
    ap.add_argument("--batch", type=int, default=8,
                    help="total batch; must divide by --ranks (default 8, the "
                         "paper's effective batch)")
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, see docstring) or cuda / cuda:N")
    ap.add_argument("--fast-attention", action="store_true",
                    help="route attention through SDPA, as the trial does")
    ap.add_argument("--compile", action="store_true",
                    help="compile encoder and decoder, as the trial does. "
                         "Applied identically to the single-process and DDP "
                         "sides, so what is measured is DDP *under* compile -- "
                         "the combination the run uses and the one neither "
                         "this script nor hft_variant_ab.py covers alone. "
                         "Under DDP, Dynamo splits the compiled graph at "
                         "gradient-bucket boundaries to overlap the "
                         "all-reduce, which is a code path that exists only "
                         "when both are on. Expect a slow first call per "
                         "process while Inductor warms up.")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float64"],
                    help="compute dtype. float32 is what training uses; "
                         "float64 is the tiebreaker -- if a float32 difference "
                         "is only accumulation noise it collapses by orders of "
                         "magnitude here, and if it is a real reduction bug it "
                         "does not move (default float32)")
    ap.add_argument("--floor-multiple", type=float, default=10.0,
                    help="how many times the measured single-process splitting "
                         "noise ('halves vs full') the ddp gradient may differ "
                         "by and still count as equal (default 10)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    if args.batch % args.ranks:
        sys.exit(f"[hft_ddp_equivalence] --batch {args.batch} does not divide "
                 f"by --ranks {args.ranks}; an uneven shard is a different "
                 f"experiment (and DistributedSampler pads rather than "
                 f"allowing one)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("[hft_ddp_equivalence] --device cuda but no CUDA available")

    # Two batches: the one under test, and a different one of the same shape to
    # calibrate what "different" is worth. Validation split, so unaugmented and
    # identical on every re-run.
    class DataArgs:
        base_path = args.base_path
        loader_batch = args.batch
        workers = 0
        augment = False
        augment_asset_root = None

    dataset = HFTDataset(hft_config(DataArgs()), split="val")
    loader = iter(DataLoader(dataset, batch_size=args.batch, shuffle=False))
    batch = to_device_dtype(next(loader), args.device, dtype)
    other = to_device_dtype(next(loader), args.device, dtype)

    print(f"checkpoint     : {Path(args.checkpoint).name}")
    print(f"batch          : {args.batch} unaugmented val excerpts "
          f"from {args.base_path}")
    print(f"split          : {args.ranks} ranks x {args.batch // args.ranks}")
    print(f"device         : {args.device} (gloo), {args.dtype}")
    print(f"fast_attention : {args.fast_attention}")
    print(f"compile        : {args.compile}\n")

    if args.compile and args.dtype == "float64":
        print("note: Inductor's float64 coverage is thinner than its float32 "
              "coverage, so a\n      failure here may be the backend rather "
              "than the reduction. Confirm any\n      float64 result against "
              "the float32 run before believing it.\n")

    module, full, halves, full_loss, slice_losses = single_process_grads(
        batch, args.checkpoint, args.fast_attention, args.device, args.ranks,
        dtype, args.compile)

    # Reference scale: a real difference, for the numbers above to be read
    # against. Same weights, different excerpts.
    module.zero_grad(set_to_none=True)
    loss_of(module, module, other).backward()
    reference = grad_vector(module)
    del module

    with tempfile.TemporaryDirectory() as tmp:
        batch_path = Path(tmp) / "batch.pt"
        out_path = Path(tmp) / "rank0.pt"
        torch.save([t.cpu() for t in batch], batch_path)

        mp.spawn(
            ddp_worker,
            args=(args.ranks, {
                "port": free_port(),
                "device": args.device,
                "checkpoint": args.checkpoint,
                "fast_attention": args.fast_attention,
                "batch_path": str(batch_path),
                "out_path": str(out_path),
                "dtype": args.dtype,
                "compile": args.compile,
            }),
            nprocs=args.ranks,
            join=True,
        )

        if not out_path.exists():
            sys.exit("[hft_ddp_equivalence] rank 0 wrote no gradient; the "
                     "spawn above failed and its traceback is what matters")
        saved = torch.load(out_path, weights_only=False)
    ddp_grad, ddp_loss = saved["grad"], saved["loss"]

    print(f"loss, full batch      : {full_loss:.8f}")
    print(f"loss, per-slice       : "
          f"{', '.join(f'{v:.8f}' for v in slice_losses)}")
    print(f"loss, ddp rank 0      : {ddp_loss:.8f}")
    print(f"  (slice losses differ from the full-batch loss by design: each is "
          f"a mean\n   over its own excerpts. It is the *gradient* that must "
          f"agree, not these.)\n")

    rows = [
        ("halves vs full", *compare(halves, full)),
        ("ddp vs full", *compare(ddp_grad, full)),
        ("ddp vs halves", *compare(ddp_grad, halves)),
        ("different batch vs full", *compare(reference, full)),
    ]
    print(f"{'comparison':<26} {'grad rel L2':>12} {'max rel elem':>14}")
    print("-" * 54)
    for name, rel, max_rel in rows[:3]:
        print(f"{name:<26} {rel:>12.2e} {max_rel:>14.2e}")
    print("-" * 54)
    print(f"{rows[3][0]:<26} {rows[3][1]:>12.2e} {rows[3][2]:>14.2e}"
          f"   <- reference scale")

    floor_rel = rows[0][1]      # halves vs full: splitting noise, no DDP
    ddp_rel = rows[1][1]
    ref_rel = rows[3][1]

    # Judged against the floor this run measured, not a constant. Splitting a
    # reduction costs accumulation error that grows with the number of elements
    # summed and shrinks with the dtype, so any fixed tolerance is either too
    # tight in float32 or meaningless in float64. 'halves vs full' is that cost
    # with the distributed code removed, which makes it the only honest
    # yardstick for 'ddp vs full' -- DDP cannot be expected to beat the
    # arithmetic.
    #
    # The reference check stays as the second condition: being near the floor
    # proves little if the floor itself is the size of a real update.
    near_floor = ddp_rel <= max(floor_rel * args.floor_multiple, 1e-12)
    below_ref = ref_rel > 0 and ddp_rel < ref_rel / 1000

    print()
    if near_floor and below_ref:
        print(f"PASS  ddp vs full is {ddp_rel:.2e}, within "
              f"{ddp_rel / max(floor_rel, 1e-300):.1f}x of the "
              f"{floor_rel:.2e} that splitting\n      the reduction costs in "
              f"{args.dtype} arithmetic alone, and "
              f"{ref_rel / max(ddp_rel, 1e-300):.1e}x smaller than a\n"
              f"      different batch. {args.ranks} x "
              f"{args.batch // args.ranks} is the same update as "
              f"1 x {args.batch}.")
        if args.dtype == "float32":
            print("\n      Confirm with --dtype float64: if this is "
                  "accumulation noise both\n      numbers collapse together, "
                  "and if it is a reduction bug they do not.")
        sys.exit(0)

    print(f"FAIL  ddp vs full is {ddp_rel:.2e} relative L2, against a "
          f"{floor_rel:.2e} arithmetic floor\n      and a {ref_rel:.2e} "
          f"reference.")
    if not below_ref:
        print("      It is within 1000x of a genuinely different batch, so "
              "this is not a\n      rounding artefact.")
    if ddp_rel > floor_rel * args.floor_multiple:
        print("      'halves vs full' is far below it, so the mean-of-means "
              "algebra holds and\n      DDP is reducing something other than "
              "the plain average. Check for a\n      parameter excluded from "
              "the reduction, or an uneven shard.")
    print("      Re-run with --dtype float64 before acting on this: it "
          "separates an\n      accumulation artefact from a real reduction "
          "bug.")
    sys.exit(1)


if __name__ == "__main__":
    main()
