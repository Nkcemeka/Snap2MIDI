#!/bin/bash
#
# The full 20-epoch hFT run on MAESTRO with the speedups: run_hft_fast.py on
# 2 H100s, fast attention and torch.compile, resuming across a chain.
#
# MEASURED 7.5 days for 20 epochs. res_gpu caps a job at 7 days, so this does
# NOT fit in one job. It misses by half a day -- the worst margin there is,
# because it looks like it fits right up until the walltime kill. Submit it as a
# chain, always, even though the whole point of the speedups was to get near one
# job:
#
#   mkdir -p /home/resh000979/Snap2MIDI/logs
#   J1=$(sbatch --parsable scripts/hft_fast.sbatch)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 scripts/hft_fast.sbatch)
#
# Two, and by a wide margin: two jobs are 14 days of walltime against 7.5 days
# of work, so the measured rate would have to be wrong by nearly 2x before a
# third was needed to finish training. A third insures against something else
# entirely -- a job dying early and wasting its slot, where a node failure at
# hour 3 costs a whole link of the chain rather than three hours.
#
# That insurance is deliberately not bought up front, because it can be bought
# later. A job can be appended while the chain is still alive:
#
#   J3=$(sbatch --parsable --dependency=afterany:$J2 scripts/hft_fast.sbatch)
#
# Only a chain that has already drained cannot be extended. So submit two, watch
# the first handoff, and append a third then if anything has been lost. A spare
# that turns out to be unnecessary is harmless either way -- it loads last.ckpt,
# sees max_epochs=20 already reached, and exits in a couple of minutes.
#
# afterany, not afterok: a walltime kill exits TIMEOUT/FAILED, so afterok would
# never fire and the chain would stall silently with no error anywhere.
#
# BEFORE SUBMITTING J1, check the run has somewhere clean to start:
#
#   ls /data/upf105/resh000979/save_dir/hft_fast/last.ckpt
#
# It must NOT exist. run_hft_fast.py sets resume_path="last", so a stray
# last.ckpt is resumed from without complaint -- and if it came from the trial
# or an aborted attempt, the "run" finishes early having trained the wrong
# thing. This is the same trap run_hft_paper.py's docstring warns about.
#
#SBATCH --job-name=hft_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/resh000979/Snap2MIDI/logs/hft_fast_%j.out
#SBATCH --error=/home/resh000979/Snap2MIDI/logs/hft_fast_%j.err
#SBATCH --partition=res_gpu
#SBATCH --account=upf105_b
#
# ntasks stays 1. torchrun makes the two processes, not Slurm. Asking Slurm for
# 2 tasks as well would start two torchruns, each spawning two ranks, and the
# four would deadlock on a rendezvous none of them agrees about.
#
# Lightning reads its world size from torchrun's environment because
# train_hft.py no longer pins SLURMEnvironment; Lightning's own detection puts
# TorchElastic ahead of Slurm precisely so this works. If a future edit
# reintroduces that plugin, this script hangs at startup with no error.
#
# Slurm opens --output/--error BEFORE the script runs, so the logs directory
# has to exist already or the job dies at submission with no explanation.
#
# ---------------------------------------------------------------------------
# NO --signal/--requeue. Tested 2026-08-05 (jobs 3101512, 3101523): Lightning's
# SIGUSR1 handler writes its checkpoint atomically (temp file, then rename) and
# Slurm puts TMPDIR on /scratch while save_dir is on /data, so the rename is
# EXDEV. Exporting TMPDIR onto /data does not rescue it either -- the dataloader
# workers receive SIGUSR1 too and hit `CUDA error: initialization error`,
# because a worker has no CUDA context. Worse than not requeueing, a failed
# handler leaves a truncated hpc_ckpt_*.ckpt in save_dir, which is exactly where
# Lightning looks on the next start. The dependency chain above does the same
# job using the resume path that is actually tested.
# ---------------------------------------------------------------------------

set -e
set -x

module purge
module load conda

# Hardcoded, and it has to be. `conda info --base` does not resolve in this
# batch environment -- `module load conda` does not leave the executable
# somewhere a non-interactive shell finds it, so the substitution comes back
# empty and this sources /etc/profile.d/conda.sh, which is not conda.
source /prod/precompiled/conda/etc/profile.d/conda.sh
conda activate snap2midi
export WANDB_MODE=offline

# Both GPUs, checked before anything expensive. pl.Trainer(devices=2) with one
# visible GPU fails late and confusingly; this fails immediately and says why.
#
# No srun, unlike hft_paper.sbatch. Wrapping this in srun killed job 3104746
# before it started:
#
#   srun: fatal: cpus_per_task set by two different environment variables
#         SLURM_CPUS_PER_TASK=64 != SLURM_TRES_PER_TASK=cpu:32
#
# The allocation has 64 CPUs (32 cores x 2 threads) while the request is
# recorded as cpu:32, and current Slurm refuses to launch a step when those
# disagree. srun was never right here anyway: the training below is torchrun,
# launched from this batch shell, so this shell is the environment to validate.
python -c "
import torch
n = torch.cuda.device_count()
assert n >= 2, f'need 2 GPUs, torch sees {n}'
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(n)])
"

# The tensorboard logger writes to ./logs relative to the working directory, so
# this cd decides where the run's metrics land. Everything else in
# run_hft_fast.py uses absolute paths.
cd /home/resh000979/Snap2MIDI

# 32 CPUs shared by 2 ranks x 16 dataloader workers. At 4 threads each that is
# 128 threads on 32 cores; the contention shows up as jitter in the step time.
# The loader has ~15x more headroom than it needs (measured: 690 excerpts/s at
# 8 workers against ~45/s consumed), so starving it slightly costs nothing.
export OMP_NUM_THREADS=2

# Keep temp on the same filesystem as save_dir. Lightning writes checkpoints
# atomically -- temp file, then rename -- and Slurm puts TMPDIR on /scratch
# while save_dir is on /data, where rename() across filesystems is EXDEV. This
# is what broke the requeue handler above. With a 7.5-day chain there are ~80
# selection checkpoints plus a rolling one every 5000 steps going through this
# path, so it is not a latent concern here, it is the hot path.
export TMPDIR=/data/upf105/resh000979/tmp
mkdir -p "$TMPDIR"

# torch.compile caches its generated kernels here. On /data rather than under
# TMPDIR so the cache survives the job -- which matters much more for a chain
# than it did for the trial: without it every job in the chain pays the
# compilation again on resume.
export TORCHINDUCTOR_CACHE_DIR=/data/upf105/resh000979/torchinductor_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# What the GPUs actually did, sampled every 30 s. Read it with
#   scripts/gpu_watch.sh summary logs/gpu_fast_<jobid>.csv
# and note what the utilisation column does and does not mean -- the header of
# that script explains why a 100% reading is not evidence of anything.
scripts/gpu_watch.sh sample 30 logs/gpu_fast_${SLURM_JOB_ID}.csv &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

# --standalone: single node, so torchrun picks its own rendezvous port instead
# of needing MASTER_ADDR/MASTER_PORT. No srun -- torchrun is the launcher here,
# and putting srun in front would nest two launchers.
torchrun --standalone --nproc_per_node=2 /home/resh000979/Snap2MIDI/run_hft_fast.py
