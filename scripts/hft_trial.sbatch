#!/bin/bash
#
# The timing trial: run_hft_trial.py on 2 H100s, from scratch, 141,000 steps.
#
#   sbatch scripts/hft_trial.sbatch                      # ~4 h, 2 val points
#   HFT_TRIAL_STEPS=71000 sbatch scripts/hft_trial.sbatch  # ~2 h, 1 val point
#
# Expected ~4 h: ~3.4 h of training at the projected ~11.7 it/s, plus the two
# quarter-epoch validations and a few minutes of torch.compile.
#
# --time is 8 h, not 4. Slurm bills *elapsed* time, not requested time, so the
# headroom is free: a job that finishes in 4 h costs 4 h whatever this says.
# What the number does affect is the scheduler -- a shorter request backfills
# into gaps more easily and starts sooner -- and the downside of guessing low
# is a kill with no result. 8 h is 2x the estimate, which covers the projection
# being wrong by half while still being short enough to backfill. To spend less,
# lower HFT_TRIAL_STEPS; lowering this line does not.
#
# One job, not a chain. Nothing here resumes, and it must not: the trial exists
# to produce a from-zero curve to lay over the 12 h control (job 3100103).
#
#SBATCH --job-name=hft_trial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/home/resh000979/Snap2MIDI/logs/hft_trial_%j.out
#SBATCH --error=/home/resh000979/Snap2MIDI/logs/hft_trial_%j.err
#SBATCH --partition=res_gpu
#SBATCH --account=upf105_b
#
# ntasks stays 1. torchrun makes the two processes, not Slurm -- that is the
# whole difference from hft_paper.sbatch. Asking Slurm for 2 tasks *as well*
# would start two torchruns, each spawning two ranks, and the four would
# deadlock on a rendezvous none of them agrees about.
#
# Lightning reads its world size from torchrun's environment because
# train_hft.py no longer pins SLURMEnvironment; Lightning's own detection puts
# TorchElastic ahead of Slurm precisely so this works. If a future edit
# reintroduces that plugin, this script hangs at startup with no error.
#
# Slurm opens --output/--error before the script runs, so this must exist first:
#   mkdir -p /home/resh000979/Snap2MIDI/logs

set -e
set -x

module purge
module load conda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate snap2midi
export WANDB_MODE=offline

# Both GPUs, checked before anything expensive. pl.Trainer(devices=2) with one
# visible GPU fails late and confusingly; this fails immediately and says why.
srun --ntasks=1 python -c "
import torch
n = torch.cuda.device_count()
assert n >= 2, f'need 2 GPUs, torch sees {n}'
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(n)])
"

cd /home/resh000979/Snap2MIDI

# 32 CPUs shared by 2 ranks x 16 dataloader workers. At 4 threads each that is
# 128 threads on 32 cores, and the contention shows up as jitter in the step
# time -- which is the one number this job exists to measure. The loader has
# ~15x more headroom than it needs (measured: 690 excerpts/s at 8 workers
# against ~45/s consumed), so starving it slightly costs nothing.
export OMP_NUM_THREADS=2

# Keep temp on the same filesystem as save_dir. Lightning writes checkpoints
# atomically -- temp file, then rename -- and Slurm puts TMPDIR on /scratch
# while save_dir is on /data, where rename() across filesystems is EXDEV. This
# is what broke the requeue handler; see scripts/hft_paper.sbatch.
export TMPDIR=/data/upf105/resh000979/tmp
mkdir -p "$TMPDIR"

# torch.compile caches its generated kernels here. On /data rather than the
# default under TMPDIR so the cache survives the job, and a rerun (or the real
# run afterwards) skips the few minutes of compilation.
export TORCHINDUCTOR_CACHE_DIR=/data/upf105/resh000979/torchinductor_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# What the GPUs actually did, sampled every 30 s. Read it with
#   scripts/gpu_watch.sh summary logs/gpu_trial_<jobid>.csv
# and note what the utilisation column does and does not mean -- the header of
# that script explains why a 100% reading here is not evidence of anything.
scripts/gpu_watch.sh sample 30 logs/gpu_trial_${SLURM_JOB_ID}.csv &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

# --standalone: single node, so torchrun picks its own rendezvous port instead
# of needing MASTER_ADDR/MASTER_PORT. No srun -- torchrun is the launcher here,
# and putting srun in front would nest two launchers.
torchrun --standalone --nproc_per_node=2 /home/resh000979/Snap2MIDI/run_hft_trial.py
