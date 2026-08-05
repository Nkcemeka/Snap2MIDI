#!/bin/bash
#
# The full 20-epoch hFT run on MAESTRO. Measured on this hardware, not guessed:
# the 12 h trial (job 3100103) gave 3.9 it/s training and 14.7 it/s validation on
# an H100 PCIe, so
#
#   hours/epoch = (280659 / 3.9 + 4 * 34273 / 14.7) / 3600 = 22.6 h
#   20 epochs   = 452 h = 19 days
#
# res_gpu caps a job at 7 days, so that is 3 jobs. They are NOT requeued (see the
# block below) -- they are submitted up front as a dependency chain, each picking
# up from last.ckpt:
#
#   J1=$(sbatch --parsable scripts/hft_paper.sbatch)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 scripts/hft_paper.sbatch)
#   J3=$(sbatch --parsable --dependency=afterany:$J2 scripts/hft_paper.sbatch)
#
# afterany, not afterok: a walltime kill exits TIMEOUT/FAILED, so afterok would
# never fire and the chain would stall silently with no error anywhere.
#
#SBATCH --job-name=train_hft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/resh000979/Snap2MIDI/logs/hft_%j.out
#SBATCH --error=/home/resh000979/Snap2MIDI/logs/hft_%j.err
#SBATCH --partition=res_gpu
#SBATCH --account=upf105_b
#
# Slurm opens --output/--error BEFORE the script runs, so the logs directory has
# to exist already or the job dies at submission with no explanation:
#   mkdir -p /home/resh000979/Snap2MIDI/logs
#
# ---------------------------------------------------------------------------
# LEAVE THESE THREE COMMENTED OUT. Tested 2026-08-05 (jobs 3101512, 3101523) and
# they do not work here.
#
# The idea was that --signal fires SIGUSR1 five minutes before the walltime kill,
# Lightning catches it, saves at the exact current step and resubmits itself, and
# --requeue permits that. What actually happens is that the handler dies:
#
#   OSError: [Errno 18] Invalid cross-device link:
#     '/scratch/upf105/.../tmpXXXX' -> '/data/upf105/.../hpc_ckpt_1.ckpt'
#
# Lightning writes that checkpoint atomically -- temp file, then rename -- and
# Slurm puts TMPDIR on /scratch while save_dir is on /data. rename() across
# filesystems is EXDEV. Exporting TMPDIR onto /data (done below) does not rescue
# it either: the dataloader workers receive SIGUSR1 too, run the same handler,
# and hit `CUDA error: initialization error` because a worker has no CUDA
# context. The job then dies on `DataLoader worker exited unexpectedly`.
#
# Worse than not requeueing, a failed handler leaves a truncated hpc_ckpt_*.ckpt
# in save_dir, which is exactly where Lightning looks on the next start.
#
# So: no requeue. The dependency chain in the header does the same job, using the
# resume path that is actually tested, and the only cost is the work done since
# the last rolling checkpoint -- bounded by ckpt_every_n_steps=5000 in
# run_hft_paper.py to ~21 min per handoff, ~42 min across the whole run.
#
##SBATCH --signal=USR1@300
##SBATCH --requeue
##SBATCH --open-mode=append
# ---------------------------------------------------------------------------

set -e
set -x

module purge
module load conda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate snap2midi
export WANDB_MODE=offline

# Fail now if torch cannot see a GPU. pl.Trainer uses accelerator="auto", which
# does NOT error without one -- it silently trains on CPU, ~50x slower, with a
# normal-looking progress bar. That would burn the whole allocation before
# anyone noticed. Run under srun so the check sees exactly the environment the
# training will; a bare python might not inherit CUDA_VISIBLE_DEVICES in the
# batch shell on every Slurm configuration, and would then fail a healthy job.
srun python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE TO TORCH'; print('GPU:', torch.cuda.get_device_name(0))"

# The tensorboard logger writes to ./logs relative to the working directory, so
# this cd decides where the run's metrics land. Everything else in
# run_hft_paper.py uses absolute paths.
cd /home/resh000979/Snap2MIDI

# Deliberately small, and not 32-minus-workers. This is inherited by all 16
# dataloader workers and bounds the OpenMP pools numpy/scipy use inside them
# during augmentation. torch already caps workers at one thread each, so this is
# belt-and-braces; the main process is GPU-bound and needs no more than this.
export OMP_NUM_THREADS=4

# Slurm sets TMPDIR to a per-job directory on /scratch, which is a different
# filesystem from the /data holding save_dir. Anything that saves atomically --
# temp file, then rename -- then fails with EXDEV, because rename() cannot cross
# filesystems. That is what killed the requeue handler above. Nothing in the
# current path depends on it, but the trap is latent for any future atomic write
# into save_dir, so keep the temp directory on the same filesystem as the target.
export TMPDIR=/data/upf105/resh000979/tmp
mkdir -p "$TMPDIR"

# srun rather than a bare `python`. It was originally here so SIGUSR1 could reach
# Python -- sbatch signals job steps but explicitly NOT the batch shell, and a
# command run directly in the script is part of the batch shell. That path is
# abandoned (see above), but srun stays: it is what the GPU check above runs
# under, and keeping both consistent means the training sees exactly the
# environment the check validated.
srun python /home/resh000979/Snap2MIDI/run_hft_paper.py
