#!/bin/bash
#
# Run hFT until the first validation completes, to measure the real per-epoch
# cost on this hardware. With val_check_interval=0.25 that happens at training
# step 70,164 of 280,659 -- between ~3.3 h (if the H100 is 3x an A100) and
# ~10.9 h (if it is no faster). 12 h covers the pessimistic end.
#
# What to read off it:
#   training it/s   from the progress bar
#   validation it/s from the progress bar during the validation pass
#   hours/epoch = (280660 / train_it_s + 4 * 34273 / val_it_s) / 3600
#
# The first validation is done the moment a file appears in save_dir named
#   hft-epoch=00-valid_total_loss=*.ckpt
# At that point scancel the job -- everything after it is just more of the same,
# and nothing is wasted: the rolling checkpoint means the real run resumes from
# wherever this one stopped.
#
#SBATCH --job-name=train_hft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
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
# FOR THE FULL 20-EPOCH RUN, uncomment these three and set EPOCHS=20.
#
# One epoch is longer than any walltime here, so the full run gets killed
# mid-epoch and has to survive it: --signal fires SIGUSR1 five minutes before
# the kill, which Lightning catches to save and resubmit; --requeue permits the
# resubmission; --open-mode stops it truncating the log.
#
# They are OFF for the measurement run above on purpose. With them on, the job
# does not stop when the walltime expires -- it requeues and keeps training, so
# a run meant to end after the first validation would quietly become the full
# 20 epochs.
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

# srun rather than a bare `python`, and only for the sake of --signal above:
# sbatch signals job steps but explicitly NOT the batch shell, and a command run
# directly in the script is part of the batch shell. Without srun the SIGUSR1
# would never reach Python. Harmless while --signal is commented out; leaving it
# here means enabling the full run is only about uncommenting.
srun python /home/resh000979/Snap2MIDI/run_hft_paper.py
