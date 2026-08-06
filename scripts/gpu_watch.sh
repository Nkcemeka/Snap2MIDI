#!/bin/bash
#
# Record what the GPU was doing during a job, and reduce it afterwards.
#
#   scripts/gpu_watch.sh sample [interval_s] [outfile]   # background, inside the sbatch
#   scripts/gpu_watch.sh summary <csvfile>               # read it back
#
# In an sbatch, before the training line:
#
#   scripts/gpu_watch.sh sample 30 logs/gpu_${SLURM_JOB_ID}.csv &
#   WATCH=$!
#   srun python run_hft_paper.py
#   kill $WATCH
#
# At 30 s that is ~20k rows over a 7-day job, about 2 MB. The sampler is a
# plain nvidia-smi loop, so it costs nothing measurable and cannot affect the
# result it is measuring.
#
# To look at a job already running, without touching its sbatch:
#
#   srun --jobid=<JOBID> --overlap --pty nvidia-smi
#   srun --jobid=<JOBID> --overlap --pty scripts/gpu_watch.sh sample 5 /dev/stdout
#
# READ THE UTILISATION NUMBER CORRECTLY. `utilization.gpu` is the percentage of
# sampled intervals in which *at least one kernel was resident* -- it is an
# occupancy-of-time figure, not an efficiency one. A model that spends its life
# in layernorms, softmaxes and dropout reads 95-100% here while using a few per
# cent of the card's arithmetic. hFT is exactly that model: measured with
# scripts/hft_profile.py it sustains ~24 TFLOP/s against an H100 PCIe's 378
# TF32 peak. So a high number here does NOT mean there is nothing to win --
# it means nvidia-smi cannot answer the question. Use it for:
#
#   * memory.used  -- how much headroom there really is
#   * power.draw   -- a genuinely memory-bound kernel draws far under the cap
#   * gaps         -- utilisation dropping to 0 periodically is the dataloader
#                     starving the GPU, or checkpoint writes stalling the step
#
# For "how much of the GPU is being used" in the sense that matters, run
# scripts/hft_profile.py and compare achieved TFLOP/s to the card's peak.

set -euo pipefail

FIELDS=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,temperature.gpu

case "${1:-}" in
  sample)
    interval="${2:-30}"
    out="${3:-gpu_watch.csv}"
    [ "$out" = /dev/stdout ] || mkdir -p "$(dirname "$out")"
    # stdbuf so the file is readable while the job is still running; nvidia-smi
    # block-buffers into a pipe or file otherwise and you see nothing for hours.
    exec stdbuf -oL nvidia-smi --query-gpu="$FIELDS" --format=csv,nounits -l "$interval" > "$out"
    ;;

  summary)
    csv="${2:?usage: gpu_watch.sh summary <csvfile>}"

    # Percentiles, not just a mean: the mean hides a job that ran at 100% for
    # six days and 0% for one, which is the shape a stalled chain makes.
    # Sorted with sort(1) rather than awk's asort, which is a gawk extension
    # and absent from the mawk that is default on plenty of compute nodes.
    stats() {
      awk -F', *' -v c="$1" 'NR > 1 && NF >= 6 { print $c + 0 }' "$csv" | sort -n | awk -v label="$2" '
        { a[NR] = $1 }
        END {
          if (NR == 0) { printf "%-16s no samples\n", label; exit }
          printf "%-16s p10 %-8d median %-8d p90 %-8d max %d\n", label,
                 a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], a[NR]
        }'
    }

    n=$(awk -F', *' 'NR > 1 && NF >= 6' "$csv" | wc -l)
    total=$(awk -F', *' 'NR == 2 { print $6; exit }' "$csv")
    echo "samples        : $n   (memory total $total MiB)"
    stats 3 "gpu util %"
    stats 5 "memory MiB"
    stats 7 "power W"
    awk -F', *' 'NR > 1 && NF >= 6 { n++; if ($3 + 0 < 5) idle++ }
                 END { if (n) printf "%-16s %d (%.1f%%)  <- dataloader stalls, checkpoint writes, or a dead job\n",
                                     "samples <5% util", idle, 100.0*idle/n }' "$csv"
    ;;

  *)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
