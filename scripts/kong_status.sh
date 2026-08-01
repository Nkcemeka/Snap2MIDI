#!/usr/bin/env bash
# Where the kong experiment pair is right now.  Usage: bash scripts/kong_status.sh
set -u
cd "$(dirname "$0")/.."

echo "=== runner ==="
[ -f logs/kong_pair.status ] && cat logs/kong_pair.status || echo "never started"
pid=$(pgrep -f "bin/python run_kong_experiment" | head -1)
if [ -n "$pid" ]; then
    ps -o pid=,etime=,cmd= -p "$pid" | sed 's/^/running: /'
else
    echo "no training process alive"
fi

# No progress bar reaches the log without a TTY, and wandb 0.27 does not write
# wandb-summary.json until the run ends, so live step and losses come from the
# server. The run id is in the log line wandb prints on startup.
echo
echo "=== metrics ==="
# -h: with two files grep prefixes the filename, which would corrupt the id.
run_id=$(grep -haoE "runs/[a-z0-9]+" logs/kong_augmented.log logs/kong_baseline.log \
    2>/dev/null | tail -1 | cut -d/ -f2)
if [ -z "$run_id" ]; then
    echo "  no wandb run yet"
else
    .venv/bin/python - "$run_id" <<'PY' 2>/dev/null
import sys, wandb
r = wandb.Api().run(f"simone-chieppa99-universitat-pompeu-fabra/lightning_logs/{sys.argv[1]}")
s = r.summary
step = s.get("trainer/global_step", 0)
print(f"  {r.name} [{r.state}] step {step}/55000 ({step/550:.1f}%)")
for k in ("train_total_loss", "valid_total_loss", "train_onset_loss",
          "train_frame_loss", "train_velocity_loss"):
    if k in s:
        print(f"    {k:20s} {s[k]:.4f}")
print(f"  ~{(55000 - step) * 0.465 / 3600:.1f} h left in this run at 465 ms/step")
print(f"  {r.url}")
PY
fi

for run in baseline augmented; do
    log="logs/kong_$run.log"
    dir="save_dir/kong_$run"
    [ -f "$log" ] || continue
    echo
    echo "=== $run ==="
    grep -iE "error|traceback|out of memory|Killed" "$log" | tail -2
    if [ -d "$dir" ]; then
        echo "checkpoints (step / val loss):"
        ls -1 "$dir" | grep -oE "step=[0-9]+-loss=valid_total_loss=[0-9.]+" \
            | sed 's/^/  /' | sort -t= -k2 -n
        [ -f "$dir/last.ckpt" ] && \
            echo "  last.ckpt written $(date -r "$dir/last.ckpt" '+%H:%M:%S')"
    else
        echo "  no checkpoints yet (first lands at step 5000)"
    fi
done

echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
