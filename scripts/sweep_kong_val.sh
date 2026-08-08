#!/usr/bin/env bash
# K2: rank every surviving kong_paper_aug checkpoint on decoded note F1 over a
# fixed 40-piece subset of data/kong_maestro/val, rather than on
# valid_total_loss -- the five survivors span 0.09% in that loss, which is
# noise, and it is ~102% frame-driven while the reported metric is note F1.
#
# The subset is the first 40 sorted files, identical for every candidate, so
# the comparison is paired. ~8 min per checkpoint at 11.6 s/piece.
#
# Usage: bash scripts/sweep_kong_val.sh [limit]
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LIMIT=${1:-40}
STATUS=logs/kong_sweep.status

for ckpt in save_dir/kong_paper_aug/kong-step=*.ckpt; do
    echo "$(date -Is) evaluating $(basename "$ckpt")" >> "$STATUS"
    $PY scripts/eval_kong_paper.py --dataset maestro-val --limit "$LIMIT" \
        --checkpoint "$ckpt" >> logs/kong_sweep.log 2>&1
    echo "$(date -Is) rc=$?" >> "$STATUS"
done
echo "$(date -Is) sweep complete" >> "$STATUS"
