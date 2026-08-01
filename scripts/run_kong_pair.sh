#!/usr/bin/env bash
# Baseline then augmented, sequentially on one GPU. Augmented only starts if
# baseline exits clean, so a crash does not burn the second slot too.
#
#   bash scripts/run_kong_pair.sh          # detached, survives closing the shell
#   tail -f logs/kong_baseline.log         # watch
#   kill $(cat logs/kong_pair.pid)         # stop the pair
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

PY=.venv/bin/python
echo "$(date -Is) starting baseline" >> logs/kong_pair.status
$PY run_kong_experiment.py baseline > logs/kong_baseline.log 2>&1
baseline_rc=$?
echo "$(date -Is) baseline exited rc=$baseline_rc" >> logs/kong_pair.status

if [ $baseline_rc -ne 0 ]; then
    echo "$(date -Is) augmented skipped, baseline failed" >> logs/kong_pair.status
    exit $baseline_rc
fi

echo "$(date -Is) starting augmented" >> logs/kong_pair.status
$PY run_kong_experiment.py augmented > logs/kong_augmented.log 2>&1
echo "$(date -Is) augmented exited rc=$?" >> logs/kong_pair.status
