#!/usr/bin/env bash
# MAPS evaluation of both arms, using each run's best valid_total_loss
# checkpoint as recorded by the ModelCheckpoint callback in last.ckpt.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for run in baseline augmented; do
    ckpt=$($PY - "$run" <<'PY'
import sys, torch
d = torch.load(f"save_dir/kong_{sys.argv[1]}/last.ckpt", map_location="cpu",
               weights_only=False)
cb = [v for k, v in d["callbacks"].items() if "ModelCheckpoint" in k][0]
print(cb["best_model_path"])
PY
)
    echo "$(date -Is) evaluating $run: $(basename "$ckpt")" >> logs/kong_eval.status
    $PY run_kong_experiment.py evaluate "$ckpt" > "logs/eval_$run.log" 2>&1
    echo "$(date -Is) $run eval exited rc=$?" >> logs/kong_eval.status
done
