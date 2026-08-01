#!/usr/bin/env bash
# Evaluate the paper-configuration Kong run (save_dir/kong_paper_aug) on both
# test sets, using the best valid_total_loss checkpoint recorded by the
# ModelCheckpoint callback.
#
# MAESTRO test is the primary number: the run reproduces Kong et al. §IV-C at
# full budget, so its note F1 is comparable to the published 96.72. MAPS test is
# the same set the 55k-step augmentation A/B was scored on, so it keeps this run
# on the same axis as those arms.
#
# Usage: bash scripts/eval_kong_paper.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=save_dir/kong_paper_aug
STATUS=logs/kong_paper_eval.status

ckpt=$($PY - <<'PY'
import torch
d = torch.load("save_dir/kong_paper_aug/last.ckpt", map_location="cpu",
               weights_only=False)
cb = [v for k, v in d["callbacks"].items() if "ModelCheckpoint" in k][0]
print(cb["best_model_path"])
PY
)
echo "$(date -Is) best checkpoint: $(basename "$ckpt")" >> "$STATUS"

# MAESTRO first: it is the number the run exists to produce, so if the machine
# is claimed for something else midway the important half is already done.
for pair in "maestro:./data/kong_maestro/test" "maps:./data/kong_maps/test"; do
    name=${pair%%:*}
    path=${pair#*:}
    echo "$(date -Is) evaluating on $name ($path)" >> "$STATUS"
    $PY - "$path" "$ckpt" > "logs/eval_paper_aug_$name.log" 2>&1 <<'PY'
import sys, pprint
import snap2midi as s2m
frame, note = s2m.evaluator.Evaluator().evaluate_kong(sys.argv[1], sys.argv[2])
print("\n=== frame ==="); pprint.pprint(frame)
print("\n=== note ==="); pprint.pprint(note)
PY
    echo "$(date -Is) $name eval exited rc=$?" >> "$STATUS"
done
