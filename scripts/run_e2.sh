#!/usr/bin/env bash
# E2: tune all 5 models on all 3 tiers (sequential to stay inside RAM).
set -u
cd "$(dirname "$0")/.."
for t in t1 t2 t3; do
  .venv/bin/python -m src.modelling.train --tier "$t" --model all 2>&1 | grep -v -i warning | tee -a "results/logs/train_$t.log"
done
