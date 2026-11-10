#!/usr/bin/env bash
# Matched-seed runs: one process per seed, on the kitchen rebuilt from the seed (~6 GB per process).
# Extra arguments go to run_episode.py (e.g. --no-pruning or --no-filter for a reference run).
#   scripts/run_campaign.sh outputs/nav "0 1 2" --plan plans/nav_mats.json --classifier qwen
set -euo pipefail
OUT=${1:?out dir}; SEEDS=${2:?seeds}; shift 2
export MUJOCO_GL=egl PYTHONHASHSEED=0
for SEED in $SEEDS; do
  python scripts/run_episode.py --seed "$SEED" --out "$OUT" "$@" 2>&1 | grep -E '^RESULT|Traceback|Error' | cut -c1-700
done
