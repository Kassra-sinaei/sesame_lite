#!/usr/bin/env bash
# Matched-seed runs: one process per seed runs every arm, each on the kitchen rebuilt from the seed (~6 GB per process).
#   scripts/run_campaign.sh outputs/nav "0 1 2" "A B C" --plan plans/nav_mats.json --classifier claude
set -euo pipefail
OUT=${1:?out dir}; SEEDS=${2:?seeds}; ARMS=${3:?arms}; shift 3
export MUJOCO_GL=egl PYTHONHASHSEED=0
for SEED in $SEEDS; do
  python scripts/run_episode.py --seed "$SEED" --arms $ARMS --out "$OUT" "$@" 2>&1 | grep -E '^RESULT|WARNING: scene|Traceback|Error' | cut -c1-700
done
