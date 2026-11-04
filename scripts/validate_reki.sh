#!/usr/bin/env bash
# Check REKI-2B (kyutai/reki-2b-robocasa365) against its model-card numbers with REKI's own rollout
# code and the official RoboCasa365 multitask protocol: pretrain-split kitchens, robocasa 1.0.1 task
# horizons, episode seeds drawn from --seed 42. The sim runs the pinned stack of REKI's install.sh
# (robocasa b4684e6 = 1.0.1, robosuite 5ce6643, mujoco 3.3.1, numpy 2.2.5, OSMesa from mesalib 25.0.5).
#
#   bash scripts/validate_reki.sh OpenCabinet,PickPlaceCounterToCabinet 10
set -euo pipefail
TP=${THIRD_PARTY:-$HOME/Documents/Chewy/code/third_party}
REKI=$TP/reki
TASKS=${1:-OpenCabinet,PickPlaceCounterToCabinet}
N=${2:-10}
OUT=${3:-$HOME/Documents/Chewy/code/sesame_lite/outputs/reki_validation}
SIM_PY=$(conda run -n sesame-robocasa which python)

# robocasa 1.0.1 worktree shadows the editable robocasa of sesame-robocasa; cd away so no robocasa/ dir shadows it.
SERVER="cd /tmp && env PYTHONNOUSERSITE=1 PYTHONPATH=$TP/robocasa-1.0.1 NUMBA_DISABLE_JIT=1 \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_PRELOAD=$REKI/storage/osmesa/lib/libOSMesa.so.8 \
LD_LIBRARY_PATH=$REKI/storage/osmesa/lib \
$SIM_PY -s $REKI/scripts/eval/robocasa365_sim_server.py --port 0 --port-file {portfile} >> {portfile}.log 2>&1"

cd "$REKI"
env PYTHONNOUSERSITE=1 PYTHONPATH= TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 storage/env/bin/python -s scripts/eval/rollout.py \
    --env_type robocasa365 --task "$TASKS" --ckpt_dir storage/ckpt/reki-2b-robocasa365 \
    --n_episodes "$N" --num_workers 1 --sim_server_cmd "$SERVER" --log_dir "$OUT" --save_videos
