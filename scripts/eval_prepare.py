"""Prepare one evaluation scene (sesame_lite.eval_scene): outputs/eval/scenes/<instruction>_s<seed>/.

    MUJOCO_GL=egl python -s scripts/eval_prepare.py --instruction counter_to_sink --seed 1000 [--exclude a.json ...]

``--exclude``: JSON files with a "kitchens" list of {"layout", "style"}; such a kitchen is not built. Prints one
PREP line; a seed the rules cannot build leaves failed.json (with the reason) instead of scene.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402  (registers the Eval tasks)
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import ENV_KWARGS, HORIZON, INSTRUCTIONS, SCENES_DIR, EvalFilterEnv, ScenePrepError, prepare_scene, scene_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--instruction", required=True, choices=tuple(INSTRUCTIONS))
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--exclude", nargs="*", default=[])
args = ap.parse_args()
out = SCENES_DIR / scene_name(args.instruction, args.seed)
if (out / "scene.json").exists():
    print("PREP " + json.dumps({"scene": out.name, "status": "exists"}))
    sys.exit(0)
exclude = {(k["layout"], k["style"]) for f in args.exclude for k in json.loads(Path(f).read_text())["kitchens"]}
t0 = time.time()
base = SesameRoboCasaEnv(INSTRUCTIONS[args.instruction]["task"], env_kwargs=dict(ENV_KWARGS), episode_length=HORIZON)
env = EvalFilterEnv(base, filter=False)
try:
    scene = prepare_scene(env, args.instruction, args.seed, out, exclude)
    status = {"scene": out.name, "status": "ok", "kitchen": scene["kitchen"], "path_length_m": scene["start"]["path_length_m"], "language": scene["language"]}
except ScenePrepError as exc:
    meta = env.rs.get_ep_meta()
    status = {"scene": out.name, "status": "failed", "reason": str(exc), "kitchen": {"layout": meta["layout_id"], "style": meta["style_id"]}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "failed.json").write_text(json.dumps(status, indent=1))
status["wall_s"] = round(time.time() - t0, 1)
print("PREP " + json.dumps(status), flush=True)
base.close()
