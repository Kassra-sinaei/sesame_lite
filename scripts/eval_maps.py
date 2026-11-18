"""Build the two maps of a prepared scene again from its scene.json (the maps were deleted to save space).

    MUJOCO_GL=egl python -s scripts/eval_maps.py --scene counter_to_sink_s1000 [--out DIR]

The scene is set up exactly as stored (eval_scene.apply_scene); its oracle snapshot must equal the stored one, so the
kitchen and every object are the scene's own. The maps use the scene seed for the depth noise, as in prepare_scene.
``--out``: write map_full and map_look there instead of into the scene folder (a check that two builds agree).
Prints one MAPS line.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import SCENES_DIR, EvalFilterEnv, apply_scene  # noqa: E402
from sesame_lite.mapping import build_map  # noqa: E402
from sesame_lite.scene_graph import capture_scene_graph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()
src = SCENES_DIR / args.scene
scene = json.loads((src / "scene.json").read_text())
out = Path(args.out) if args.out else src
t0 = time.time()
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, filter=False)
env.env.reset(seed=scene["seed"])
apply_scene(env, scene)
oracle, lookup = capture_scene_graph(env.rs)
line = {"scene": args.scene, "snapshot_ok": oracle.snapshot_id == scene["snapshot_id"]}
if line["snapshot_ok"]:
    out.mkdir(parents=True, exist_ok=True)
    build_map(env.rs, oracle, lookup, out / "map_full", seed=scene["seed"])
    build_map(env.rs, oracle, lookup, out / "map_look", seed=scene["seed"], look_from=scene["start"]["xy"])
line["wall_s"] = round(time.time() - t0, 1)
print("MAPS " + json.dumps(line), flush=True)
base.close()
