"""One evaluation run: a prepared scene (scripts/eval_prepare.py), one method, one map kind (EVAL_SPEC.md, section 5).

    MUJOCO_GL=egl python -s scripts/eval_run.py --scene counter_to_sink_s1000 --method nominal --map full

Writes outputs/eval/runs/<scene>/<method>_<map>.json (+ .npz, .dsg.json); a finished run is never repeated. The
plan is the scene's shared plan (never replanned); pi0.5's sampling noise is seeded with the scene seed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402  (registers the Eval tasks)
from pi05_nominal import Pi05Policy  # noqa: E402
from sesame_lite.env import MissionRunner, SesameRoboCasaEnv, _json_default  # noqa: E402
from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR, EvalFilterEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402

METHODS = ("nominal", "ee", "fixed", "noissf", "proposed")


def method_kwargs(method: str) -> dict:
    """The rows of Table II as options of the one filter (pruning on in every filtered row)."""
    if method == "nominal":
        return {"filter": False}
    if method == "ee":
        return {"rows_for": "hand"}
    if method == "fixed":
        return {"rho_fixed": json.loads((EVAL_DIR / "results" / "rho_fixed.json").read_text())["rho_fixed_m"]}
    if method == "noissf":
        return {"policy": FilterPolicy(issf=False)}
    return {}


ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--map", default="full", choices=("full", "look"))
args = ap.parse_args()
scene_dir = SCENES_DIR / args.scene
scene = json.loads((scene_dir / "scene.json").read_text())
out = EVAL_DIR / "runs" / args.scene / f"{args.method}_{args.map}.json"
if out.exists():
    print(f"RESULT exists {out}")
    sys.exit(0)
t0 = time.time()
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, map_kind=args.map, **method_kwargs(args.method))
obs, _ = env.reset(scene_dir=scene_dir)
vla = Pi05Policy(scene["task"])
torch.manual_seed(scene["seed"])  # the policy's sampling noise: the same sequence for every method of a scene
success, stalled = MissionRunner(env, vla=vla).run(obs)
result = {"scene": args.scene, "method": args.method, "map": args.map, "seed": scene["seed"], "wall_s": time.time() - t0, **env.summary(success, stalled)}
env.dump(out, result)
keys = ("scene", "method", "map", "outcome", "success", "steps", "contact_ticks", "true_clearance_min", "relaxed_share", "unmapped_task_objects", "wall_s")
print("RESULT " + json.dumps({k: result[k] for k in keys}, default=_json_default), flush=True)
base.close()
