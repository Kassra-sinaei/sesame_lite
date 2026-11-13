"""One evaluation run: a prepared scene (scripts/eval_prepare.py), one method, one map kind (EVAL_SPEC.md, section 5).

    MUJOCO_GL=egl python -s scripts/eval_run.py --scene counter_to_sink_s1000 --method nominal --map full

Writes outputs/eval/runs/<scene>/<method>_<map>.json (+ .npz, .dsg.json), or with ``--tuning r2`` the not-counted
run of a tuning round to outputs/eval_debug/tuning/r2/; a finished run is never repeated. The
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
from sesame_lite.eval_scene import DEBUG_DIR, EVAL_DIR, FOLLOWER_LOOKAHEAD_M, SCENES_DIR, EvalFilterEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402

METHODS = ("nominal", "ee", "fixed", "noissf", "proposed")
TITLES = {"nominal": "Nominal", "ee": "End-effector filter", "fixed": "Fixed margin", "noissf": "Proposed without ISSf", "proposed": "Proposed"}


def method_kwargs(method: str, overrides: dict) -> dict:
    """The rows of Table II as options of the one filter (pruning on in every filtered row); ``overrides``: FilterPolicy
    constants of a tuning round (stage 0b), the same for every filtered method."""
    policy = FilterPolicy(**overrides)
    if method == "nominal":
        return {"filter": False, "policy": policy}
    if method == "ee":
        return {"rows_for": "hand", "policy": policy}
    if method == "fixed":
        return {"rho_fixed": json.loads((EVAL_DIR / "results" / "rho_fixed.json").read_text())["rho_fixed_m"], "policy": policy}
    if method == "noissf":
        return {"policy": FilterPolicy(**{**overrides, "issf": False})}
    return {"policy": policy}


ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--map", default="full", choices=("full", "look"))
ap.add_argument("--policy", default="{}", help='JSON FilterPolicy overrides (tuning rounds only), e.g. {"alpha": 3.0}')
ap.add_argument("--video", choices=("auto", "on", "off"), default="auto", help="auto: on for counted Nominal and Proposed runs on the full-pass map (section 9)")
ap.add_argument("--lookahead", type=float, default=FOLLOWER_LOOKAHEAD_M, help="base follower look-ahead [m] (tuning rounds only)")
ap.add_argument("--scenes-root", default=None, help="a directory of scene variants (tuning only), e.g. outputs/eval_debug/scenes_v0.025")
ap.add_argument("--tuning", default=None, help="a tuning round (stage 0b, not counted), e.g. r2")
args = ap.parse_args()
scene_dir = (Path(args.scenes_root) if args.scenes_root else SCENES_DIR) / args.scene
scene = json.loads((scene_dir / "scene.json").read_text())
out = (DEBUG_DIR / "tuning" / args.tuning if args.tuning else EVAL_DIR / "runs") / args.scene / f"{args.method}_{args.map}.json"
if out.exists():
    print(f"RESULT exists {out}")
    sys.exit(0)
t0 = time.time()
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, map_kind=args.map, **method_kwargs(args.method, json.loads(args.policy)))
obs, _ = env.reset(scene_dir=scene_dir)
env.method_title = TITLES[args.method] + (", look-around map" if args.map == "look" else "")
vla = Pi05Policy(scene["task"])
video = None
if args.video == "on" or (args.video == "auto" and not args.tuning and args.method in ("nominal", "proposed") and args.map == "full"):
    from sesame_lite.viz import EvalVideo

    video = EvalVideo(env, (DEBUG_DIR / "tuning_videos" / args.tuning if args.tuning else EVAL_DIR / "videos") / args.scene / f"{args.method}_{args.map}")
torch.manual_seed(scene["seed"])  # the policy's sampling noise: the same sequence for every method of a scene
success, stalled = MissionRunner(env, vla=vla, on_frame=video, lookahead=args.lookahead).run(obs)
result = {"scene": args.scene, "method": args.method, "map": args.map, "seed": scene["seed"], "follower_lookahead_m": args.lookahead, "wall_s": time.time() - t0, **env.summary(success, stalled)}
env.dump(out, result)
if video is not None:
    video.close(result)
keys = ("scene", "method", "map", "outcome", "success", "steps", "contact_ticks", "true_clearance_min", "relaxed_share", "unmapped_task_objects", "wall_s")
print("RESULT " + json.dumps({k: result[k] for k in keys}, default=_json_default), flush=True)
base.close()
