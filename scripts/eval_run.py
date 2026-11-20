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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS (set before CUDA starts)
import torch  # noqa: E402

# repeatable runs: identical inputs give identical outputs on the GPU (pi0.5 and the filter's sphere model); without this a
# scene's outcome could change between identical runs (outputs/eval/results/revert_check.json)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

import sesame_lite.tasks  # noqa: F401,E402  (registers the Eval tasks)
from pi05_nominal import Pi05Policy  # noqa: E402
from sesame_lite.env import MissionRunner, SesameRoboCasaEnv, _json_default  # noqa: E402
from sesame_lite.eval_scene import DEBUG_DIR, EVAL_DIR, FOLLOWER_LOOKAHEAD_M, SCENES_DIR, EvalFilterEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402

METHODS = ("nominal", "ee", "fixed", "noissf", "proposed", "qwen_pairs", "patch", "nominal_nav", "aegis")
TITLES = {"nominal": "Nominal", "ee": "End-effector filter", "fixed": "Fixed margin", "noissf": "Proposed without ISSf", "proposed": "Proposed",
          "qwen_pairs": "Pairs chosen by Qwen (reference)", "patch": "Finger patch (reference)", "nominal_nav": "Nominal", "aegis": "AEGIS"}


def method_kwargs(method: str, overrides: dict) -> dict:
    """The rows of Table II as options of the one filter; ``overrides``: FilterPolicy constants of a tuning round, the same
    for every filtered method. In every row the language model (Qwen) labels the objects once per segment (stored per
    scene and map kind, shared by the methods), and the gripper may touch the objects the plan names (the task objects
    and the support of the picked object); Nominal has no filter and is scored with the same allowed pairs.
    Reference methods outside Table II: "qwen_pairs" (Qwen also chooses the objects the gripper may touch; rounds 5-6
    and the confusion matrix) and "patch" (whole task objects exempt, finger patch on the support; rounds 4-5)."""
    policy = FilterPolicy(**overrides)
    plan = {"pair_pruning": "rule", "classifier": "qwen"}
    if method == "nominal":
        return {"filter": False, "policy": policy, "pair_pruning": "rule"}
    if method == "ee":
        return {"rows_for": "hand", "policy": policy, **plan}
    if method == "fixed":  # the median Proposed radius (rho_fixed.json, computed with the default constants); every
        # Proposed radius contains m_d = d_bar^2 / (4 eps alpha), so with other constants the median moves by the change in m_d
        rho = json.loads((EVAL_DIR / "results" / "rho_fixed.json").read_text())["rho_fixed_m"] - FilterPolicy().m_d + policy.m_d
        return {"rho_fixed": rho, "policy": policy, **plan}
    if method == "noissf":
        return {"policy": FilterPolicy(**{**overrides, "issf": False}), **plan}
    if method == "proposed":
        return {"policy": policy, **plan}
    if method == "qwen_pairs":
        return {"policy": policy, "pair_pruning": "model", "classifier": "qwen"}
    if method in ("nominal_nav", "aegis"):  # comparison table v3: navigation by Proposed; manipulation without it (AEGIS: src/aegis)
        return {"policy": policy, **plan, "filter_segments": (0,)}
    return {"policy": policy}


ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--map", default="full", choices=("full", "look"))
ap.add_argument("--policy", default="{}", help='JSON FilterPolicy overrides (tuning rounds only), e.g. {"alpha": 3.0}')
ap.add_argument("--video", choices=("auto", "on", "off"), default="auto", help="auto: on for counted Nominal and Proposed runs on the full-pass map (section 9)")
ap.add_argument("--lookahead", type=float, default=FOLLOWER_LOOKAHEAD_M, help="base follower look-ahead [m] (tuning rounds only)")
ap.add_argument("--scenes-root", default=None, help="a directory of scene variants (tuning only), e.g. outputs/eval_debug/scenes_v0.025")
ap.add_argument("--horizon", type=int, default=None, help="episode length in ticks (default: the scene's)")
ap.add_argument("--policy-seed", type=int, default=None, help="seed of the XR1 policy's queries (default: the scene seed); repeated runs of a scene (v3, 2026-11-20)")
ap.add_argument("--tuning", default=None, help="a tuning round (stage 0b, not counted), e.g. r2")
ap.add_argument("--vla", default="pi05", choices=("pi05", "xr1"), help="the VLA nominal: lerobot/pi05_robocasa, or Xiaomi-Robotics-1-RoboCasa365 (scripts/xr1_server.py must run)")
ap.add_argument("--disturbance", default=None, choices=("d1", "d2", "a1", "a2"), help="a velocity disturbance (env.VelocityDisturbance), the same for every method")
ap.add_argument("--arm-kv", type=float, default=None, help="the arm drive's tracking gain (default: JointVelocityDrive.kv = 100), the same for every method")
args = ap.parse_args()
scene_dir = (Path(args.scenes_root) if args.scenes_root else SCENES_DIR) / args.scene
scene = json.loads((scene_dir / "scene.json").read_text())
out = (DEBUG_DIR / "tuning" / args.tuning if args.tuning else EVAL_DIR / "runs") / args.scene / f"{args.method}_{args.map}.json"
if out.exists():
    print(f"RESULT exists {out}")
    sys.exit(0)
t0 = time.time()
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=args.horizon or scene["horizon"])
env_cls, video_cls = EvalFilterEnv, None
if args.method == "aegis":
    from aegis.env import AegisEnv, AegisVideo

    env_cls, video_cls = AegisEnv, AegisVideo
env = env_cls(base, map_kind=args.map, disturbance=args.disturbance, arm_kv=args.arm_kv, **method_kwargs(args.method, json.loads(args.policy)))
obs, _ = env.reset(scene_dir=scene_dir)
env.method_title = TITLES[args.method] + (", look-around map" if args.map == "look" else "")
if args.vla == "xr1":
    from xr1_nominal import XR1Policy

    vla = XR1Policy(base, scene["seed"] if args.policy_seed is None else args.policy_seed)
else:
    vla = Pi05Policy(scene["task"])
video = None
if args.video == "on" or (args.video == "auto" and not args.tuning and args.method in ("nominal", "proposed") and args.map == "full"):
    from sesame_lite.viz import EvalVideo

    video = (video_cls or EvalVideo)(env, (DEBUG_DIR / "tuning_videos" / args.tuning if args.tuning else EVAL_DIR / "videos") / args.scene / f"{args.method}_{args.map}")
torch.manual_seed(scene["seed"])  # the policy's sampling noise: the same sequence for every method of a scene
success, stalled = MissionRunner(env, vla=vla, on_frame=video, lookahead=args.lookahead).run(obs)
result = {"scene": args.scene, "method": args.method, "map": args.map, "seed": scene["seed"], "policy_seed": scene["seed"] if args.policy_seed is None else args.policy_seed, "follower_lookahead_m": args.lookahead, "disturbance_level": args.disturbance, "vla": args.vla, "wall_s": time.time() - t0, **env.summary(success, stalled)}
env.dump(out, result)
if video is not None:
    video.close(result)
keys = ("scene", "method", "map", "outcome", "success", "steps", "contact_ticks", "true_clearance_min", "relaxed_share", "unmapped_task_objects", "wall_s")
print("RESULT " + json.dumps({k: result[k] for k in keys}, default=_json_default), flush=True)
base.close()
