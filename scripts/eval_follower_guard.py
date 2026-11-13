"""Guard for the base follower's look-ahead (tuning_amendments.md, A3): the shared path of a scene driven without
clutter (all six clutter objects stored under the floor), filter off, driving segment only.

    MUJOCO_GL=egl python -s scripts/eval_follower_guard.py --scene open_drawer_s1 --lookahead 0.6

Prints one GUARD line: reached the standoff (the unchanged done check), ticks, final pose against the goal, largest
sideways deviation from the path, smallest true clearance and every fixture touched.
"""

import argparse
import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import NavFollower, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import COUNTER_CLUTTER, FLOOR_CLUTTER, RULE, SCENES_DIR, EvalFilterEnv, hold_still, place_body, stored_bodies, _Prep  # noqa: E402
from sesame_lite.scene_graph import _raw_data, _raw_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--lookahead", type=float, required=True)
ap.add_argument("--max-ticks", type=int, default=800)
args = ap.parse_args()
scene = json.loads((SCENES_DIR / args.scene / "scene.json").read_text())
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, filter=False)
env.reset(scene_dir=SCENES_DIR / args.scene)
m, d = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
prep = _Prep(env, scene["instruction"], scene["seed"], SCENES_DIR / args.scene)
names = list(FLOOR_CLUTTER + COUNTER_CLUTTER)
for k, name in enumerate(names):  # the scene without clutter
    c = prep.clutter[name]
    place_body(m, d, c["body"], c["gids"], np.asarray(scene["standoff"]["xy"]) + np.array([0.5 * k, 0.0]), 0.0, prep.floor_z - RULE["store_below_floor_m"])
hold_still(m, stored_bodies(env.rs, names))
seg = env.segment
follower = NavFollower(seg.path_polyline_xy, seg.goal_yaw, policy=env.policy, lookahead=args.lookahead)
path = seg.path_polyline_xy
while env.tick < args.max_ticks and not env.segment_done() and not env.stalled():
    env.step(follower(env.robot_pose()[0][:2], env.base_yaw(), env.base_theta()))


def lateral(p):
    s = path[1:] - path[:-1]
    t = np.clip(np.einsum("ij,ij->i", p - path[:-1], s) / np.einsum("ij,ij->i", s, s), 0, 1)
    return float(np.min(np.linalg.norm(path[:-1] + t[:, None] * s - p, axis=1)))


gt = env.ground_truth
xy, yaw = env.robot_pose()[0][:2], env.base_yaw()
out = {"scene": args.scene, "lookahead_m": args.lookahead, "reached": bool(env.segment_done()), "ticks": env.tick,
       "final_xy_error_m": round(float(np.linalg.norm(xy - path[-1])), 4), "final_yaw_error_rad": round(float(np.arctan2(np.sin(yaw - seg.goal_yaw), np.cos(yaw - seg.goal_yaw))), 4),
       "max_sideways_m": round(max(lateral(np.asarray(g["base_xy"])) for g in gt), 4),
       "true_clearance_min_m": round(min((g["clearance"] for g in gt if g["clearance"] is not None), default=float("nan")), 4),
       "touched": sorted({o for g in gt for o in g["touched"]}), "touched_ticks": sum(bool(g["touched"]) for g in gt)}
print("GUARD " + json.dumps(out), flush=True)
base.close()
