"""HANDOFF §7 check: does one seed build one kitchen in every process? Prints one JSON line with the oracle 3DSG
snapshot_id for (plan, seed). Run it in several processes and compare. ``--no-counter-patch`` restores RoboCasa's
``set`` in counter.py (control: shows whether the address-ordered dedupe changes the scene). Server-side addition.

    MUJOCO_GL=egl python -s scripts/check_determinism.py --plan plans/open_drawer.json --seed 0
"""

import argparse
import builtins
import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402  (registers NavigateKitchenClutter)
from sesame_lite.determinism import SAMPLER_SEED  # noqa: E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.scene_graph import capture_scene_graph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-counter-patch", action="store_true")
args = ap.parse_args()
plan = load_plan(args.plan)
env = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=plan.get("horizon"))
if args.no_counter_patch:
    from robocasa.models.fixtures import counter

    counter.set = builtins.set
env.reset(seed=args.seed)
meta = env.rs.get_ep_meta()
g, _ = capture_scene_graph(env.rs)
print("DET " + json.dumps({"plan": plan["mission"], "seed": args.seed, "counter_patch": not args.no_counter_patch, "pid": os.getpid(),
                           "snapshot_id": g.snapshot_id, "n_nodes": len(g.obstacle_nodes()), "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"),
                           "lang": meta.get("lang"), "unseeded_rng_calls": SAMPLER_SEED[1]}), flush=True)
env.close()
