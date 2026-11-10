"""Build (or check) the Hydra maps of one mission scene in their own process: the reset does the mapping passes.

    MUJOCO_GL=egl .venv/bin/python -s scripts/map_scene.py --plan plans/nav_mats.json --seed 100

Writes outputs/maps/<mission>_s<seed>_<n>/ (n = 0 at reset, 1 after the injections) unless the cache exists.
``plan_mission.py`` and ``run_episode*.py`` with ``--graph-source hydra`` then load the same maps, so node ids agree
between planning and execution. ``--labels category`` writes a second map with category labels instead of instance
labels (the check of spec 4.8 item 5, no episode with it) to outputs/maps/<mission>_s<seed>_<n>_category/.
"""

import argparse
import json
import os
import sys
from dataclasses import replace

if os.environ.get("PYTHONHASHSEED") != "0":  # same scene as run_episode (see sesame_lite/determinism.py)
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402  (registers the task variants)
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.mapping import MappingParams, build_map, map_dir  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--labels", default="instance", choices=("instance", "category"))
args = ap.parse_args()

mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False, graph_source="hydra")
env.reset(seed=args.seed, plan=mission)  # builds the instance-label maps (or loads them and checks the scene)
if args.labels == "category":  # the same kitchen state as the last instance map, mapped again with category labels
    out = map_dir(mission["mission"], args.seed, env.n_capture - 1, labels="category")
    build_map(env.rs, env.oracle_graph, env.oracle_lookup, out, seed=args.seed, params=replace(MappingParams(), labels="category"))
for n in range(env.n_capture):
    d = map_dir(mission["mission"], args.seed, n)
    meta = json.loads((d / "meta.json").read_text())
    print("MAP " + json.dumps({"dir": str(d), "oracle_snapshot_id": meta["oracle_snapshot_id"], "hydra": meta["hydra"], "seconds": meta["seconds"],
                               "nodes": len(env.graph.nodes) if n == env.n_capture - 1 else None}))
base.close()
