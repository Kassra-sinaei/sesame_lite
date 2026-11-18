"""rho_fixed for the fixed-margin row (EVAL_SPEC.md, section 5; Kasra, 2026-11-12: from the three seed-100 design maps).

The design scenes (plans close_fridge, nav_clutter, pick_place_far, seed 100) are mapped again with the current
mapping pass into outputs/eval/design_maps/; for every segment of each plan, the proposed filter's rows are formed
(pruning, delta split over the segment's admitted objects) and the radii of their balls are pooled over all points,
objects, segments and plans. rho_fixed is the median. Writes outputs/eval/results/rho_fixed.json.

    MUJOCO_GL=egl python -s scripts/eval_rho_fixed.py <plan>     (one plan per process: writes its radii)
    python scripts/eval_rho_fixed.py --combine
"""

import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402

from sesame_lite.eval_scene import EVAL_DIR  # noqa: E402

PLANS = ("close_fridge", "nav_clutter", "pick_place_far")
OUT = EVAL_DIR / "design_maps_v025"  # (the first campaign's 5 cm design maps: design_maps/)

if sys.argv[1] == "--combine":
    parts = {p: json.loads((OUT / f"radii_{p}.json").read_text()) for p in PLANS}
    rho = np.concatenate([np.concatenate([np.asarray(s["rho"]) for s in parts[p]["segments"]]) for p in PLANS])
    res = {"rho_fixed_m": float(np.median(rho)), "definition": "median of the proposed balls' radii rho_k over every point of every admitted (row) object, "
           "every segment, of the seed-100 design scenes close_fridge, nav_clutter, pick_place_far (maps rebuilt with the current mapping pass)",
           "n_points": int(len(rho)), "quantiles_m": dict(zip(["10%", "25%", "50%", "75%", "90%"], np.percentile(rho, [10, 25, 50, 75, 90]).round(5).tolist())),
           "per_plan": {p: {"n_points": sum(len(s["rho"]) for s in parts[p]["segments"]), "median_m": float(np.median(np.concatenate([s["rho"] for s in parts[p]["segments"]]))),
                            "segments": [{k: v for k, v in s.items() if k != "rho"} for s in parts[p]["segments"]]} for p in PLANS}}
    (EVAL_DIR / "results" / "rho_fixed.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_plan"}))
    sys.exit(0)

import sesame_lite.mapping as mapping  # noqa: E402
import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402

mapping.MAPS_DIR = OUT  # everything of the evaluation under outputs/eval/
name = sys.argv[1]
plan = load_plan(f"plans/{name}.json")
base = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=1200)
env = SafetyFilterEnv(base)
env.reset(seed=100, plan=plan)
segments = []
for k in range(len(env.segments)):
    env.begin_segment(k)
    rho = np.concatenate([b.rho for b in env.row_balls.values()] or [np.zeros(0)])
    segments.append({"segment": k, "kind": env.segments[k].kind, "n_admitted": len(env.row_ids), "row_objects": list(env.row_ids), "rho": rho.round(6).tolist()})
(OUT / f"radii_{name}.json").write_text(json.dumps({"plan": name, "snapshot_id": env.oracle_graph.snapshot_id, "segments": segments}))
print("RADII", name, [(s["kind"], s["n_admitted"], len(s["rho"])) for s in segments], flush=True)
base.close()
