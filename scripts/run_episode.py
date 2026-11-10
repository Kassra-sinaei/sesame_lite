"""Run one (task, seed) in one process (the kitchen is rebuilt from the seed) and write receipts (+ video).

    python scripts/run_episode.py --plan plans/nav_mats.json --seed 0 --classifier qwen \
        --env-kwargs '{"layout_ids": [46], "style_ids": [46], "clutter_mode": 1}' --video
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # string hashing orders a few RoboCasa choices; fix it before anything is imported
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

from sesame_lite.env import CLASSIFIERS, _json_default, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="NavigateKitchen")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-pruning", dest="pruning", action="store_false", help="every scene-graph node is a row (contact exemptions still apply)")
ap.add_argument("--no-filter", dest="filter", action="store_false", help="rows are evaluated, the action passes unchanged (no classifier call)")
ap.add_argument("--graph-source", default="oracle", choices=("oracle", "hydra"), help="scene graph of the filter side: the oracle, or Hydra's map (outputs/maps, built if missing)")
ap.add_argument("--nominal", default="auto", choices=("auto", "nav", "smolvla"))
ap.add_argument("--classifier", default="prior", choices=tuple(CLASSIFIERS), help="stage-2 classifier (used when pruning and filter are on)")
ap.add_argument("--plan", default=None, help="plans/<mission>.json; default: plan from the task definition")
ap.add_argument("--env-kwargs", default="{}", help='JSON forwarded to the RoboCasa Kitchen, e.g. {"layout_ids":[46],"clutter_mode":1}')
ap.add_argument("--graph-noise", default=None, help='JSON {"face_std_m": 0.01}: synthetic Gaussian face noise on the oracle graph (the fallback of the spec, not Hydra)')
ap.add_argument("--out", default="outputs/episodes")
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--video", action="store_true")
args = ap.parse_args()
result = run_episode(
    task=args.task, seed=args.seed, pruning=args.pruning, filter=args.filter, graph_source=args.graph_source, nominal=args.nominal, classifier=args.classifier, out_dir=Path(args.out),
    plan_path=args.plan, env_kwargs=json.loads(args.env_kwargs), max_steps=args.max_steps,
    graph_noise=json.loads(args.graph_noise) if args.graph_noise else None, video=args.video,
)
print("RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("layout", "exempt")}, default=_json_default))
