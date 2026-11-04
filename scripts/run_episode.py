"""Run the arms of one (task, seed) in one process (each arm rebuilds the same kitchen from the seed) and write receipts (+ videos).

    python scripts/run_episode.py --plan plans/nav_mats.json --seed 0 --arms A B C --classifier claude \
        --env-kwargs '{"layout_ids": [46], "style_ids": [46], "clutter_mode": 1}' --video
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # string hashing orders a few RoboCasa choices; fix it before anything is imported
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

from sesame_lite.env import ARMS, CLASSIFIERS, _json_default, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="NavigateKitchen")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arms", nargs="+", default=["A", "C"], choices=ARMS)
ap.add_argument("--nominal", default="auto", choices=("auto", "nav", "smolvla"))
ap.add_argument("--classifier", default="prior", choices=tuple(CLASSIFIERS), help="stage-2 classifier used by arm C")
ap.add_argument("--plan", default=None, help="plans/<mission>.json; default: plan from the task definition")
ap.add_argument("--env-kwargs", default="{}", help='JSON forwarded to the RoboCasa Kitchen, e.g. {"layout_ids":[46],"clutter_mode":1}')
ap.add_argument("--graph-noise", default=None, help='JSON {"eps_pos_m":..,"eps_yaw_rad":..,"eps_extent_m":..} (robustness hook)')
ap.add_argument("--out", default="outputs/episodes")
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--video", action="store_true")
args = ap.parse_args()
for result in run_episode(
    task=args.task, seed=args.seed, arms=args.arms, nominal=args.nominal, classifier=args.classifier, out_dir=Path(args.out),
    plan_path=args.plan, env_kwargs=json.loads(args.env_kwargs), max_steps=args.max_steps,
    graph_noise=json.loads(args.graph_noise) if args.graph_noise else None, video=args.video,
):
    print("RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("layout", "exempt")}, default=_json_default))
