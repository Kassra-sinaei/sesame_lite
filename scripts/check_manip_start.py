"""Comparison table v3, step 1: is the state at the start of the manipulation segment the same in two rows?

    python -s scripts/check_manip_start.py --ref "runs/{scene}/proposed_full.json" --row "results/v3/shared/runs/{scene}/nominal_nav_full.json"

Per scene: the first tick of segment 1 in each run, the largest difference of the robot's joints (q: base, torso, arm,
fingers) and of the whole joint state (qpos: robot, doors, drawers, free objects) at that tick, and the largest robot
joint difference over the navigation ticks. Scenes without both run files are listed as missing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_segment import V3_SCENES  # noqa: E402


def start(path: Path):
    d = json.loads(path.read_text())
    seg = np.array([g["segment"] for g in d["ground_truth"]])
    z = np.load(path.with_suffix(".npz"))
    return int(np.argmax(seg == 1)) if (seg == 1).any() else None, z["q"], z["qpos"]


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--row", required=True)
ap.add_argument("--scenes", nargs="+", default=V3_SCENES)
args = ap.parse_args()
worst_q = worst_qpos = 0.0
for scene in args.scenes:
    a, b = Path(args.ref.format(scene=scene)), Path(args.row.format(scene=scene))
    if not (a.exists() and b.exists()):
        print(f"{scene:24s} missing")
        continue
    (ta, qa, pa), (tb, qb, pb) = start(a), start(b)
    if ta is None or tb is None or ta != tb:
        print(f"{scene:24s} manipulation starts at tick {ta} and {tb}")
        worst_q = worst_qpos = np.inf
        continue
    dq, dp, nav = float(np.abs(qa[ta] - qb[tb]).max()), float(np.abs(pa[ta] - pb[tb]).max()), float(np.abs(qa[:ta] - qb[:tb]).max())
    worst_q, worst_qpos = max(worst_q, dq), max(worst_qpos, dp)
    print(f"{scene:24s} tick {ta}: robot joints {dq:.3g}, whole joint state {dp:.3g}; robot joints over the navigation {nav:.3g}")
print(f"largest difference at the start of manipulation: robot joints {worst_q:.3g}, whole joint state {worst_qpos:.3g}")
