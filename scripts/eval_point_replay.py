"""text_numbers.json (EVAL_SPEC.md, section 10): rows and QP time with one row per map point instead of the composed
barrier (PDF eq. 5), replayed on the logged states of the Proposed runs.

    MUJOCO_GL=egl python -s scripts/eval_point_replay.py [--every 20]

For every Proposed run (full-pass map) the scene and its segments are rebuilt; on every ``--every``-th logged tick the
same QP is solved twice: with the filter's objects (composed barrier, one row per sphere and object) and with every map
point as its own object (one row per sphere and point). Same q, nominal command, known velocities and box U (the base
part of U from the base velocity measured between logged ticks). Writes outputs/eval/results/point_rows_replay.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR, EvalFilterEnv  # noqa: E402
from sesame_lite.filter import CONTROL_DT_S, SphereFilter  # noqa: E402
from sesame_lite.uncertainty import Balls  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--source", default=None, help="a directory of runs instead of outputs/eval/runs (a code check)")
ap.add_argument("--out", default=None, help="output file instead of outputs/eval/results/point_rows_replay.json")
args = ap.parse_args()


def solve(flt, q, rec, v0, env):
    bounds = flt.box(q, env.base_model.bounds(v0), env.arm_q_lo, env.arm_q_hi, CONTROL_DT_S)
    t0 = time.perf_counter()
    _, r, _ = flt.filter(tick=rec["tick"], q=q, u_nom=np.asarray(rec["u_nominal"]), known_qdot=np.asarray(rec["known_velocity"]), bounds=bounds)
    return r.n_pairs, r.n_rows_qp, r.solve_ms, (time.perf_counter() - t0) * 1e3


out = {"definition": __doc__.strip().splitlines()[0], "every_nth_tick": args.every, "runs": []}
for path in sorted((Path(args.source) if args.source else EVAL_DIR / "runs").glob("*/proposed_full.json")):
    d = json.loads(path.read_text())
    q_log = np.load(path.with_suffix(".npz"))["q"]
    scene = path.parent.name
    sc = json.loads((SCENES_DIR / scene / "scene.json").read_text())
    base = SesameRoboCasaEnv(sc["task"], env_kwargs=dict(sc["env_kwargs"]), episode_length=sc["horizon"])
    env = EvalFilterEnv(base, pair_pruning="rule", classifier="qwen")  # the scene's stored labels
    env.reset(scene_dir=SCENES_DIR / scene)
    stats = {"composed": [], "per_point": []}
    for seg in sorted({g["segment"] for g in d["ground_truth"]}):
        env.begin_segment(seg)
        balls, masks = env._filter_objects(env.segments[seg])
        point_balls, point_masks = {}, {}
        for oid, b in balls.items():
            for k in range(len(b.points)):
                point_balls[f"{oid}#{k}"] = Balls(points=b.points[k:k + 1], rho=b.rho[k:k + 1], capped=b.capped[k:k + 1], kappa=b.kappa, s_max=b.s_max)
                if masks and oid in masks:
                    point_masks[f"{oid}#{k}"] = masks[oid]
        composed, per_point = env.filter, SphereFilter(env.robot, env.policy, row_spheres=env.filter.row_spheres)
        per_point.set_objects(point_balls, point_masks or None)
        for rec in d["receipts"][::args.every]:
            t = rec["tick"]
            if rec["segment"] != seg or t == 0:
                continue
            q = q_log[t]
            v0 = (q_log[t, :3] - q_log[t - 1, :3]) / CONTROL_DT_S
            stats["composed"].append(solve(composed, q, rec, v0, env))
            stats["per_point"].append(solve(per_point, q, rec, v0, env))
    row = {"scene": scene, "ticks": len(stats["composed"])}
    for k, v in stats.items():
        a = np.array(v)
        row[k] = {"pairs_median": float(np.median(a[:, 0])), "rows_in_qp_median": float(np.median(a[:, 1])), "qp_ms_median": float(np.median(a[:, 2])),
                  "tick_ms_median": float(np.median(a[:, 3]))}
    out["runs"].append(row)
    print("REPLAY " + json.dumps(row), flush=True)
    base.close()
for k in ("composed", "per_point"):
    out[k] = {m: float(np.median([r[k][m] for r in out["runs"]])) for m in ("pairs_median", "rows_in_qp_median", "qp_ms_median", "tick_ms_median")} if out["runs"] else None
Path(args.out or EVAL_DIR / "results" / "point_rows_replay.json").write_text(json.dumps(out, indent=1))
