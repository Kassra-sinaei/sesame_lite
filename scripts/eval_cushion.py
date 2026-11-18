"""The cushion between the barrier and the true clearance, on maps of two voxel sizes (tuning, not counted).

    MUJOCO_GL=egl python -s scripts/eval_cushion.py --round t1 --methods proposed noissf --roots A B [--every 5]

For each run of the round, the logged joint states are replayed in each scene variant (the same scene with its maps
built at another voxel size, roots A and B under outputs/eval_debug/). At every ``every``-th tick of the manipulation:
- h: the smallest value, over the robot's spheres and the map's points, of distance(sphere centre, point) - point
  radius - sphere radius, with the point radius of PDF eq. 4 (delta_j = 0.01, the filter's m_d), over the objects that
  keep their rows (the plan's task objects and their supports are left out);
- the true clearance: the smallest signed distance from the robot's spheres to the true geometry of the same objects;
- cushion = true clearance - h.
Reported for the ticks where h < 5 cm (where the filter acts): median and 10/90th percentiles. Also the time of one
barrier evaluation on the GPU (SphereFilter with every object of the segment) and the map size.
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
import torch  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import DEBUG_DIR, EvalFilterEnv, apply_scene  # noqa: E402
from sesame_lite.filter import FilterPolicy, SphereFilter  # noqa: E402
from sesame_lite.planner import resolve_role  # noqa: E402
from sesame_lite.robot_spheres import RobotSpheres  # noqa: E402
from sesame_lite.scene_graph import HydraSceneGraph, IdBridge, _collision_geoms, _raw_data, _raw_model, capture_scene_graph  # noqa: E402
from sesame_lite.truth import CollisionGeometry  # noqa: E402
from sesame_lite.uncertainty import balls  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--round", required=True)
ap.add_argument("--methods", nargs="+", default=["proposed"])
ap.add_argument("--roots", nargs="+", required=True)
ap.add_argument("--every", type=int, default=5)
args = ap.parse_args()
pol = FilterPolicy()
out = {r: {"cushion": [], "h": [], "eval_ms": [], "points": []} for r in args.roots}
for run in sorted(p for m in args.methods for p in (DEBUG_DIR / "tuning" / args.round).glob(f"*/{m}_full.json")):
    d = json.loads(run.read_text())
    q_log = np.load(run.with_suffix(".npz"))["q"]
    seg = np.array([g["segment"] for g in d["ground_truth"]])
    ticks = np.flatnonzero(seg == 1)[::args.every]
    for root in args.roots:
        sdir = DEBUG_DIR / root / run.parent.name
        if not (sdir / "scene.json").exists():
            continue
        scene = json.loads((sdir / "scene.json").read_text())
        base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
        env = EvalFilterEnv(base, filter=False)
        env.env.reset(seed=scene["seed"])
        apply_scene(env, scene)
        m, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
        oracle, lookup = capture_scene_graph(env.rs)
        robot = RobotSpheres(m, data)
        # the objects that keep their rows: all but the plan's task objects and what they stand on
        free = {resolve_role(scene["plan"]["roles"][r], env.rs, lookup) for r in scene["plan"]["exempt"]}
        free |= {lookup.support[o] for o in list(free) if o in lookup.support}
        graph = HydraSceneGraph(sdir / "map_full", oracle.labels, robot_pose=(np.zeros(3), 0.0))
        bridge = IdBridge.from_instances(graph, oracle)
        p = np.load(sdir / "map_full" / "points.npz")
        node_ids = [str(x) for x in p["node_ids"]]
        P, rho, owner = [], [], []
        for k, nid in enumerate(node_ids):
            a, b = p["offsets"][k], p["offsets"][k + 1]
            o = bridge.oracle_of.get(nid)
            if b <= a or o is None or o in free:
                continue
            bl = balls(p["points"][a:b], p["std"][a:b], delta_j=0.01, voxel_size=float(p["voxel_size"]), m_r=float(p["m_r"]), m_d=pol.m_d)
            P.append(bl.points), rho.append(bl.rho), owner.append(o)
        objs = sorted(set(owner))
        geo = CollisionGeometry(m, {o: _collision_geoms(m, lookup.bodies[o]) for o in objs})
        Pt, Rt = torch.tensor(np.vstack(P), device="cuda", dtype=torch.float64), torch.tensor(np.concatenate(rho), device="cuda", dtype=torch.float64)
        out[root]["points"].append(int(len(Pt)))
        flt = SphereFilter(robot, pol)  # the filter's own barrier evaluation, timed
        flt.set_objects({f"o{i}": balls(P[i], np.zeros(len(P[i])), delta_j=0.01, voxel_size=float(p["voxel_size"]), m_r=float(p["m_r"]), m_d=pol.m_d) for i in range(len(P))}, None)
        for t in ticks:
            c, r = robot.spheres(q_log[t])
            ct = torch.tensor(c, device="cuda", dtype=torch.float64)
            h = float((torch.cdist(ct, Pt) - Rt[None] - torch.tensor(r, device="cuda", dtype=torch.float64)[:, None]).min())
            true, _ = geo.clearance(data, c, r)
            if true is not None and h < 0.05:
                out[root]["cushion"].append(true - h)
                out[root]["h"].append(h)
            if t == ticks[len(ticks) // 2]:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    flt.barrier(q_log[t])
                torch.cuda.synchronize()
                out[root]["eval_ms"].append((time.perf_counter() - t0) / 20 * 1e3)
        base.close()
    print("DONE", run.parent.name, run.stem, flush=True)
for root, v in out.items():
    c = np.array(v["cushion"])
    print("CUSHION " + json.dumps({"root": root, "ticks_h_below_5cm": len(c), "cushion_cm_median": round(100 * float(np.median(c)), 2) if len(c) else None,
                                   "cushion_cm_10_90": [round(100 * float(x), 2) for x in np.percentile(c, [10, 90])] if len(c) else None,
                                   "barrier_eval_ms_median": round(float(np.median(v["eval_ms"])), 2) if v["eval_ms"] else None,
                                   "map_points_median": int(np.median(v["points"])) if v["points"] else None}))
