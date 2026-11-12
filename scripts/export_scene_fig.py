"""Export one trial for an external figure: exports/scene_fig/ (all positions in metres, world frame).

    MUJOCO_GL=egl .venv/bin/python -s scripts/export_scene_fig.py \
        --episode outputs/demos/pick_place_far/PickPlaceSinkToCounterFar_s100_sesame.json

The step is chosen by a fixed rule: in the last segment (the manipulation segment of pick_place_far), the tick with the
smallest barrier value h, i.e. where a robot sphere comes closest to an admitted object.

  mesh.npz       Hydra's mesh of the kitchen: vertices (V, 3), faces (F, 3), colors (V, 3) in [0, 1]
  objects.npz    per object node k (concatenated, with offsets): points (M, 3), sigma (M,) = s_m (eq. 2), rho (M,) = the
                 ball radius of eq. 4 for this segment; node ids, labels, admitted and exempt flags, kappa_j, s_max_j
  objects.json   the same per node, with the arrays as lists
  robot.npz      sphere centres (N, 3) and radii (N,) at the step, the base path of the whole trial (T, 3: x, y and the
                 height of the base centre), and the step index
  meta.json      kitchen id, instruction, delta, kappa, s_max (sigma_max), N and the other filter values of the run
  frame.png      the episode video's frame of the same step (outside camera; the robot spheres are drawn on it)

"admitted" means the object has rows in this segment (admitted by pruning and not exempt). Every object's balls use the
segment's delta_j = delta / (number of admitted objects); kappa_j depends on its own point count (eq. 4).
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import av  # noqa: E402
import numpy as np  # noqa: E402
import spark_dsg as dsg  # noqa: E402
from PIL import Image  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True)
ap.add_argument("--out", default="exports/scene_fig")
args = ap.parse_args()
path = Path(args.episode)
ep = json.loads(path.read_text())
res, plan = ep["result"], ep["plan"]
q_log = np.load(path.with_suffix(".npz"))["q"]
seg = max(r["segment"] for r in ep["receipts"])
step = min((r for r in ep["receipts"] if r["segment"] == seg), key=lambda r: r["h_min"])["tick"]

base = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=plan.get("horizon"))
env = SafetyFilterEnv(base, filter=True, classifier=res["classifier"], policy=FilterPolicy(**{k: res["policy"][k] for k in ("alpha", "d_bar", "tau_m", "delta")}))
env.reset(seed=res["seed"], plan=plan)  # the same scene and the same cached map
env.begin_segment(seg)  # the segment's pruning and rows
if list(env.row_ids) != list(ep["row_ids"]):  # the episode file records the last segment's rows
    raise RuntimeError(f"the rows rebuilt here differ from the episode's: {env.row_ids} vs {ep['row_ids']}")
policy = env.policy
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# mesh
g = dsg.DynamicSceneGraph.load(str(env.graph.map_dir / "dsg.json"))
raw = np.asarray(g.mesh.get_vertices(), dtype=np.float64)
np.savez_compressed(out / "mesh.npz", vertices=raw[:3].T, faces=np.asarray(g.mesh.get_faces(), dtype=np.int64).T, colors=raw[3:6].T)

# objects: every node's balls for this segment (delta split over the admitted objects)
n_rows = len(env.row_ids)
balls = env.graph.balls(n_admitted=n_rows, delta=policy.delta, m_d=policy.m_d)
rows, exempt = set(env.row_ids), set(env.segments[seg].exempt_ids)
ids, objs, P, S, R, off = [], [], [], [], [], [0]
for nid, node in env.graph.nodes.items():
    b = balls[nid]
    ids.append(nid)
    objs.append({"id": nid, "label": node.category, "admitted": nid in rows, "exempt": nid in exempt, "n_points": int(len(b.points)),
                 "kappa": b.kappa, "s_max": b.s_max, "points": b.points.round(5).tolist(), "sigma": node.std.round(6).tolist(), "rho": b.rho.round(5).tolist()})
    P.append(b.points)
    S.append(node.std)
    R.append(b.rho)
    off.append(off[-1] + len(b.points))
np.savez_compressed(out / "objects.npz", ids=np.array(ids), labels=np.array([o["label"] for o in objs]), admitted=np.array([o["admitted"] for o in objs]),
                    exempt=np.array([o["exempt"] for o in objs]), kappa=np.array([o["kappa"] for o in objs]), s_max=np.array([o["s_max"] for o in objs]),
                    offsets=np.array(off), points=np.vstack(P), sigma=np.concatenate(S), rho=np.concatenate(R))
(out / "objects.json").write_text(json.dumps(objs))

# robot at the step, and the base path (x, y and the height of the base centre)
centres, radii = env.robot.spheres(q_log[step])
z = float(env.robot_pose()[0][2])
path_xyz = np.array([[*gt["base_xy"], z] for gt in ep["ground_truth"]])
np.savez_compressed(out / "robot.npz", centres=centres, radii=radii, links=np.array(env.robot.links), base_path=path_xyz, step=step)

# meta
adm = [o for o in objs if o["admitted"]]
meta = {
    "kitchen": res["layout"], "task": res["task"], "mission": plan["mission"], "seed": res["seed"], "instruction": res["task_description"],
    "step": step, "segment": seg, "segment_kind": ep["segments"][seg]["kind"], "ticks": res["steps"], "outcome": res["outcome"],
    "rule_for_the_step": "in the last segment, the tick with the smallest barrier value h",
    "h_min_at_step": next(r for r in ep["receipts"] if r["tick"] == step)["h_min"],
    "closest_pair_at_step": next(r for r in ep["receipts"] if r["tick"] == step)["h_min_pair"],
    "N_spheres": int(len(radii)), "N_admitted_objects": n_rows, "N_objects": len(objs),
    "delta": policy.delta, "delta_j": policy.delta / n_rows,
    "kappa_admitted": {"min": min(o["kappa"] for o in adm), "median": float(np.median([o["kappa"] for o in adm])), "max": max(o["kappa"] for o in adm)},
    "sigma_max_admitted_m": {"min": min(o["s_max"] for o in adm), "median": float(np.median([o["s_max"] for o in adm])), "max": max(o["s_max"] for o in adm)},
    "sigma_max_rule": "s_max = voxel_size / kappa_j (PDF, Numbers)",
    "m_r_m": env.graph.m_r, "e_max_m": env.graph.e_max, "m_d_m": policy.m_d, "voxel_size_m": env.graph.voxel_size_m,
    "alpha": policy.alpha, "d_bar": policy.d_bar, "eps": policy.eps, "tau_m": policy.tau_m,
    "units": "metres, world frame",
}
(out / "meta.json").write_text(json.dumps(meta, indent=1))

# frame: the video frame rendered after tick step - 1, i.e. the state at the start of tick `step` (the q above)
with av.open(str(path.with_suffix(".mp4"))) as c:
    for k, f in enumerate(c.decode(video=0)):
        if k == step - 1:
            Image.fromarray(f.to_ndarray(format="rgb24")).save(out / "frame.png")
            break
print("EXPORT " + json.dumps({k: meta[k] for k in ("step", "N_spheres", "N_admitted_objects", "kappa_admitted", "sigma_max_admitted_m")}))
base.close()
