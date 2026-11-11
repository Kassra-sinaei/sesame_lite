"""Checks of the Hydra map of one scene, written next to the map: check.md, check.json.

    MUJOCO_GL=egl .venv/bin/python -s scripts/check_map.py --plan plans/nav_clutter.json --seed 100

The scene is rebuilt from the seed and the cached maps are loaded (nothing is mapped again). Ground truth is used
here, for evaluation only. There is no check of the map claim (PDF Lemma 1) in this version (spec v4, section 9).
  1. points (spec v4, section 5): points per node, points dropped for having no std, e_max and m_r, the share of
     capped points per node and the distribution of kappa_j * s_bar_m (kappa for every node admitted, as planning)
  2. match table: per oracle node, the Hydra node(s) of its instance, centre distance, extent difference, tilt, yaw
  3. the oriented-box check: OBB volume <= RAABB volume of the same vertices, and the tilt of upright fixtures
  4. the mapping route and the noise model on planar walls
  5. the category-label map of the same scene (if built with map_scene.py --labels category): merges, splits, and
     the clearance of the robot's start footprint from the balls
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402
import spark_dsg as dsg  # noqa: E402
from shapely.geometry import MultiPoint  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.mapping import map_dir  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402
from sesame_lite.planner import base_clearance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
args = ap.parse_args()

mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False)
env.reset(seed=args.seed, plan=mission)  # loads the cached maps; the last one is the scene the episode runs in
n_map = env.n_capture - 1
out_dir = map_dir(mission["mission"], args.seed, n_map)
hydra, oracle, bridge, labels = env.graph, env.oracle_graph, env.bridge, env.oracle_graph.labels
oracle_name = {v: k for k, v in env.oracle_lookup.node_of.items()}
policy = FilterPolicy()
report, summary = [], {}


def tilt_deg(R):
    R = np.asarray(R)
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2, :])), -1, 1))))


def yaw_deg(R):
    """Yaw of the box's most horizontal axis, folded to [-45, 45) degrees (an axis-aligned box has 0)."""
    R = np.asarray(R)
    k = int(np.argmin(np.abs(R[2, :])))
    a = np.degrees(np.arctan2(R[1, k], R[0, k]))
    return float((a + 45) % 90 - 45)


def world_dims(node):
    """Box dimensions along the world axes they are closest to (for the extent comparison)."""
    R = np.asarray(node.rotation if node.rotation is not None else np.eye(3))
    dims = 2 * np.asarray(node.half_extents)
    out = np.zeros(3)
    for i in range(3):
        out[int(np.argmax(np.abs(R[:, i])))] = dims[i]
    return out


def min_area_rect(xy):
    """Smallest-area rectangle around 2-D points (monotone-chain hull, rotating edges), returns its area."""
    pts = np.unique(np.round(np.asarray(xy, dtype=np.float64), 9), axis=0)
    if len(pts) < 3:
        return 0.0
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    if hull.geom_type != "Polygon":  # collinear points (a few-vertex node): no area
        return 0.0
    hull = np.asarray(hull.exterior.coords)[:-1]
    best = np.inf
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        n = np.linalg.norm(e)
        if n < 1e-12:
            continue
        ux = e / n
        uy = np.array([-ux[1], ux[0]])
        px, py = hull @ ux, hull @ uy
        best = min(best, (px.max() - px.min()) * (py.max() - py.min()))
    return float(best)


# ------------------------------------------------------------------ 1. points
pts_summary = json.loads((out_dir / "points_summary.json").read_text())
balls = env.plan_balls  # kappa for every node admitted (planning, standoff, pruning, start check)
point_rows = []
for nid, b in balls.items():
    node = hydra.nodes[nid]
    point_rows.append({"hydra_id": nid, "category": node.category, "points": len(b.points), "dropped": pts_summary["dropped_by_node"].get(nid, 0),
                       "kappa": b.kappa, "s_max_m": b.s_max, "capped_share": float(b.capped.mean()),
                       "kappa_s_bar_median_m": float(np.median(b.kappa * np.minimum(node.std, b.s_max))), "rho_max_m": float(b.rho.max())})
ks = np.concatenate([b.kappa * np.minimum(hydra.nodes[nid].std, b.s_max) for nid, b in balls.items()])
capped = np.concatenate([b.capped for b in balls.values()])
summary["points"] = {"nodes": len(balls), "points": int(sum(len(b.points) for b in balls.values())), "dropped_points": pts_summary["dropped_points"],
                     "nodes_without_points": hydra.nodes_without_points, "e_max_m": hydra.e_max, "m_r_m": hydra.m_r, "m_d_m": policy.m_d,
                     "std_source_counts": pts_summary["std_source_counts"], "capped_share": float(capped.mean()),
                     "kappa_s_bar_quantiles_m": dict(zip(["min", "10%", "50%", "90%", "99%", "max"], np.percentile(ks, [0, 10, 50, 90, 99, 100]).round(5).tolist()))}
P = summary["points"]
report += [f"# Hydra map check: {mission['mission']}, seed {args.seed}, map n = {n_map}", "",
           f"Map: `{out_dir.relative_to(Path.cwd()) if out_dir.is_relative_to(Path.cwd()) else out_dir}` (oracle snapshot `{oracle.snapshot_id}`).", "",
           "## 1. Points", "",
           f"- {P['points']} points on {P['nodes']} object nodes; {P['dropped_points']} mesh vertices dropped for having no std (per node below); nodes left with no point: "
           + (", ".join(f"H:{h}" for h in P["nodes_without_points"]) or "none") + ".",
           f"- Std from the grid edge (eq. 2): {P['std_source_counts']['grid_edge']} points; from the bound sigma0^2/min(w) of the voxels around the vertex: {P['std_source_counts']['bound']}.",
           f"- e_max = {100 * P['e_max_m']:.2f} cm (longest edge of the mesh triangles inside object nodes), m_r = e_max / sqrt(3) = {100 * P['m_r_m']:.2f} cm; m_d = {100 * P['m_d_m']:.1f} cm.",
           f"- With every node admitted (delta_j = {policy.delta} / {len(balls)}): capped points (s_m > s_max) {100 * P['capped_share']:.1f} % of all; "
           "kappa_j * s_bar_m: " + ", ".join(f"{k} {100 * v:.2f} cm" for k, v in P["kappa_s_bar_quantiles_m"].items()) + ".", "",
           "| Hydra | category | points | dropped | kappa_j | s_max [mm] | capped [%] | median kappa*s_bar [mm] | largest rho [cm] |", "|---|---|---|---|---|---|---|---|---|"]
report += [f"| H:{r['hydra_id']} | {r['category']} | {r['points']} | {r['dropped']} | {r['kappa']:.2f} | {1000 * r['s_max_m']:.1f} | {100 * r['capped_share']:.1f} | {1000 * r['kappa_s_bar_median_m']:.1f} | {100 * r['rho_max_m']:.1f} |" for r in point_rows] + [""]

# ------------------------------------------------------------------ 2. match table
rows, unmatched_oracle = [], []
for oracle_id, onode in oracle.nodes.items():
    hydra_ids = bridge.hydra_of.get(oracle_id, [])
    if not hydra_ids:
        unmatched_oracle.append(oracle_id)
        continue
    for hydra_id in hydra_ids:
        h = hydra.nodes[hydra_id]
        rows.append({"oracle_id": oracle_id, "hydra_id": hydra_id, "category": onode.category, "name": oracle_name[oracle_id],
                     "center_dist_m": float(np.linalg.norm(np.asarray(h.center) - np.asarray(onode.center))),
                     "extent_diff_m": (world_dims(h) - np.asarray(onode.extent)).tolist(),
                     "tilt_deg": tilt_deg(h.rotation), "yaw_deg": yaw_deg(h.rotation), "n_vertices": hydra.n_vertices(hydra_id)})
unmatched_hydra = [hid for hid in hydra.nodes if hid not in bridge.oracle_of]
summary["match"] = {"oracle_nodes": len(oracle.nodes), "hydra_nodes": len(hydra.nodes), "oracle_without_hydra": unmatched_oracle,
                    "hydra_without_oracle": unmatched_hydra, "instances_with_several_nodes": {o: v for o, v in bridge.hydra_of.items() if len(v) > 1}}
report += ["## 2. Match table (oracle node -> Hydra node of its instance)", "",
           "Extent difference = Hydra box dimensions (along the world axis each box axis is closest to) minus the oracle box extent; "
           "the oracle box bounds the collision geoms, the Hydra box the visible surface. Tilt = angle between the box's most vertical axis and the vertical; "
           "yaw = angle of its most horizontal axis, folded to [-45, 45) degrees (oracle boxes are axis-aligned, yaw 0).", "",
           "| oracle | Hydra | category | centre dist [cm] | extent diff x / y / z [cm] | tilt [deg] | yaw [deg] | vertices |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    ex = " / ".join(f"{100 * v:+.1f}" for v in r["extent_diff_m"])
    report.append(f"| O:{r['oracle_id']} | H:{r['hydra_id']} | {r['category']} | {100 * r['center_dist_m']:.1f} | {ex} | {r['tilt_deg']:.1f} | {r['yaw_deg']:+.1f} | {r['n_vertices']} |")
report += ["", f"**Oracle nodes with no Hydra node** ({len(unmatched_oracle)}): " + (", ".join(f"O:{o} ({oracle.nodes[o].category}, {' x '.join(f'{100 * e:.0f}' for e in oracle.nodes[o].extent)} cm, `{oracle_name[o]}`)" for o in unmatched_oracle) or "none") + ".",
           f"**Hydra nodes with no oracle node** ({len(unmatched_hydra)}): " + (", ".join(f"H:{h}" for h in unmatched_hydra) or "none") + ".",
           "**Instances seen as several Hydra nodes**: " + (", ".join(f"O:{o} -> " + ", ".join(f"H:{h}" for h in v) for o, v in bridge.hydra_of.items() if len(v) > 1) or "none") + ".", ""]

# ------------------------------------------------------------------ 2b. oriented boxes vs RAABB, tilt of upright fixtures
graph = dsg.DynamicSceneGraph.load(str(out_dir / "dsg.json"))
V = np.asarray(graph.mesh.get_vertices())[:3].T.astype(np.float64)
obb_rows = []
for node in graph.get_layer(dsg.DsgLayers.OBJECTS).nodes:
    hid = str(node.id)
    if hid not in hydra.nodes:
        continue
    P = V[np.asarray(list(node.attributes.mesh_connections))]
    raabb = min_area_rect(P[:, :2]) * float(np.ptp(P[:, 2]))
    obb = float(np.prod(np.asarray(node.attributes.bounding_box.dimensions, dtype=np.float64)))
    obb_rows.append({"hydra_id": hid, "obb_m3": obb, "raabb_m3": raabb, "tilt_deg": tilt_deg(hydra.nodes[hid].rotation), "fixture": hydra.is_fixture(hid)})
violations = [r for r in obb_rows if r["obb_m3"] > r["raabb_m3"] * (1 + 1e-4) + 1e-9]
fixture_tilts = np.array([r["tilt_deg"] for r in obb_rows if r["fixture"]])
leaning = [r for r in obb_rows if r["fixture"] and r["tilt_deg"] > 3.0]
summary["obb"] = {"nodes": len(obb_rows), "obb_larger_than_raabb": [r["hydra_id"] for r in violations], "fixture_tilt_deg_quantiles": np.percentile(fixture_tilts, [50, 90, 100]).round(2).tolist() if len(fixture_tilts) else None,
                  "fixtures_tilted_over_3_deg": [(r["hydra_id"], round(r["tilt_deg"], 1)) for r in leaning], "objects_tilted_over_3_deg": [(r["hydra_id"], round(r["tilt_deg"], 1)) for r in obb_rows if not r["fixture"] and r["tilt_deg"] > 3.0]}
report += ["## Oriented boxes", "",
           f"- OBB volume <= RAABB volume of the same vertices (RAABB recomputed here with an exact hull): {len(obb_rows) - len(violations)} of {len(obb_rows)} nodes" + (f"; larger: {', '.join(r['hydra_id'] for r in violations)}" if violations else "") + ".",
           f"- Tilt of fixture boxes against the vertical: median {np.median(fixture_tilts):.2f} deg, 90 % {np.percentile(fixture_tilts, 90):.2f} deg, max {fixture_tilts.max():.2f} deg; over 3 deg: " + (", ".join(f"H:{h} ({t} deg, {hydra.nodes[h].category})" for h, t in summary['obb']['fixtures_tilted_over_3_deg']) or "none") + ".",
           "- Fixtures over 3 deg, with their vertex count and OBB / upright-box volume: " + (", ".join(f"H:{r['hydra_id']} {hydra.nodes[r['hydra_id']].category} ({r['tilt_deg']:.1f} deg, {hydra.n_vertices(r['hydra_id'])} vertices, {r['obb_m3'] / r['raabb_m3']:.2f})" for r in leaning) or "none") + ".",
           "- Free objects tilted over 3 deg: " + (", ".join(f"H:{h} ({t} deg, {hydra.nodes[h].category})" for h, t in summary['obb']['objects_tilted_over_3_deg']) or "none") + ".", ""]

route = json.loads((out_dir / "meta.json").read_text())["route"]
lv = route.get("large_vertical_surfaces", {})
summary["route"] = {k: (len(v) if isinstance(v, list) else v) for k, v in lv.items()}
report += ["## Mapping route and large vertical surfaces", "",
           f"- Camera poses: {route['camera_poses']} ({route.get('targeted_poses', 0)} added so that every large vertical surface, >= 0.5 m^2, is seen within 45 deg of its normal).",
           f"- Large vertical surfaces: {lv.get('faces')}; seen within 45 deg by the base route: {lv.get('seen_by_base_route')}; given an added view: {len(lv.get('given_a_view', []))}; "
           f"no position sees them within 45 deg: {len(lv.get('not_reachable', []))}; hidden from every position (backs, outer wall faces): {len(lv.get('hidden', []))}.",
           "- Added views: " + ("; ".join(lv.get("given_a_view", [])) or "none") + ".",
           "- Not reachable within 45 deg: " + ("; ".join(lv.get("not_reachable", [])) or "none") + ".", ""]

from sesame_lite.scene_graph import _raw_data, _raw_model  # noqa: E402

model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
hydra_dsg = dsg.DynamicSceneGraph.load(str(out_dir / "dsg.json"))
Vmap = np.asarray(hydra_dsg.mesh.get_vertices())[:3].T.astype(np.float64)

# ------------------------------------------------------------------ 3b. noise model on planar walls
from sesame_lite.uncertainty import VoxelGrid, vertex_std  # noqa: E402

grid = VoxelGrid(out_dir / "voxels.npz")
sigma0_sq = json.loads((out_dir / "meta.json").read_text())["params"]["sigma0_sq"]
wall_rows = []
for hydra_id, hnode in hydra.nodes.items():
    oracle_id = bridge.to_oracle(hydra_id)
    if hnode.category != "wall" or oracle_id is None:
        continue
    P = Vmap[np.asarray(list(hydra_dsg.get_node(dsg.NodeSymbol("O", int(hydra_id[2:-1]))).attributes.mesh_connections))]
    far = (P[:, 2] > 0.2) & (P[:, 2] < 2.6)
    for o, other in oracle.nodes.items():
        if o != oracle_id:
            lo3, hi3 = other.aabb()
            far &= np.linalg.norm(np.maximum(np.maximum(lo3 - P, P - hi3), 0), axis=1) > 0.2
    sv, how = vertex_std(P, grid, sigma0_sq)
    ok = far & (how == 0)
    if ok.sum() < 100:
        continue
    Q = P[ok]
    w, E = np.linalg.eigh(np.cov((Q - Q.mean(0)).T))
    zres = ((Q - Q.mean(0)) @ E[:, 0]) / sv[ok]
    wall_rows.append((hydra_id, int(ok.sum()), float(np.std(zres)), float(np.median(np.abs(zres)))))
summary["planar_walls"] = wall_rows
report += ["## Noise model on planar walls", "",
           "Wall vertices more than 20 cm from every other oracle box and from floor and ceiling height: residual r from a fitted plane divided by the "
           "predicted vertex std s (eq. 2). With a correct noise model and sigma0^2, r/s ~ N(0, 1): std 1.00, median |r/s| 0.67.", "",
           "| Hydra node | vertices | std(r/s) | median abs(r/s) |", "|---|---|---|---|"] + [f"| H:{h} | {n} | {a:.2f} | {b:.2f} |" for h, n, a, b in wall_rows] + [""]

# ------------------------------------------------------------------ 5. category labels
cat_dir = map_dir(mission["mission"], args.seed, n_map, labels="category")


def start_clearance(graph_balls) -> float:
    """Smallest signed distance between the robot's start footprint and the balls in the base's height band."""
    return base_clearance(env.rect, env.robot_pose()[0][:2], env.base_yaw(), graph_balls, env.z_band)


summary["start_clearance_instance_map_m"] = start_clearance(balls)
if (cat_dir / "points.npz").exists():
    from sesame_lite.scene_graph import HydraSceneGraph

    bp, R, _ = env.robot_pose()
    cat_graph = HydraSceneGraph(cat_dir, labels, robot_pose=(bp, float(np.arctan2(R[1, 0], R[0, 0]))))
    cg = dsg.DynamicSceneGraph.load(str(cat_dir / "dsg.json"))
    CV = np.asarray(cg.mesh.get_vertices())[:3].T.astype(np.float64)
    lo_o = {o: n.aabb()[0] for o, n in oracle.nodes.items()}
    hi_o = {o: n.aabb()[1] for o, n in oracle.nodes.items()}
    members = {}  # category hydra node -> oracle instances covering its vertices
    for node in cg.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        hid = str(node.id)
        if hid not in cat_graph.nodes:
            continue
        cat = cat_graph.nodes[hid].label
        cands = [o for o, n in oracle.nodes.items() if n.label == cat]
        P = CV[np.asarray(list(node.attributes.mesh_connections))]
        if not cands:
            members[hid] = {}
            continue
        d = np.stack([np.linalg.norm(np.maximum(np.maximum(lo_o[o] - P, P - hi_o[o]), 0), axis=1) for o in cands], axis=1)
        owner = np.where(d.min(axis=1) <= 0.05, np.argmin(d, axis=1), -1)
        counts = {cands[k]: int((owner == k).sum()) for k in set(owner.tolist()) if k >= 0}
        members[hid] = {o: c for o, c in counts.items() if c >= max(10, 0.05 * len(P))}
    merged = {h: sorted(m) for h, m in members.items() if len(m) >= 2}
    by_oracle = {}
    for h, m in members.items():
        for o in m:
            by_oracle.setdefault(o, []).append(h)
    split = {o: hs for o, hs in by_oracle.items() if len(hs) >= 2}
    summary["category_labels"] = {"nodes": len(cat_graph.nodes), "instance_map_nodes": len(hydra.nodes), "merged": merged, "split": split,
                                  "oracle_instances_merged": sum(len(v) for v in merged.values()),
                                  "start_clearance_m": start_clearance(cat_graph.balls(n_admitted=len(cat_graph.nodes), delta=policy.delta, m_d=policy.m_d))}
    report += ["## 5. Category labels instead of instance labels (same scene, no episode)", "",
               f"- Nodes: {len(cat_graph.nodes)} with category labels vs {len(hydra.nodes)} with instance labels (oracle: {len(oracle.nodes)}).",
               f"- Category nodes covering two or more oracle instances (merged): {len(merged)}, covering {summary['category_labels']['oracle_instances_merged']} instances: "
               + ("; ".join(f"H:{h} ({cat_graph.nodes[h].category}, {' x '.join(f'{100 * v:.0f}' for v in world_dims(cat_graph.nodes[h]))} cm) <- " + ", ".join(f"O:{o}" for o in m) for h, m in merged.items()) or "none") + ".",
               f"- Oracle instances split over two or more category nodes: {len(split)}" + (": " + "; ".join(f"O:{o} -> " + ", ".join(f"H:{h}" for h in hs) for o, hs in split.items()) if split else "") + ".",
               f"- Clearance of the robot's start footprint from the balls in the base's height band: category map {100 * summary['category_labels']['start_clearance_m']:+.1f} cm, "
               f"instance map {100 * summary['start_clearance_instance_map_m']:+.1f} cm.", ""]
else:
    report += ["## 5. Category labels", "", "Not built (run map_scene.py --labels category).", ""]

(out_dir / "check.md").write_text("\n".join(report))
(out_dir / "check.json").write_text(json.dumps({"summary": summary, "points": point_rows, "match_rows": rows, "obb": obb_rows}, indent=1, default=float))
print("\n".join(report[:12]))
print("CHECK " + json.dumps(summary, default=float)[:3000])
base.close()
