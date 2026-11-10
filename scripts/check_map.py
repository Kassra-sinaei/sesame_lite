"""Phase 1 checks of the Hydra map of one scene (spec 4.8), written next to the map: check.md, check.json, face_sigma.pdf.

    MUJOCO_GL=egl .venv/bin/python -s scripts/check_map.py --plan plans/nav_clutter.json --seed 100

The scene is rebuilt from the seed and the cached maps are loaded (nothing is mapped again). Ground truth is used
here, for evaluation only:
  1. faces.json lists every node with category, box and six faces
  2. match table: per oracle node, the Hydra node(s) of its instance, centre distance, extent difference, tilt, yaw
  3. coverage: per observed face of an instance with exactly one Hydra node, the error e = truth - b_hat (truth = the
     outermost point of the instance's visual geoms along the face normal) against kappa*sigma and kappa*sigma + m_r
  4. face_sigma.pdf: top view with oracle, Hydra and confidence boxes, and a histogram of kappa*sigma
  5. the category-label map of the same scene (if built with map_scene.py --labels category): merges, splits, and
     whether the robot's start footprint overlaps a box
  plus the oriented-box check: OBB volume <= RAABB volume of the same vertices, and the tilt of upright fixtures.
kappa uses delta split over every node of the map (the most conservative admitted count) unless --n-admitted is given.
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import spark_dsg as dsg  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from shapely.geometry import MultiPoint  # noqa: E402
from shapely.geometry import Polygon as ShapelyPolygon  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.mapping import map_dir  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.uncertainty import confidence_box, kappa  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "figure.facecolor": "white",
})

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.01, help="risk of the confidence boxes (spec 5.1)")
ap.add_argument("--m-d", type=float, default=0.025, help="ISSf disturbance margin d_bar^2 / (4 eps alpha) (spec 5.3)")
ap.add_argument("--n-admitted", type=int, default=None, help="admitted count for kappa (default: every node of the map)")
args = ap.parse_args()

mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False, graph_source="hydra")
env.reset(seed=args.seed, plan=mission)  # loads the cached maps; the last one is the scene the episode runs in
n_map = env.n_capture - 1
out_dir = map_dir(mission["mission"], args.seed, n_map)
hydra, oracle, bridge, labels = env.base_graph, env.oracle_graph, env.bridge, env.oracle_graph.labels  # estimated boxes (no confidence inflation)
oracle_name = {v: k for k, v in env.oracle_lookup.node_of.items()}
faces_file = json.loads((out_dir / "faces.json").read_text())
voxel = hydra.voxel_size_m
n_adm = args.n_admitted or len(hydra.nodes)
kap = kappa(args.delta, n_adm)
kap1 = kappa(args.delta, 1)
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


def footprint(node, half=None, center=None):
    frag = node.fragment()
    R = frag.rotation
    h = np.asarray(frag.half_extents if half is None else half)
    c = np.asarray(frag.center if center is None else center)
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = c + (signs * h) @ R.T
    return np.asarray(MultiPoint([tuple(p) for p in corners[:, :2]]).convex_hull.exterior.coords)


def min_area_rect(xy):
    """Smallest-area rectangle around 2-D points (monotone-chain hull, rotating edges), returns its area."""
    pts = np.unique(np.round(np.asarray(xy, dtype=np.float64), 9), axis=0)
    if len(pts) < 3:
        return 0.0
    hull = np.asarray(MultiPoint([tuple(p) for p in pts]).convex_hull.exterior.coords)[:-1]
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


# ------------------------------------------------------------------ 1. faces.json
complete = all(len(n["faces"]) == 6 and all(set(f) >= {"offset", "sigma", "observed"} for f in n["faces"]) for n in faces_file["nodes"])
src = faces_file["vertex_std_source_counts"]
summary["faces"] = {"nodes": len(faces_file["nodes"]), "every_node_has_six_faces": complete, "vertex_std_source_counts": src,
                    "max_box_reproduction_error_m": faces_file["max_box_reproduction_error_m"]}
report += [f"# Hydra map check: {mission['mission']}, seed {args.seed}, map n = {n_map}", "",
           f"Map: `{out_dir.relative_to(Path.cwd()) if out_dir.is_relative_to(Path.cwd()) else out_dir}` (oracle snapshot `{oracle.snapshot_id}`).", "",
           "## 1. faces.json", "",
           f"- {len(faces_file['nodes'])} object nodes, each with six faces (offset, sigma, observed): {'yes' if complete else 'NO'}.",
           f"- b_hat (max over the node's vertices along each face normal) reproduces Hydra's own box to {1000 * faces_file['max_box_reproduction_error_m']:.4f} mm.",
           f"- Vertex std from the grid edge (eq. 2): {src['grid_edge']} supporting vertices; from the bound sigma0^2/min(w) (vertex not on a sign-changing grid edge of the final TSDF): {src['bound']}; none: {src['none']}.", ""]

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

# ------------------------------------------------------------------ 3. coverage
FACE = ["+x", "-x", "+y", "-y", "+z", "-z"]
records, unobserved = [], []
single = {o: v[0] for o, v in bridge.hydra_of.items() if len(v) == 1}
faces_by_node = {n["node_id"]: n for n in faces_file["nodes"]}
for oracle_id, hydra_id in single.items():
    for k, f in enumerate(faces_by_node[hydra_id]["faces"]):
        rec = {"oracle_id": oracle_id, "hydra_id": hydra_id, "face": k, "category": hydra.nodes[hydra_id].category,
               "own_near_truth": f["own_near_truth"], "best_view_angle_deg": f["best_view_angle_deg"]}
        if not f["observed"] or f["truth"] is None:
            unobserved.append(rec)
            continue
        rec.update(e_m=f["truth"] - f["offset"], sigma_m=f["sigma"])
        records.append(rec)
e = np.array([r["e_m"] for r in records])
sig = np.array([r["sigma_m"] for r in records])
z = e / sig
# m_r sweep (spec item 5); "nearly none" fixed before the run: at most NEARLY_NONE of the observed faces
NEARLY_NONE = 0.02
M_R_FACTORS = (0.25, 0.5, 1.0)
sweep = {f: int(np.sum(e > kap * sig + f * voxel)) for f in M_R_FACTORS}
passing = [f for f in M_R_FACTORS if sweep[f] <= NEARLY_NONE * len(records)]
chosen = passing[0] if passing else None
m_r = (chosen if chosen is not None else M_R_FACTORS[-1]) * voxel
n_over = int(np.sum(e > kap * sig))
reasons = {"no own vertex near the true supporting point": sum(not r["own_near_truth"] for r in unobserved),
           "never viewed within 60 deg of the normal": sum(r["own_near_truth"] and not (r["best_view_angle_deg"] is not None and r["best_view_angle_deg"] <= 60) for r in unobserved)}
summary["coverage"] = {"kappa": kap, "n_admitted_for_kappa": n_adm, "kappa_single_object": kap1, "observed_faces": len(records), "unobserved_faces": len(unobserved),
                       "unobserved_reasons": reasons, "e_mean_m": float(e.mean()), "e_std_m": float(e.std()),
                       "e_over_sigma_quantiles": dict(zip(["1%", "10%", "50%", "90%", "99%", "max"], np.percentile(z, [1, 10, 50, 90, 99, 100]).round(2).tolist())),
                       "faces_e_gt_kappa_sigma": n_over, "m_r_sweep": {f"{f} voxel": n for f, n in sweep.items()}, "nearly_none_fraction": NEARLY_NONE,
                       "chosen_m_r_voxels": chosen, "m_r_m": m_r, "passed": chosen is not None}
report += ["## 3. Coverage of the confidence faces", "",
           f"Faces of instances with exactly one Hydra node ({len(single)} instances). Truth = the outermost point of the instance's visual geoms along the face normal. "
           "A face is **observed** only if (a) this Hydra node has mesh vertices within 1.5 voxels of the true supporting point, and (b) a mapping frame saw the "
           "face's supporting vertex within 60 deg of the face normal. Unobserved faces get no claim. e = truth - b_hat (positive: the true surface lies outside Hydra's box).", "",
           f"- kappa = {kap:.2f} (delta = {args.delta} over {n_adm} nodes x 6 faces); single-object kappa = {kap1:.2f}.",
           f"- Observed faces: {len(records)}. Not observed (no claim): {len(unobserved)}: " + "; ".join(f"{k}: {v}" for k, v in reasons.items()) + ".",
           f"- e: mean {100 * e.mean():+.2f} cm, std {100 * e.std():.2f} cm, min {100 * e.min():+.2f} cm, max {100 * e.max():+.2f} cm.",
           f"- sigma: median {1000 * np.median(sig):.2f} mm, 90 % {1000 * np.percentile(sig, 90):.2f} mm, max {1000 * sig.max():.2f} mm; kappa*sigma median {1000 * kap * np.median(sig):.2f} mm.",
           "- e / sigma quantiles: " + ", ".join(f"{k} {v:+.2f}" for k, v in summary["coverage"]["e_over_sigma_quantiles"].items()) + ".",
           f"- Faces with e > kappa*sigma: {n_over} of {len(records)}.", "",
           f"m_r sweep (\"nearly none\" fixed before the run: at most {100 * NEARLY_NONE:.0f} % of the observed faces, i.e. {int(NEARLY_NONE * len(records))}):", "",
           "| m_r | observed faces with e > kappa*sigma + m_r |", "|---|---|"] + [f"| {f} voxel = {100 * f * voxel:.2f} cm | {n} |" for f, n in sweep.items()] + ["",
           f"**Chosen m_r: {chosen} voxel = {100 * m_r:.2f} cm.**" if chosen is not None else "**No m_r in the sweep leaves nearly none: the coverage check fails.**", ""]
over = sorted([r for r in records if r["e_m"] > kap * r["sigma_m"] + m_r], key=lambda r: -(r["e_m"] - kap * r["sigma_m"]))
if over:
    report += [f"Every observed face with e > kappa*sigma + m_r (m_r = {100 * m_r:.2f} cm):", "",
               "| oracle | Hydra | category | face | e [cm] | sigma [mm] | best view [deg] | e - kappa*sigma - m_r [cm] |", "|---|---|---|---|---|---|---|---|"]
    for r in over:
        report.append(f"| O:{r['oracle_id']} | H:{r['hydra_id']} | {r['category']} | {FACE[r['face']]} | {100 * r['e_m']:+.2f} | {1000 * r['sigma_m']:.2f} | {r['best_view_angle_deg']} | {100 * (r['e_m'] - kap * r['sigma_m'] - m_r):+.2f} |")
    report.append("")
report += ["Faces not observed (no claim):", "", "| Hydra | category | face | own vertex near truth | best view [deg] |", "|---|---|---|---|---|"]
report += [f"| H:{r['hydra_id']} | {r['category']} | {FACE[r['face']]} | {'yes' if r['own_near_truth'] else 'no'} | {r['best_view_angle_deg'] if r['best_view_angle_deg'] is not None else 'never'} |" for r in unobserved] + [""]
summary["coverage"]["exceedances"] = over
summary["coverage"]["unobserved"] = unobserved

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

# ------------------------------------------------------------------ 4. figure
fig = plt.figure(figsize=(7.16, 3.3), layout="constrained")
axs = fig.subplot_mosaic([["map", "hist"]], width_ratios=[1.6, 1])
ax = axs["map"]
for onode in oracle.nodes.values():
    ax.add_patch(Polygon(footprint(onode), closed=True, facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.4, zorder=1))
for hnode in hydra.nodes.values():
    ax.add_patch(Polygon(footprint(hnode), closed=True, fill=False, edgecolor="black", lw=0.5, zorder=3))
    c, h, _ = confidence_box(hnode.faces, np.asarray(hnode.rotation), kappa_value=kap, m_r=m_r, m_d=args.m_d)
    ax.add_patch(Polygon(footprint(hnode, half=h, center=c), closed=True, fill=False, edgecolor="#C0392B", lw=0.5, ls=(0, (2.5, 1.5)), zorder=4))
bp = env.robot_pose()[0]
ax.plot(bp[0], bp[1], marker="o", ms=4, color="#1F4E9A", zorder=6)
lo = np.min([n.aabb()[0][:2] for n in oracle.nodes.values()], axis=0) - 0.2
hi = np.max([n.aabb()[1][:2] for n in oracle.nodes.values()], axis=0) + 0.2
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(handles=[Patch(facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.4, label="oracle box"), Line2D([], [], color="black", lw=0.6, label="Hydra box"),
                   Line2D([], [], color="#C0392B", lw=0.6, ls=(0, (2.5, 1.5)), label=r"confidence box ($\kappa\sigma + m_r + m_d$)"),
                   Line2D([], [], marker="o", ls="", color="#1F4E9A", ms=4, label="robot start")], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
ax.set_title("(a) top view", loc="left")
ax = axs["hist"]
all_sigma = np.array([f["sigma"] for n in faces_file["nodes"] for f in n["faces"] if f["observed"] and f["sigma"] is not None])
ax.hist(100 * kap * all_sigma, bins=np.logspace(-2, 2.2, 43), color="#4D4D4D", lw=0)
ax.set_xscale("log")
ax.set_xlabel(r"$\kappa\sigma$ [cm]"); ax.set_ylabel("observed faces")
ax.set_title(rf"(b) $\kappa\sigma$ over observed faces ($\kappa$ = {kap:.2f})", loc="left")
fig.savefig(out_dir / "face_sigma.pdf")
plt.close(fig)
report += ["## 4. Figure", "", f"`face_sigma.pdf`: (a) top view with oracle boxes (grey), Hydra boxes (black) and confidence boxes (red dashed; kappa = {kap:.2f}, m_r = {100 * m_r:.2f} cm, m_d = {100 * args.m_d:.1f} cm); (b) histogram of kappa*sigma over the {len(all_sigma)} observed faces of all nodes.", ""]

# ------------------------------------------------------------------ 5. category labels
cat_dir = map_dir(mission["mission"], args.seed, n_map, labels="category")


def start_overlaps(graph_nodes):
    """Boxes (in the base's height band) that the robot's start footprint overlaps."""
    st = env.robot_state()
    base = ShapelyPolygon(env.filter.base_vertices_world(st))
    hits = []
    for nid, n in graph_nodes.items():
        lo3, hi3 = n.aabb()
        if hi3[2] < env.z_band[0] or lo3[2] > env.z_band[1]:
            continue
        if base.intersects(ShapelyPolygon(footprint(n))):
            hits.append((nid, n.category))
    return hits


summary["start_overlap_instance_map"] = start_overlaps(hydra.nodes)
if (cat_dir / "faces.json").exists():
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
                                  "oracle_instances_merged": sum(len(v) for v in merged.values()), "start_overlap": start_overlaps(cat_graph.nodes)}
    report += ["## 5. Category labels instead of instance labels (same scene, no episode)", "",
               f"- Nodes: {len(cat_graph.nodes)} with category labels vs {len(hydra.nodes)} with instance labels (oracle: {len(oracle.nodes)}).",
               f"- Category nodes covering two or more oracle instances (merged): {len(merged)}, covering {summary['category_labels']['oracle_instances_merged']} instances: "
               + ("; ".join(f"H:{h} ({cat_graph.nodes[h].category}, {' x '.join(f'{100 * v:.0f}' for v in world_dims(cat_graph.nodes[h]))} cm) <- " + ", ".join(f"O:{o}" for o in m) for h, m in merged.items()) or "none") + ".",
               f"- Oracle instances split over two or more category nodes: {len(split)}" + (": " + "; ".join(f"O:{o} -> " + ", ".join(f"H:{h}" for h in hs) for o, hs in split.items()) if split else "") + ".",
               "- Robot start footprint overlaps a box in the base's height band: category map " + (", ".join(f"H:{h} ({c})" for h, c in summary["category_labels"]["start_overlap"]) or "none")
               + "; instance map " + (", ".join(f"H:{h} ({c})" for h, c in summary["start_overlap_instance_map"]) or "none") + ".", ""]
else:
    report += ["## 5. Category labels", "", "Not built (run map_scene.py --labels category).", ""]

(out_dir / "check.md").write_text("\n".join(report))
(out_dir / "check.json").write_text(json.dumps({"summary": summary, "match_rows": rows, "obb": obb_rows}, indent=1, default=float))
print("\n".join(report[:12]))
print("CHECK " + json.dumps({k: v for k, v in summary.items() if k != "coverage"} | {"coverage": {k: v for k, v in summary["coverage"].items() if k != "exceedances"}}, default=float)[:3000])
base.close()
