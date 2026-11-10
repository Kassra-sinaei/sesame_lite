"""Pictures of the scene graph Hydra built for one scene (spec item 8), written to outputs/viz/<mission>_s<seed>/:

    MUJOCO_GL=egl .venv/bin/python -s scripts/scene_graph_viz.py --plan plans/nav_clutter.json --seed 100

  scene_graph.html   one self-contained interactive 3D page: Hydra's mesh coloured by node, every node's box as a
                     wireframe with its label on hover (id, category, size, tilt), every layer and edge Hydra produced
                     (objects, places, 2D places, room, building, the mapping camera's poses), toggles for mesh / boxes /
                     enlarged boxes / true boxes (the true boxes are the simulator's, hidden by default)
  scene_graph.pdf    left: the simulator render from a 3/4 view; middle: Hydra's mesh and boxes from the same camera;
                     right: top view with true boxes (grey), Hydra boxes (black), enlarged boxes (red dashed)
  obb_vs_raabb.pdf   the same scene mapped with stock RAABB boxes and with the fork's OBB boxes, and a per-node volume table
  turntable.mp4      10 s turntable of the 3D scene graph (mesh, boxes, places, room, edges)

The graph panels use only what Hydra produced (its dsg.json: mesh, nodes, boxes, edges; the enlarged boxes are computed
from Hydra's faces). Simulator ground truth appears only where named: the left render and the true boxes.
"""

import argparse
import colorsys
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import spark_dsg as dsg  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from shapely.geometry import MultiPoint  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402
from sesame_lite.mapping import MappingParams, build_map, map_dir  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.scene_graph import HydraSceneGraph, _raw_data, _raw_model  # noqa: E402
from sesame_lite.uncertainty import kappa  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"], "font.weight": "normal",
    "axes.titleweight": "normal", "axes.labelweight": "normal", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "figure.facecolor": "white",
})
SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(SIGNS[i] != SIGNS[j]) == 1]  # the 12 box edges
LAYER_NAMES = {"2": "objects", "2[97]": "mapping camera poses", "3": "places", "3[1]": "2D places", "4": "rooms", "5": "buildings"}


def node_colors(n: int) -> np.ndarray:
    return np.array([colorsys.hsv_to_rgb((0.61803 * k) % 1.0, 0.55, 0.92) for k in range(n)])


def corners(center, half, R):
    return np.asarray(center) + (SIGNS * np.asarray(half)) @ np.asarray(R).T


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(np.asarray(R)[2, :])), -1, 1))))


def footprint(center, half, R):
    return np.asarray(MultiPoint([tuple(p) for p in corners(center, half, R)[:, :2]]).convex_hull.exterior.coords)


class HydraMap:
    """What Hydra produced for one scene: mesh, object boxes, every node and edge (from its dsg.json)."""

    def __init__(self, path: Path, reader: HydraSceneGraph):
        self.g = dsg.DynamicSceneGraph.load(str(path / "dsg.json"))
        raw = np.asarray(self.g.mesh.get_vertices(), dtype=np.float64)
        self.V, self.F = raw[:3].T, np.asarray(self.g.mesh.get_faces(), dtype=np.int64).T
        self.reader = reader
        self.objects = []  # (id, category, center, half, R, vertex indices)
        for node in self.g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
            nid = str(node.id)
            if nid not in reader.nodes:
                continue
            n, bb = reader.nodes[nid], node.attributes.bounding_box
            self.objects.append((nid, n.category, np.asarray(bb.world_P_center, float), np.asarray(bb.dimensions, float) / 2, np.asarray(bb.world_R_center, float),
                                 np.asarray(list(node.attributes.mesh_connections), dtype=np.int64)))
        self.vertex_node = np.full(len(self.V), -1)
        for k, (*_, conn) in enumerate(self.objects):
            self.vertex_node[conn] = k
        self.colors = node_colors(len(self.objects))
        self.layers = {}  # layer key -> [(id, position)]
        for key in self.g.layer_keys:
            lid, part = (int(str(key).split("[")[0]), int(str(key).split("[")[1][:-1])) if "[" in str(key) else (int(str(key)), 0)
            layer = self.g.get_layer(lid, part) if part else self.g.get_layer(lid)
            self.layers[str(key)] = [(str(n.id), np.asarray(n.attributes.position, float)) for n in layer.nodes]
        pos = {nid: p for nodes in self.layers.values() for nid, p in nodes}
        self.edges = [(pos[str(dsg.NodeSymbol(e.source))], pos[str(dsg.NodeSymbol(e.target))]) for e in self.g.edges
                      if str(dsg.NodeSymbol(e.source)) in pos and str(dsg.NodeSymbol(e.target)) in pos]


# ------------------------------------------------------------------ MuJoCo rendering of the Hydra mesh


def hydra_model(hmap: HydraMap, wall_alpha: float) -> mujoco.MjModel:
    """A MuJoCo model holding only Hydra's mesh (one mesh per object node, one for the rest), for rendering."""
    assets, geoms = [], []
    groups = [(k, np.flatnonzero(hmap.vertex_node == k)) for k in range(len(hmap.objects))] + [(-1, np.flatnonzero(hmap.vertex_node == -1))]
    for k, verts in groups:
        keep = np.all(np.isin(hmap.F, verts), axis=1) if len(verts) else np.zeros(len(hmap.F), bool)
        tri = hmap.F[keep]
        if len(tri) == 0:
            continue
        used, inverse = np.unique(tri, return_inverse=True)
        inverse = inverse.reshape(-1, 3)
        both = np.vstack([inverse, inverse[:, ::-1]])  # both windings: Hydra's triangles face either way
        name = f"m{k + 1}"
        assets.append(f'<mesh name="{name}" vertex="{" ".join(f"{v:.4f}" for v in hmap.V[used].ravel())}" face="{" ".join(map(str, both.ravel()))}" inertia="shell"/>')
        rgb = hmap.colors[k] if k >= 0 else (0.62, 0.62, 0.62)
        alpha = wall_alpha if k >= 0 and hmap.objects[k][1] == "wall" else 1.0
        geoms.append(f'<geom type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} {alpha}"/>')
    xml = (f'<mujoco><visual><global offwidth="1600" offheight="1200"/><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>'
           f'<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{"".join(assets)}</asset><worldbody><light pos="0 0 6" dir="0 0 -1"/>{"".join(geoms)}</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).reshape(9), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_line(scene, a, b, rgba, width=2.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def free_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
    return cam


def render_hydra(model, hmap: HydraMap, cam, size, *, boxes=True, graph=False, enlarged=None) -> np.ndarray:
    model.vis.global_.fovy = 45.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # geom poses
    with mujoco.Renderer(model, size[1], size[0], max_geom=20000) as r:
        r.update_scene(data, camera=cam)
        if boxes:
            for k, (_, _, c, h, R, _) in enumerate(hmap.objects):
                add_geom(r.scene, mujoco.mjtGeom.mjGEOM_LINEBOX, h, c, R, [0.05, 0.05, 0.05, 1])
        if enlarged:
            for _, (c, h, R) in enlarged.items():
                add_geom(r.scene, mujoco.mjtGeom.mjGEOM_LINEBOX, h, c, R, [0.85, 0.15, 0.15, 1])
        if graph:
            colors = {"3": [0.10, 0.45, 0.85, 1], "3[1]": [0.15, 0.65, 0.35, 1], "4": [0.85, 0.35, 0.10, 1], "5": [0.55, 0.15, 0.65, 1], "2[97]": [0.4, 0.4, 0.4, 1]}
            for key, nodes in hmap.layers.items():
                if key in colors:
                    for _, p in nodes:
                        add_geom(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05 if key != "4" else 0.12, 0, 0], p, np.eye(3), colors[key])
            for a, b in hmap.edges:
                add_line(r.scene, a, b, [0.2, 0.2, 0.2, 0.6], 1.5)
        return r.render().copy()


def render_sim(rs_env, cam, size) -> np.ndarray:
    model, data = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    model.vis.global_.offwidth, model.vis.global_.offheight = max(model.vis.global_.offwidth, size[0]), max(model.vis.global_.offheight, size[1])
    fovy = float(model.vis.global_.fovy)
    model.vis.global_.fovy = 45.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "" for g in range(model.ngeom)]
    try:
        with mujoco.Renderer(model, size[1], size[0], max_geom=40000) as r:
            r.update_scene(data, camera=cam, scene_option=opt)
            for i in range(r.scene.ngeom):
                gg = r.scene.geoms[i]
                if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and "wall" in names[gg.objid]:
                    gg.rgba[3] = 0.15  # see through the walls, as in the Hydra panel
            return r.render().copy()
    finally:
        model.vis.global_.fovy = fovy


# ------------------------------------------------------------------ main

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
args = ap.parse_args()
mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False, graph_source="hydra")
env.reset(seed=args.seed, plan=mission)  # loads the cached maps of this scene
n_map = env.n_capture - 1
mdir = map_dir(mission["mission"], args.seed, n_map)
out = Path("outputs/viz") / f"{mission['mission']}_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)
hmap = HydraMap(mdir, env.base_graph)
policy = FilterPolicy()
conf_graph = env.base_graph.with_confidence(kappa=kappa(policy.delta, len(env.base_graph.nodes)), m_d=policy.disturbance_offset_m, m_r_voxels=policy.m_r_voxels)
enlarged = {nid: (n.confidence["center"], n.confidence["half_extents"], np.asarray(n.rotation)) for nid, n in conf_graph.nodes.items()}
true_boxes = {oid: n.aabb() for oid, n in env.oracle_graph.nodes.items()}
V = hmap.V

# ---------------- (a) scene_graph.html
import plotly.graph_objects as go  # noqa: E402

vc = np.where(hmap.vertex_node[:, None] >= 0, hmap.colors[np.maximum(hmap.vertex_node, 0)], 0.62)
traces = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=hmap.F[:, 0], j=hmap.F[:, 1], k=hmap.F[:, 2], vertexcolor=[f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})" for r, g, b in vc],
                    name="mesh", legendgroup="mesh", showlegend=True, hoverinfo="skip", flatshading=True, lighting=dict(ambient=0.6, diffuse=0.6))]


def box_lines(items, name, color, dash=None, visible=True):
    xs, ys, zs, txt = [], [], [], []
    for label, c, h, R in items:
        P = corners(c, h, R)
        for a, b in EDGES:
            xs += [P[a][0], P[b][0], None]
            ys += [P[a][1], P[b][1], None]
            zs += [P[a][2], P[b][2], None]
            txt += [label, label, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=3, dash=dash), name=name, text=txt, hoverinfo="text", visible=True if visible else "legendonly")


hover = [f"H:{nid}<br>{cat}<br>{2 * h[0]:.2f} x {2 * h[1]:.2f} x {2 * h[2]:.2f} m<br>tilt {tilt_deg(R):.1f} deg" for nid, cat, c, h, R, _ in hmap.objects]
traces.append(box_lines([(t, c, h, R) for t, (nid, cat, c, h, R, _) in zip(hover, hmap.objects)], "Hydra boxes", "black"))
traces.append(box_lines([(f"enlarged H:{nid}", *enlarged[nid]) for nid, *_ in hmap.objects], "enlarged boxes", "#C0392B", dash="dash", visible=False))
traces.append(box_lines([(f"true box O:{oid} (simulator)", (lo + hi) / 2, (hi - lo) / 2, np.eye(3)) for oid, (lo, hi) in true_boxes.items()], "true boxes (simulator)", "#888888", visible=False))
traces.append(go.Scatter3d(x=[c[0] for _, _, c, *_ in hmap.objects], y=[c[1] for _, _, c, *_ in hmap.objects], z=[c[2] for _, _, c, *_ in hmap.objects], mode="markers",
                           marker=dict(size=3, color="black"), text=hover, hoverinfo="text", name="object nodes"))
layer_style = {"3": ("#1F77B4", 3), "3[1]": ("#2CA02C", 3), "4": ("#D95F02", 8), "5": ("#7570B3", 10), "2[97]": ("#555555", 2)}
for key, nodes in hmap.layers.items():
    if key not in layer_style or not nodes:
        continue
    P = np.array([p for _, p in nodes])
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker=dict(size=layer_style[key][1], color=layer_style[key][0]),
                               text=[f"{nid} ({LAYER_NAMES.get(key, key)})" for nid, _ in nodes], hoverinfo="text", name=f"{LAYER_NAMES.get(key, key)} ({len(nodes)})"))
ex, ey, ez = [], [], []
for a, b in hmap.edges:
    ex += [a[0], b[0], None]
    ey += [a[1], b[1], None]
    ez += [a[2], b[2], None]
traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="rgba(60,60,60,0.5)", width=1.5), name=f"edges ({len(hmap.edges)})", hoverinfo="skip"))
fig = go.Figure(traces)
fig.update_layout(title=dict(text=f"Hydra scene graph: {mission['mission']}, seed {args.seed} (click the legend to show or hide a layer)", font=dict(family="Times New Roman")),
                  scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"), font=dict(family="Times New Roman"),
                  legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
fig.write_html(out / "scene_graph.html", include_plotlyjs=True, full_html=True)

# ---------------- (b) scene_graph.pdf
lo_room = np.min([lo for lo, _ in true_boxes.values()], axis=0)
hi_room = np.max([hi for _, hi in true_boxes.values()], axis=0)
center = (V.min(axis=0) + V.max(axis=0)) / 2  # the middle of Hydra's mesh
center[2] = 0.6
span = float(np.max(V.max(axis=0)[:2] - V.min(axis=0)[:2]))
cam = free_camera(center, 1.25 * span, -135.0, -40.0)
size = (1200, 900)
img_sim = render_sim(env.rs, cam, size)
model = hydra_model(hmap, wall_alpha=0.15)
img_hydra = render_hydra(model, hmap, cam, size, boxes=True)
fig = plt.figure(figsize=(7.16, 2.35), layout="constrained")
axs = fig.subplot_mosaic([["sim", "hydra", "top"]], width_ratios=[1.33, 1.33, 1.0])
for key, img, title in (("sim", img_sim, "(a) simulator, 3/4 view"), ("hydra", img_hydra, "(b) Hydra mesh and boxes, same camera")):
    axs[key].imshow(img)
    axs[key].axis("off")
    axs[key].set_title(title, loc="left")
ax = axs["top"]
for lo, hi in true_boxes.values():
    ax.add_patch(Polygon(footprint((lo + hi) / 2, (hi - lo) / 2, np.eye(3)), closed=True, facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.3, zorder=1))
for nid, _, c, h, R, _ in hmap.objects:
    ax.add_patch(Polygon(footprint(c, h, R), closed=True, fill=False, edgecolor="black", lw=0.4, zorder=3))
    ec, eh, eR = enlarged[nid]
    ax.add_patch(Polygon(footprint(ec, eh, eR), closed=True, fill=False, edgecolor="#C0392B", lw=0.45, ls=(0, (2.5, 1.5)), zorder=4))
ax.set_xlim(lo_room[0] - 0.2, hi_room[0] + 0.2)
ax.set_ylim(lo_room[1] - 0.2, hi_room[1] + 0.2)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("(c) top view", loc="left")
ax.legend(handles=[Patch(facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.3, label="true box"), Line2D([], [], color="black", lw=0.6, label="Hydra box"),
                   Line2D([], [], color="#C0392B", lw=0.6, ls=(0, (2.5, 1.5)), label="enlarged box")], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=6)
fig.savefig(out / "scene_graph.pdf")
plt.close(fig)

# ---------------- (d) turntable.mp4 (10 s, 30 fps)
frames = []
for k in range(300):
    cam_k = free_camera(center, 1.25 * span, -135.0 + 360.0 * k / 300, -35.0)
    frames.append(render_hydra(model, hmap, cam_k, (960, 720), boxes=True, graph=True))
iio.imwrite(out / "turntable.mp4", np.stack(frames), fps=30, codec="libx264", macro_block_size=8)

# ---------------- (c) obb_vs_raabb.pdf: the same scene mapped again with stock RAABB boxes
rdir = Path(str(mdir) + "_raabb")
if not (rdir / "faces.json").exists():
    build_map(env.rs, env.oracle_graph, env.oracle_lookup, rdir, seed=args.seed, params=replace(MappingParams(), bounding_box_type="RAABB"))
bp, Rb, _ = env.robot_pose()
rgraph = HydraSceneGraph(rdir, env.oracle_graph.labels, robot_pose=(bp, 0.0))
raabb = HydraMap(rdir, rgraph)
instance = {nid: env.base_graph.instance_of(nid) for nid, *_ in hmap.objects}
r_by_instance = {}
for nid, cat, c, h, R, conn in raabb.objects:
    inst = rgraph.instance_of(nid)
    if inst not in r_by_instance or len(conn) > len(r_by_instance[inst][5]):
        r_by_instance[inst] = (nid, cat, c, h, R, conn)
fig = plt.figure(figsize=(7.16, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55], hspace=0.28, wspace=0.08)
for col, (hm, title) in enumerate(((raabb, f"(a) stock RAABB boxes ({len(raabb.objects)} nodes)"), (hmap, f"(b) fork OBB boxes ({len(hmap.objects)} nodes)"))):
    ax = fig.add_subplot(gs[0, col])
    for lo, hi in true_boxes.values():
        ax.add_patch(Polygon(footprint((lo + hi) / 2, (hi - lo) / 2, np.eye(3)), closed=True, facecolor="#E3E3E3", edgecolor="none", zorder=1))
    for nid, _, c, h, R, _ in hm.objects:
        ax.add_patch(Polygon(footprint(c, h, R), closed=True, fill=False, edgecolor="black", lw=0.4, zorder=3))
    ax.set_xlim(lo_room[0] - 0.2, hi_room[0] + 0.2)
    ax.set_ylim(lo_room[1] - 0.2, hi_room[1] + 0.2)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.set_xlabel("x [m]")
    if col == 0:
        ax.set_ylabel("y [m]")
    else:
        ax.set_yticklabels([])
rows = []
for nid, cat, c, h, R, conn in sorted(hmap.objects, key=lambda o: int(o[0][2:-1])):
    v_obb = float(np.prod(2 * h))
    r = r_by_instance.get(instance[nid])
    v_raabb = float(np.prod(2 * r[3])) if r is not None else float("nan")
    contains = "-" if r is None else ("yes" if np.all(np.abs((raabb.V[r[5]] - r[2]) @ r[4]) <= r[3] + 1e-3) else "no")
    rows.append([f"H:{nid}", cat, f"{v_obb * 1000:.1f}", "-" if r is None else f"{v_raabb * 1000:.1f}", "-" if r is None else f"{v_obb / v_raabb:.2f}", f"{tilt_deg(R):.1f}", contains])
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
half = (len(rows) + 1) // 2
cols = ["node", "category", "OBB [L]", "RAABB [L]", "ratio", "tilt", "holds"]
for k, chunk in enumerate((rows[:half], rows[half:])):
    tbl = ax.table(cellText=chunk, colLabels=cols, cellLoc="left", bbox=[0.0 + 0.505 * k, 0.0, 0.495, 1.0], colWidths=[0.13, 0.27, 0.13, 0.14, 0.11, 0.1, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.2)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_linewidth(0.2)
        cell.set_edgecolor("#BBBBBB")
ax.set_title("(c) per node of the OBB map: box volume, volume of the stock map's RAABB node of the same instance, ratio OBB / RAABB,\n"
             "tilt of the OBB [deg], and whether the stock RAABB box holds its own vertices", loc="left")
fig.savefig(out / "obb_vs_raabb.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", *sorted(str(p) for p in out.iterdir()))
base.close()
