"""Pictures of the map Hydra built for one scene (spec v4, section 12, items 2 to 5), in outputs/viz/<mission>_s<seed>/:

    MUJOCO_GL=egl .venv/bin/python -s scripts/scene_graph_viz.py --plan plans/nav_clutter.json --seed 100

  scene_graph.pdf    three panels from the same camera: (a) the simulator; (b) Hydra's mesh coloured by object node;
                     (c) the objects' points coloured by their std s_m (eq. 2)
  std_trend.pdf      s_m against the mean distance of the frames that saw the point, and against how many frames saw it
  scene_graph.html   interactive 3D page: the points drawn as balls of radius rho_m (eq. 4; delta split over every node,
                     as planning uses them), capped points (s_m > s_max) in another colour, Hydra's mesh, boxes, every
                     layer and edge it produced, and the simulator's true boxes (hidden at first); the legend toggles each
  barrier_slice.pdf  a horizontal slice through one object: level curves of the smooth barrier psi_tau (eq. 5) against
                     the exact distance to the union of its balls, min_m g_m

Only what Hydra produced (dsg.json, points.npz) is drawn, except where a panel or trace says "simulator".
"""

import argparse
import colorsys
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import spark_dsg as dsg  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy  # noqa: E402
from sesame_lite.mapping import map_dir  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.scene_graph import HydraSceneGraph, _raw_data, _raw_model  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"], "font.weight": "normal",
    "axes.titleweight": "normal", "axes.labelweight": "normal", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "savefig.dpi": 600, "figure.facecolor": "white",
})
SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(SIGNS[i] != SIGNS[j]) == 1]  # the 12 box edges
LAYER_NAMES = {"2": "objects", "2[97]": "mapping camera poses", "3": "places", "3[1]": "2D places", "4": "rooms", "5": "buildings"}
STD_NORM = LogNorm(vmin=5e-4, vmax=2e-2)  # colour scale of s_m [m] in the pictures


def node_colors(n: int) -> np.ndarray:
    return np.array([colorsys.hsv_to_rgb((0.61803 * k) % 1.0, 0.55, 0.92) for k in range(n)])


def corners(center, half, R):
    return np.asarray(center) + (SIGNS * np.asarray(half)) @ np.asarray(R).T


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(np.asarray(R)[2, :])), -1, 1))))


class HydraMap:
    """What Hydra produced for one scene: mesh, object nodes with boxes and points, every node and edge (dsg.json)."""

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


# ------------------------------------------------------------------ MuJoCo rendering of the Hydra mesh and points


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
        rgb = hmap.colors[k] if k >= 0 else (0.62, 0.62, 0.62)
        alpha = wall_alpha if k >= 0 and hmap.objects[k][1] == "wall" else 1.0
        if len(used) < 4:  # a MuJoCo mesh needs 4 vertices: a node of one triangle is drawn as dots at its vertices
            geoms += [f'<geom type="sphere" size="0.012" pos="{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" contype="0" conaffinity="0" group="1" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} {alpha}"/>' for v in hmap.V[used]]
            continue
        both = np.vstack([inverse, inverse[:, ::-1]])  # both windings: Hydra's triangles face either way
        name = f"m{k + 1}"
        assets.append(f'<mesh name="{name}" vertex="{" ".join(f"{v:.4f}" for v in hmap.V[used].ravel())}" face="{" ".join(map(str, both.ravel()))}" inertia="shell"/>')
        geoms.append(f'<geom type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} {alpha}"/>')
    xml = (f'<mujoco><visual><global offwidth="1600" offheight="1200"/><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>'
           f'<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>{"".join(assets)}</asset><worldbody><light pos="0 0 6" dir="0 0 -1"/>{"".join(geoms)}</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def empty_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="1600" offheight="1200"/><headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>'
                                          '<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/></asset><worldbody/></mujoco>')


def add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"render scene full ({scene.maxgeom} geoms)")
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).reshape(9), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def free_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
    return cam


def render(model, cam, size, points=None) -> np.ndarray:
    """Render a model; ``points`` = (centres, radius, rgba per point) are added as spheres."""
    model.vis.global_.fovy = 45.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    n_extra = 0 if points is None else len(points[0])
    with mujoco.Renderer(model, size[1], size[0], max_geom=n_extra + 10000) as r:
        r.update_scene(data, camera=cam)
        if points is not None:
            for p, rgba in zip(points[0], points[2]):
                add_geom(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [points[1], 0, 0], p, np.eye(3), rgba)
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
                    gg.rgba[3] = 0.06  # see through the walls (their brick texture needs a lower alpha than the flat Hydra panels)
            return r.render().copy()
    finally:
        model.vis.global_.fovy = fovy


# ------------------------------------------------------------------ main

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--slice-category", default="flour bag", help="barrier_slice.pdf: the object (by category) to cut through")
args = ap.parse_args()
mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False)
env.reset(seed=args.seed, plan=mission)  # loads the cached maps of this scene
mdir = map_dir(mission["mission"], args.seed, env.n_capture - 1)
out = Path("outputs/viz") / f"{mission['mission']}_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)
graph, balls, policy = env.graph, env.plan_balls, FilterPolicy()
hmap = HydraMap(mdir, graph)
pts = np.load(mdir / "points.npz")
P, S, n_views, view_dist = pts["points"], pts["std"], pts["n_views"], pts["view_distance"]
point_node = np.repeat(np.arange(len(pts["node_ids"])), np.diff(pts["offsets"]))
is_wall = np.array([graph.nodes[str(nid)].category == "wall" for nid in pts["node_ids"]])[point_node]
true_boxes = {oid: n.aabb() for oid, n in env.oracle_graph.nodes.items()}
V = hmap.V

# ---------------- item 2: scene_graph.pdf
center = (V.min(axis=0) + V.max(axis=0)) / 2  # the middle of Hydra's mesh
center[2] = 0.6
span = float(np.max(V.max(axis=0)[:2] - V.min(axis=0)[:2]))
cam = free_camera(center, 1.25 * span, -135.0, -40.0)
size = (1200, 900)
img_sim = render_sim(env.rs, cam, size)
img_mesh = render(hydra_model(hmap, wall_alpha=0.15), cam, size)
cmap = plt.get_cmap("viridis")
rgba = cmap(STD_NORM(np.clip(S, STD_NORM.vmin, STD_NORM.vmax)))
rgba[is_wall, 3] = 0.15
img_pts = render(empty_model(), cam, size, points=(P, 0.012, rgba))
fig = plt.figure(figsize=(7.16, 2.2), layout="constrained")
axs = fig.subplot_mosaic([["sim", "mesh", "pts", "cb"]], width_ratios=[1.0, 1.0, 1.0, 0.04])
for key, img, title in (("sim", img_sim, "(a) simulator"), ("mesh", img_mesh, "(b) Hydra mesh, coloured by object node"), ("pts", img_pts, r"(c) points, coloured by their std $s_m$")):
    axs[key].imshow(img)
    axs[key].axis("off")
    axs[key].set_title(title, loc="left")
cb = fig.colorbar(plt.cm.ScalarMappable(norm=STD_NORM, cmap=cmap), cax=axs["cb"])
cb.set_label(r"$s_m$ [m]")
fig.savefig(out / "scene_graph.pdf")
plt.close(fig)

# ---------------- item 3: std_trend.pdf
seen = n_views > 0
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4), layout="constrained", sharey=True)
for ax, x, xlabel, bins in ((axs[0], view_dist, "mean distance of the frames that saw the point [m]", np.linspace(0.3, 4.0, 15)),
                            (axs[1], n_views.astype(float), "frames that saw the point", np.unique(np.round(np.logspace(0, np.log10(max(n_views.max(), 2)), 15))))):
    ax.scatter(x[seen], S[seen], s=1.0, lw=0, color="#4D4D4D", alpha=0.25, rasterized=True)
    k = np.digitize(x[seen], bins)
    mids = [(np.median(x[seen][k == i]), np.median(S[seen][k == i])) for i in range(1, len(bins)) if np.sum(k == i) >= 20]
    ax.plot(*np.array(mids).T, color="#C0392B", lw=1.2, label="median per bin")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper right", frameon=False)
axs[1].set_xscale("log")
axs[0].set_ylabel(r"$s_m$ [m]")
axs[0].set_title(f"(a) against viewing distance ({int(seen.sum())} of {len(S)} points)", loc="left")
axs[1].set_title("(b) against the number of frames that saw the point", loc="left")
fig.savefig(out / "std_trend.pdf")
plt.close(fig)

# ---------------- item 4: scene_graph.html
import plotly.graph_objects as go  # noqa: E402


def ball_mesh(centres, radii):
    """Each ball as an octahedron with the ball's radius (vertex distance = rho_m), for plotly's Mesh3d."""
    U = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    T = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    verts = (centres[:, None, :] + radii[:, None, None] * U[None]).reshape(-1, 3)
    tri = (T[None] + 6 * np.arange(len(centres))[:, None, None]).reshape(-1, 3)
    return verts, tri


def box_lines(items, name, color, dash=None, visible=True):
    xs, ys, zs, txt = [], [], [], []
    for label, c, h, R in items:
        Q = corners(c, h, R)
        for a, b in EDGES:
            xs += [Q[a][0], Q[b][0], None]
            ys += [Q[a][1], Q[b][1], None]
            zs += [Q[a][2], Q[b][2], None]
            txt += [label, label, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=3, dash=dash), name=name, text=txt, hoverinfo="text", visible=True if visible else "legendonly")


vc = np.where(hmap.vertex_node[:, None] >= 0, hmap.colors[np.maximum(hmap.vertex_node, 0)], 0.62)
traces = [go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=hmap.F[:, 0], j=hmap.F[:, 1], k=hmap.F[:, 2], vertexcolor=[f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})" for r, g, b in vc],
                    name="mesh (by object)", hoverinfo="skip", flatshading=True, lighting=dict(ambient=0.6, diffuse=0.6), visible="legendonly")]
B = {k: np.concatenate([getattr(b, k) for b in balls.values()]) for k in ("points", "rho", "capped")}
kappa_of = np.concatenate([np.full(len(b.points), b.kappa) for b in balls.values()])
for capped, name, kw in ((False, r"balls (rho_m = kappa s_m + m_r + m_d)", dict(intensity=None, colorscale="Viridis")), (True, "capped balls (s_m > s_max)", dict(color="#D95F02"))):
    sel = B["capped"] == capped
    verts, tri = ball_mesh(B["points"][sel], B["rho"][sel])
    extra = dict(intensity=np.repeat(100 * B["rho"][sel], 6), colorscale="Viridis", colorbar=dict(title="rho [cm]", x=1.0, len=0.5)) if not capped else dict(color=kw["color"])
    traces.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], name=f"{name} ({int(sel.sum())})", showlegend=True,
                            flatshading=True, hoverinfo="skip", opacity=1.0, **extra))
hover = [f"H:{nid}<br>{cat}<br>{2 * h[0]:.2f} x {2 * h[1]:.2f} x {2 * h[2]:.2f} m<br>tilt {tilt_deg(R):.1f} deg<br>{len(balls[nid].points)} points, "
         f"kappa {balls[nid].kappa:.2f}, s_max {1000 * balls[nid].s_max:.1f} mm, {int(balls[nid].capped.sum())} capped" for nid, cat, c, h, R, _ in hmap.objects]
traces.append(box_lines([(t, c, h, R) for t, (nid, cat, c, h, R, _) in zip(hover, hmap.objects)], "Hydra boxes", "black", visible=False))
traces.append(box_lines([(f"true box O:{oid} (simulator)", (lo + hi) / 2, (hi - lo) / 2, np.eye(3)) for oid, (lo, hi) in true_boxes.items()], "true boxes (simulator)", "#888888", visible=False))
traces.append(go.Scatter3d(x=[c[0] for _, _, c, *_ in hmap.objects], y=[c[1] for _, _, c, *_ in hmap.objects], z=[c[2] for _, _, c, *_ in hmap.objects], mode="markers",
                           marker=dict(size=3, color="black"), text=hover, hoverinfo="text", name="object nodes"))
layer_style = {"3": ("#1F77B4", 3), "3[1]": ("#2CA02C", 3), "4": ("#D95F02", 8), "5": ("#7570B3", 10), "2[97]": ("#555555", 2)}
for key, nodes in hmap.layers.items():
    if key not in layer_style or not nodes:
        continue
    Q = np.array([p for _, p in nodes])
    traces.append(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="markers", marker=dict(size=layer_style[key][1], color=layer_style[key][0]),
                               text=[f"{nid} ({LAYER_NAMES.get(key, key)})" for nid, _ in nodes], hoverinfo="text", name=f"{LAYER_NAMES.get(key, key)} ({len(nodes)})"))
ex, ey, ez = [], [], []
for a, b in hmap.edges:
    ex += [a[0], b[0], None]
    ey += [a[1], b[1], None]
    ez += [a[2], b[2], None]
traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="rgba(60,60,60,0.5)", width=1.5), name=f"edges ({len(hmap.edges)})", hoverinfo="skip"))
fig = go.Figure(traces)
fig.update_layout(title=dict(text=f"Hydra map with confidence balls: {mission['mission']}, seed {args.seed} (delta split over all {len(balls)} nodes; m_r {100 * graph.m_r:.2f} cm, "
                                  f"m_d {100 * policy.m_d:.1f} cm). Click the legend to show or hide a layer.", font=dict(family="Times New Roman", size=14)),
                  scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"), font=dict(family="Times New Roman"),
                  legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
fig.write_html(out / "scene_graph.html", include_plotlyjs=True, full_html=True)

# ---------------- item 5: barrier_slice.pdf
nid = next(n for n, o in graph.nodes.items() if o.category == args.slice_category)
b = balls[nid]
z0 = float(np.median(b.points[:, 2]))
lo, hi = b.points[:, :2].min(axis=0) - 0.35, b.points[:, :2].max(axis=0) + 0.35
nx = 400
X, Y = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], int(nx * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
with torch.no_grad():
    x = torch.tensor(np.c_[X.ravel(), Y.ravel(), np.full(X.size, z0)], device="cuda", dtype=torch.float64)
    p = torch.tensor(b.points, device="cuda", dtype=torch.float64)
    rho = torch.tensor(b.rho, device="cuda", dtype=torch.float64)
    g = torch.cdist(x, p) - rho  # g_m(x)
    psi = (-policy.tau_m * torch.logsumexp(-g / policy.tau_m, dim=1)).cpu().numpy().reshape(X.shape)
    exact = g.min(dim=1).values.cpu().numpy().reshape(X.shape)
gap = exact - psi  # Lemma 2: 0 <= gap <= tau ln |P_j|
levels = [0.0, 0.05, 0.1, 0.2]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
ax.contour(X, Y, exact, levels=levels, colors="#9A9A9A", linewidths=2.2)
cs = ax.contour(X, Y, psi, levels=levels, colors="black", linewidths=0.7)
ax.clabel(cs, fmt=lambda v: f"{100 * v:.0f} cm", fontsize=6)
near = np.abs(b.points[:, 2] - z0) <= b.rho
ax.scatter(b.points[near, 0], b.points[near, 1], s=2, color="#C0392B", lw=0, zorder=5)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"(a) H:{nid} ({args.slice_category}), slice z = {z0:.2f} m", loc="left")
ax.legend(handles=[Line2D([], [], color="black", lw=0.7, label=r"$\psi_\tau$ (eq. 5), used"), Line2D([], [], color="#9A9A9A", lw=2.2, label=r"exact: $\min_m g_m$"),
                   Line2D([], [], marker="o", ls="", ms=2, color="#C0392B", label="points whose ball meets the slice")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
ax = axs[1]
ax.hist(1000 * gap.ravel(), bins=60, color="#4D4D4D", lw=0)
ax.axvline(1000 * policy.tau_m * np.log(len(b.points)), color="#C0392B", lw=0.8, ls="--", label=rf"$\tau \ln |P_j|$ = {1000 * policy.tau_m * np.log(len(b.points)):.1f} mm")
ax.set_xlabel(r"$\min_m g_m - \psi_\tau$ [mm] over the slice")
ax.set_ylabel("grid cells")
ax.legend(frameon=False)
ax.set_title(f"(b) the gap: min {1000 * gap.min():.3f} mm, max {1000 * gap.max():.2f} mm", loc="left")
fig.savefig(out / "barrier_slice.pdf")
plt.close(fig)
print("SLICE", nid, args.slice_category, "points", len(b.points), "gap min/max [mm]", 1000 * gap.min(), 1000 * gap.max())
print("wrote", *sorted(str(q) for q in out.iterdir()))
base.close()
