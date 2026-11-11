"""Top view of the start pose: how much room the base has, without running an episode.

    MUJOCO_GL=egl .venv/bin/python -s scripts/start_top_view.py --plan plans/close_fridge.json --seed 100

At the base's height band (up to the top of the base's spheres): the kitchen's collision geometry (simulator, grey),
the balls (radius rho_m) of the first segment's row objects (red) and of its exempt objects (blue outline), and the
robot's spheres in the band coloured by their barrier value h (red at h <= 0). One panel per filter setting: the PDF's
values and the tuned ones. Writes outputs/viz/<mission>_s<seed>/start_top_view.pdf.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.filter import FilterPolicy, SphereFilter  # noqa: E402
from sesame_lite.planner import load_plan  # noqa: E402
from sesame_lite.scene_graph import _collision_geoms, _raw_data, _raw_model  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"], "font.weight": "normal", "axes.titleweight": "normal",
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "savefig.dpi": 600})
SETTINGS = {"(a) PDF values: alpha 1, d_bar 0.05, tau 2 mm": dict(alpha=1.0, d_bar=0.05, tau_m=0.002), "(b) tuned: alpha 2, d_bar 0.03, tau 0.5 mm": {}}

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--margin", type=float, default=0.9, help="half-width of the view around the base [m]")
args = ap.parse_args()
plan = load_plan(args.plan)
base = SesameRoboCasaEnv(plan["task"], env_kwargs=dict(plan.get("env_kwargs", {})), episode_length=plan.get("horizon"))
env = SafetyFilterEnv(base, filter=False)
env.reset(seed=args.seed, plan=plan)
model, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
mujoco.mj_forward(model, data)
q = env.robot.q(data)
centres, radii = env.robot.spheres(q)
base_spheres = np.array(["mobilebase0" in link for link in env.robot.links])
band_top = float(np.max(centres[base_spheres, 2] + radii[base_spheres]))  # the top of the base's spheres
in_band = centres[:, 2] - radii <= band_top
bp = env.robot_pose()[0]
lo_xy, hi_xy = bp[:2] - args.margin, bp[:2] + args.margin

# the kitchen's collision geometry in the band (simulator)
geoms = []
for oid, bodies in env.oracle_lookup.bodies.items():
    for g in _collision_geoms(model, bodies):
        R, p, s, t = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g], model.geom_size[g], model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s[:3] @ R.T + p
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            v = np.vstack([np.c_[s[0] * np.cos(a), s[0] * np.sin(a), np.full(24, z)] for z in (-s[1], s[1])]) @ R.T + p
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]] @ R.T + p
        else:
            continue
        if v[:, 2].min() > band_top or np.any(v[:, :2].max(axis=0) < lo_xy) or np.any(v[:, :2].min(axis=0) > hi_xy):
            continue
        try:
            hull = v[:, :2][ConvexHull(v[:, :2]).vertices]
        except QhullError:  # a footprint with no area (a plate seen edge-on): drawn as its bounding segment
            hull = v[:, :2][[np.argmin(v[:, 0]), np.argmax(v[:, 0])]]
        geoms.append(hull)

seg = env.segments[0]
fig, axs = plt.subplots(1, len(SETTINGS), figsize=(7.16, 4.0), layout="constrained")
for ax, (title, overrides) in zip(axs, SETTINGS.items()):
    policy = replace(FilterPolicy(), **overrides)
    n_rows = len(env.row_ids)
    balls = env.graph.balls(n_admitted=n_rows, delta=policy.delta, m_d=policy.m_d)
    flt = SphereFilter(env.robot, policy)
    flt.set_objects({nid: balls[nid] for nid in env.row_ids})
    bar = flt.barrier(q)
    h = bar.h.min(axis=1)
    ax.add_collection(PatchCollection([Polygon(g, closed=True) for g in geoms], facecolor="#CFCFCF", edgecolor="#7F7F7F", lw=0.3, zorder=1))
    for nid, b in balls.items():
        if nid not in env.row_ids and nid not in seg.exempt_ids:
            continue
        keep = (b.points[:, 2] - b.rho <= band_top) & np.all((b.points[:, :2] >= lo_xy - 0.2) & (b.points[:, :2] <= hi_xy + 0.2), axis=1)
        exempt = nid in seg.exempt_ids
        ax.add_collection(PatchCollection([Circle(p[:2], r) for p, r in zip(b.points[keep], b.rho[keep])],
                                          facecolor="none" if exempt else "#E07060", edgecolor="#1F4E9A" if exempt else "none", lw=0.3, alpha=0.6 if exempt else 0.25, zorder=2))
    for c, r, hh, ok in zip(centres, radii, h, in_band):
        if ok:
            ax.add_patch(Circle(c[:2], r, facecolor="#C0392B" if hh <= 0 else "#2E8B57", edgecolor="black", lw=0.3, alpha=0.55, zorder=3))
    i = int(np.argmin(h))
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_title(f"{title}\nm_d {100 * policy.m_d:.2f} cm; smallest h {100 * h.min():+.1f} cm; {int((h <= 0).sum())} spheres at h <= 0", loc="left")
axs[0].set_ylabel("y [m]")
fig.suptitle(f"{plan['mission']}, seed {args.seed}, start pose, at the base's height (z <= {band_top:.2f} m)", fontsize=8)
fig.legend(handles=[Polygon([[0, 0]], facecolor="#CFCFCF", edgecolor="#7F7F7F", lw=0.3, label="kitchen geometry (simulator)"),
                    Circle((0, 0), 1, facecolor="#E07060", alpha=0.4, label="balls of the protected objects"),
                    Circle((0, 0), 1, facecolor="none", edgecolor="#1F4E9A", lw=0.6, label="balls of the exempt objects"),
                    Circle((0, 0), 1, facecolor="#2E8B57", alpha=0.55, label="robot sphere, h > 0"),
                    Circle((0, 0), 1, facecolor="#C0392B", alpha=0.55, label="robot sphere, h <= 0")], loc="outside lower center", ncol=5, frameon=False)
out = Path("outputs/viz") / f"{plan['mission']}_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "start_top_view.pdf")
print("wrote", out / "start_top_view.pdf", "band top", round(band_top, 3))
base.close()
