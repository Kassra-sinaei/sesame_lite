"""Figures of one episode with the ISSf filter on confidence boxes (spec 5.5), from its receipts only.

    .venv/bin/python scripts/issf_figures.py outputs/phase2/nav_mats/NavigateKitchenMats_s100_sesame.json

Writes next to the episode file:
  <stem>_h.pdf        h of the closest row (confidence boxes) and the true clearance (oracle boxes) over time
  <stem>_map.pdf      top view: oracle boxes, the filter side's estimated boxes, confidence boxes, path and trajectory
  <stem>_ksigma.pdf   histogram of kappa*sigma over the faces of the admitted nodes (rows), per segment
The graph source is printed on every figure (e.g. "synthetic face noise, not Hydra").
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 600, "figure.facecolor": "white",
})
COL1, COL2 = 3.5, 7.16  # IEEE column widths [in]
DT = 0.05


def footprint(center, half, rotation=None):
    """Top-view outline of a box (convex hull of its corners projected on x-y)."""
    from shapely.geometry import MultiPoint

    R = np.eye(3) if rotation is None else np.asarray(rotation)
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = np.asarray(center) + (signs * np.asarray(half)) @ R.T
    return np.asarray(MultiPoint([tuple(p) for p in corners[:, :2]]).convex_hull.exterior.coords)


def main(path: Path) -> None:
    ep = json.loads(path.read_text())
    res, rec, gt = ep["result"], ep["receipts"], ep["ground_truth"]
    source = res.get("graph_label", res.get("graph_source", ""))
    stem = path.with_suffix("")
    t = np.arange(len(rec)) * DT

    # (1) h of the closest row and the true clearance
    fig, ax = plt.subplots(figsize=(COL1, 1.9), layout="constrained")
    h_row = np.array([np.nan if r["h_min"] is None else r["h_min"] for r in rec])
    h_true = np.array([np.nan if g["h_min_all"] is None else g["h_min_all"] for g in gt])
    ax.plot(t, 100 * h_row, color="#0072B2", lw=1.0, label="closest row $h$ (confidence boxes)")
    ax.plot(t, 100 * h_true, color="black", lw=1.0, ls="--", label="true clearance (oracle boxes)")
    ax.axhline(0, color="#C0392B", lw=0.5)
    for k, seg_start in enumerate(sorted({r["epoch"]: i for i, r in reversed(list(enumerate(rec)))}.values())):
        if k:
            ax.axvline(seg_start * DT, color="#9A9A9A", lw=0.5, ls=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[cm]")
    ax.set_title(f"{res['task']}, seed {res['seed']} ({source})", loc="left")
    ax.legend(loc="best", frameon=False)
    fig.savefig(f"{stem}_h.pdf")
    plt.close(fig)

    # (2) top view
    nodes = {n["node_id"]: n for n in ep["graph"]["nodes"]}
    rows = set(ep["row_ids"])
    conf = ep["confidence"][-1]["boxes"] if ep["confidence"] else {}
    xy = np.array([g["base_xy"] for g in gt])
    eef = np.array([g["eef"] for g in gt])
    pts = [xy, eef[:, :2]] + [np.asarray(s["path_polyline_xy"]) for s in ep["segments"] if s["path_polyline_xy"]]
    lo, hi = np.min([p.min(axis=0) for p in pts], axis=0) - 1.0, np.max([p.max(axis=0) for p in pts], axis=0) + 1.0
    fig, ax = plt.subplots(figsize=(COL1, COL1), layout="constrained")
    for n in ep["oracle_graph"]["nodes"]:
        ax.add_patch(Polygon(footprint(n["center"], n["half_extents"]), closed=True, facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.35, zorder=1))
    for nid, n in nodes.items():
        ax.add_patch(Polygon(footprint(n["center"], n["half_extents"], n.get("rotation")), closed=True, fill=False, edgecolor="black", lw=0.4, zorder=3))
        if nid in rows and nid in conf:
            ax.add_patch(Polygon(footprint(conf[nid]["center"], conf[nid]["half_extents"], n.get("rotation")), closed=True, fill=False, edgecolor="#C0392B", lw=0.5, ls=(0, (2.5, 1.5)), zorder=4))
    for s in ep["segments"]:
        if s["path_polyline_xy"]:
            p = np.asarray(s["path_polyline_xy"])
            ax.plot(p[:, 0], p[:, 1], color="#1F4E9A", lw=0.7, ls=":", zorder=5)
            ax.plot(*p[-1], marker="*", ms=7, color="#1F4E9A", mec="white", mew=0.4, zorder=7)
    ax.plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=1.1, zorder=6)
    ax.plot(eef[:, 0], eef[:, 1], color="#E69F00", lw=0.7, zorder=6)
    ax.plot(*xy[0], marker="o", ms=4, color="black", zorder=7)
    hit = np.array([g["base_xy"] for g in gt if g["touched"]])
    if len(hit):
        ax.plot(hit[:, 0], hit[:, 1], "x", color="#C0392B", ms=3.5, mew=0.8, zorder=8)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{res['outcome'].replace('_', ' ')} ({source})", loc="left")
    handles = [Patch(facecolor="#D9D9D9", edgecolor="#9A9A9A", lw=0.35, label="oracle box"), Line2D([], [], color="black", lw=0.5, label="estimated box"),
               Line2D([], [], color="#C0392B", lw=0.6, ls=(0, (2.5, 1.5)), label="confidence box (row)"), Line2D([], [], color="#1F4E9A", lw=0.7, ls=":", label="planned path"),
               Line2D([], [], color="#0072B2", lw=1.1, label="base"), Line2D([], [], color="#E69F00", lw=0.7, label="end effector")]
    if len(hit):
        handles.append(Line2D([], [], marker="x", ls="", color="#C0392B", ms=4, label="contact"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=1.6, columnspacing=0.9)
    fig.savefig(f"{stem}_map.pdf")
    plt.close(fig)

    # (3) kappa*sigma over the admitted faces
    fig, ax = plt.subplots(figsize=(COL1, 1.8), layout="constrained")
    for c in ep["confidence"]:
        v = 100 * np.asarray(c["kappa_sigma_admitted_faces"])
        if len(v):
            ax.hist(v, bins=30, alpha=0.7, lw=0, label=f"segment {c['segment']}: $\\kappa$ = {c['kappa']:.2f}, {len(v) // 6} nodes")
    ax.set_xlabel(r"$\kappa\sigma$ [cm]"); ax.set_ylabel("faces")
    ax.set_title(f"admitted faces ({source})", loc="left")
    ax.legend(frameon=False)
    fig.savefig(f"{stem}_ksigma.pdf")
    plt.close(fig)
    print("wrote", f"{stem}_h.pdf", f"{stem}_map.pdf", f"{stem}_ksigma.pdf")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(Path(arg))
