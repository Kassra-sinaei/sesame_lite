"""Publication figures for one matched-seed demo, from the episode receipts only.

    python scripts/make_figures.py outputs/demo_nav --task NavigateKitchenMats --seed 0 --arms A B C

Writes, next to the receipts:
  fig_<task>_s<seed>_query.pdf       3DSG query of the SESAME arm (single column): every part's footprint coloured by its
                                     role (constraint, task-contact exempt, passable, irrelevant, outside the region),
                                     the swept corridor / manipulation disc, path, start and goal
  fig_<task>_s<seed>_compare.pdf     arm comparison (double column): (a) trajectories with contacts, (b) true clearance over
                                     every non-exempt part, (c) progress to the goal
  fig_<task>_s<seed>_transcript.pdf  the candidate list sent to the stage-2 classifier and the labels returned
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "savefig.dpi": 600, "figure.facecolor": "white",
})
COL1, COL2 = 3.5, 7.16  # IEEE column widths [in]
ROLE_COLOR = {"row": "#C0392B", "exempt": "#2A6FD0", "passable": "#2E9E5B", "irrelevant": "#9A9A9A", "outside": "#E6E6E6"}
ROLE_LABEL = {"row": "constraint row", "exempt": "task-contact exempt", "passable": "passable (no row)", "irrelevant": "irrelevant (no row)", "outside": "outside the region"}
ARM = {  # Okabe-Ito colours + line styles (colour is never the only cue)
    "A": dict(label="No filter", color="#7F7F7F", ls="-", lw=1.1),
    "B": dict(label="Naive CBF (all objects)", color="#E69F00", ls="--", lw=1.3),
    "B+": dict(label="All objects, target exempt", color="#CC79A7", ls="-.", lw=1.2),
    "C": dict(label="SESAME-3D (ours)", color="#0072B2", ls="-", lw=1.8),
}


def load(out_dir, task, seed, arm):
    return json.load(open(Path(out_dir) / f"{task}_s{seed}_{arm.replace('+', 'plus')}.json"))


def footprint(part):
    c, h = np.asarray(part["center"]), np.asarray(part["half_extents"])
    x, y, z, w = part["orientation_xyzw"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * h[:2] @ R.T + c[:2]


def corridor_polygon(path, r):
    """Outline of the swept corridor: the planned polyline buffered by r (shapely), i.e. the region of eq. (anav)."""
    from shapely.geometry import LineString

    return np.asarray(LineString(np.asarray(path, dtype=float)).buffer(r, quad_segs=24).exterior.coords)


def part_roles(ep, seg_index=0):
    seg, bundle = ep["segments"][seg_index], ep["bundles"][seg_index]
    rows, labels, cands, exempt = set(ep.get("row_ids", [])), bundle["labels"], set(bundle["candidate_ids"]), seg["exempt_parts"]
    out = {}
    for node in ep["graph"]["nodes"]:
        if node["kind"] == "robot":
            continue
        nid = node["node_id"]
        for p in node["parts"]:
            if p["part_id"] in rows:
                r = "row"
            elif nid in exempt and (not exempt[nid] or any(s in p["body"] for s in exempt[nid])):
                r = "exempt"
            elif labels.get(nid) in ("passable", "irrelevant"):
                r = labels[nid]
            elif nid in cands:
                r = "row"
            else:
                r = "outside"
            out[p["part_id"]] = r
    return out


def mission_extent(ep, pad=0.45):
    seg, bundle = ep["segments"][0], ep["bundles"][0]
    pts = [ep["ground_truth"][0]["base_xy"]]
    if seg["path_polyline_xy"]:
        pts += seg["path_polyline_xy"]
        pad += bundle["corridor_radius_m"] or 0.0
    else:
        pts += [seg["center_xy"]]
        pad += bundle["region_radius_m"] or 1.0
    pts = np.asarray(pts)
    return pts.min(axis=0) - pad, pts.max(axis=0) + pad


def draw_parts(ax, ep, roles, *, muted=False, z_max=None):
    """Footprints, drawn largest first so small objects stay visible on top of counters."""
    parts = [(p, roles[p["part_id"]]) for n in ep["graph"]["nodes"] if n["kind"] != "robot" for p in n["parts"]]
    parts.sort(key=lambda pr: -np.prod(pr[0]["half_extents"][:2]))
    for p, r in parts:
        if z_max is not None and p["center"][2] - p["half_extents"][2] > z_max:
            continue
        face = ROLE_COLOR[r]
        if muted and not p["part_id"].startswith("object:"):  # fixtures as context, free objects keep their role colour
            face = "#D6D6D6"
        ax.add_patch(Polygon(footprint(p), closed=True, facecolor=face, edgecolor="#4D4D4D" if r != "outside" else "#BDBDBD", linewidth=0.35, alpha=0.9 if r != "outside" else 0.7, zorder=2))


def annotate_injections(ax, ep):
    nodes = {n["node_id"]: n for n in ep["graph"]["nodes"]}
    for inj in ep.get("plan", {}).get("injections", []):
        n = nodes.get(f"object:{inj['name']}")
        if n is None:
            continue
        c = np.asarray(n["center"][:2])
        name = n["category"].replace("_", " ")
        ax.annotate(f"{name}\n({inj['ground_truth']})", c, xytext=(0, -16 if inj["ground_truth"] == "passable" else -18), textcoords="offset points",
                    ha="center", va="top", fontsize=6, color="#2E9E5B" if inj["ground_truth"] == "passable" else "#C0392B",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"), zorder=7)


def style_map(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")


def query_map(ep, out_path):
    seg, bundle, res = ep["segments"][0], ep["bundles"][0], ep["result"]
    roles = part_roles(ep)
    lo, hi = mission_extent(ep)
    fig, ax = plt.subplots(figsize=(COL1, COL1 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 0.9), layout="constrained")
    if seg["path_polyline_xy"]:
        path = np.asarray(seg["path_polyline_xy"])
        ax.add_patch(Polygon(corridor_polygon(path, bundle["corridor_radius_m"]), closed=True, facecolor="#2A6FD0", alpha=0.08, edgecolor="#2A6FD0", lw=0.6, ls="--", zorder=1))
        ax.plot(path[:, 0], path[:, 1], color="#1F4E9A", lw=0.9, zorder=4)
        ax.plot(*path[-1], marker="*", ms=9, color="#1F4E9A", mec="white", mew=0.5, zorder=6)
    else:
        ax.add_patch(plt.Circle(seg["center_xy"], bundle["region_radius_m"], facecolor="#2A6FD0", alpha=0.08, edgecolor="#2A6FD0", lw=0.6, ls="--", zorder=1))
    draw_parts(ax, ep, roles)
    ax.plot(*ep["ground_truth"][0]["base_xy"], marker="o", ms=4.5, color="black", zorder=6)
    annotate_injections(ax, ep)
    style_map(ax, lo, hi)
    present = [r for r in ("row", "exempt", "passable", "irrelevant", "outside") if r in roles.values()]
    handles = [Patch(facecolor=ROLE_COLOR[r], edgecolor="#4D4D4D", lw=0.35, label=ROLE_LABEL[r]) for r in present]
    handles += [Line2D([], [], color="#1F4E9A", lw=0.9, label="planned path"), Patch(facecolor="#2A6FD0", alpha=0.15, edgecolor="#2A6FD0", ls="--", lw=0.6, label="swept corridor"),
                Line2D([], [], marker="o", ls="", color="black", ms=4, label="start"), Line2D([], [], marker="*", ls="", color="#1F4E9A", ms=7, label="goal")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, handlelength=1.4, columnspacing=1.0)
    ax.text(0.02, 0.98, f"{res['n_nodes']} nodes $\\to$ {len(bundle['candidate_ids'])} in corridor $\\to$ {len(bundle['admitted_ids'])} constrained\n"
            f"{res['n_fragments']} parts $\\to$ {len(ep['row_ids'])} constraint rows", transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", lw=0.4), zorder=8)
    fig.savefig(out_path)
    plt.close(fig)


def compare(eps, out_path, margin_m=0.005):
    ref = next((e for e in eps if e["result"]["arm"] == "C"), eps[-1])
    lo, hi = mission_extent(ref, pad=0.25)
    goal = np.asarray(ref["segments"][0]["path_polyline_xy"][-1]) if ref["segments"][0]["path_polyline_xy"] else None
    fig = plt.figure(figsize=(COL2, 2.35), layout="constrained")
    axs = fig.subplot_mosaic([["map", "h", "d"]], width_ratios=[1.35, 1, 1])
    ax = axs["map"]
    draw_parts(ax, ref, part_roles(ref), muted=True, z_max=1.6)
    for ep in eps:
        a = ARM[ep["result"]["arm"]]
        xy = np.array([g["base_xy"] for g in ep["ground_truth"]])
        ax.plot(xy[:, 0], xy[:, 1], color=a["color"], ls=a["ls"], lw=a["lw"], zorder=5)
        hit = np.array([g["base_xy"] for g in ep["ground_truth"] if g["touched"]])
        if len(hit):
            ax.plot(hit[:, 0], hit[:, 1], "x", color="#C0392B", ms=3.5, mew=0.9, zorder=6)
        ax.plot(*xy[-1], marker="s" if ep["result"]["stalled"] else "o", ms=3.5, color=a["color"], mec="black", mew=0.4, zorder=7)
    ax.plot(*ref["ground_truth"][0]["base_xy"], marker="o", ms=4, color="black", zorder=7)
    if goal is not None:
        ax.plot(*goal, marker="*", ms=8, color="#1F4E9A", mec="white", mew=0.5, zorder=7)
    style_map(ax, lo, hi)
    ax.set_title("(a) base trajectories", loc="left")

    ax = axs["h"]
    for ep in eps:
        a = ARM[ep["result"]["arm"]]
        t = np.arange(len(ep["ground_truth"])) * 0.05
        h = np.array([g["h_min_all"] if g["h_min_all"] is not None else np.nan for g in ep["ground_truth"]])
        ax.plot(t, 100 * h, color=a["color"], ls=a["ls"], lw=a["lw"])
    ax.axhspan(-30, -100 * margin_m, color="#C0392B", alpha=0.10, lw=0)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-100 * margin_m, color="#C0392B", lw=0.5, ls=":")
    ax.text(0.98, 0.04, "penetration", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#C0392B")
    ymin = min(-8.0, np.nanmin([100 * (g["h_min_all"] if g["h_min_all"] is not None else 0) for ep in eps for g in ep["ground_truth"]]) - 1)
    ax.set_ylim(ymin, 30)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clearance $h_{\\min}$ [cm]")
    ax.set_title("(b) true clearance, all non-exempt parts", loc="left")

    ax = axs["d"]
    for ep in eps:
        a = ARM[ep["result"]["arm"]]
        if goal is None:
            break
        xy = np.array([g["base_xy"] for g in ep["ground_truth"]])
        t = np.arange(len(xy)) * 0.05
        ax.plot(t, np.linalg.norm(xy - goal, axis=1), color=a["color"], ls=a["ls"], lw=a["lw"])
        ax.plot(t[-1], np.linalg.norm(xy[-1] - goal), marker="s" if ep["result"]["stalled"] else "o", ms=3.5, color=a["color"], mec="black", mew=0.4)
    ax.axhline(0.2, color="black", lw=0.5, ls=":")
    ax.text(0.98, 0.2, "success radius", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance to goal [m]")
    ax.set_title("(c) progress", loc="left")

    handles = [Line2D([], [], color=ARM[e["result"]["arm"]]["color"], ls=ARM[e["result"]["arm"]]["ls"], lw=ARM[e["result"]["arm"]]["lw"], label=ARM[e["result"]["arm"]]["label"]) for e in eps]
    handles += [Line2D([], [], marker="x", ls="", color="#C0392B", ms=4, label="contact"), Line2D([], [], marker="o", ls="", color="white", mec="black", ms=4, label="end: goal"),
                Line2D([], [], marker="s", ls="", color="white", mec="black", ms=4, label="end: stalled")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, handlelength=2.0, columnspacing=1.2)
    fig.savefig(out_path)
    plt.close(fig)


def transcript(ep, out_path):
    bundle, graph, seg = ep["bundles"][0], {n["node_id"]: n for n in ep["graph"]["nodes"]}, ep["segments"][0]
    cls = bundle["classification"] or {}
    rows, colors = [], []
    for cid in bundle["candidate_ids"]:
        n = graph[cid]
        x, y, z = (2 * h for h in n["half_extents"])
        lab = bundle["labels"].get(cid, "obstacle")
        if cid in bundle["overrides"]:
            effect = "override: kept as obstacle"
        elif cid in seg["exempt_parts"]:
            effect = "task contact: " + (", ".join(seg["exempt_parts"][cid]) if seg["exempt_parts"][cid] else "whole node") + " exempt"
        else:
            effect = "rows kept" if lab == "obstacle" else "rows removed"
        rows.append([cid.split(":", 1)[1], f"{n['category'].replace('_', ' ')} ({x:.2f} x {y:.2f} x {z:.2f} m)", lab, effect])
        colors.append({"passable": "#DDF1E4", "irrelevant": "#ECECEC"}.get(lab, "white"))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][2] == "obstacle", rows[i][0]))
    rows, colors = [rows[i] for i in order], [colors[i] for i in order]
    fig, ax = plt.subplots(figsize=(COL2, 0.15 * len(rows) + 0.55), layout="constrained")
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["3DSG node", "name sent to the classifier", "label", "effect on the filter"], loc="upper center", cellLoc="left", colWidths=[0.30, 0.38, 0.10, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("#BBBBBB")
        cell.set_height(1.0 / (len(rows) + 1))
        if r == 0:
            cell.set_text_props(fontweight="normal", color="white")
            cell.set_facecolor("#4D4D4D")
        else:
            cell.set_facecolor(colors[r - 1])
    lat = cls.get("latency_s")
    ax.set_title(f"Stage-2 query, goal \"{seg['goal_text']}\": {len(rows)} candidates, model {cls.get('model_identity') or ep['result']['classifier']}"
                 + (f", {lat:.1f} s, one call per segment" if lat else ""), loc="left", fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "C"])
    args = ap.parse_args()
    eps = [load(args.out_dir, args.task, args.seed, a) for a in args.arms]
    stem = Path(args.out_dir) / f"fig_{args.task}_s{args.seed}"
    ours = next((e for e in eps if e["result"]["arm"] == "C"), eps[-1])
    query_map(ours, f"{stem}_query.pdf")
    compare(eps, f"{stem}_compare.pdf")
    if ours["bundles"][0]["classification"]:
        transcript(ours, f"{stem}_transcript.pdf")
    print("wrote", *(str(p) for p in sorted(Path(args.out_dir).glob(f"fig_{args.task}_s{args.seed}_*.pdf"))))
