"""Timeline of one episode (spec v4, section 12, item 7): outputs/viz/<mission>_s<seed>/<Task>_s<seed>_<tag>_timeline.pdf

    .venv/bin/python scripts/timeline.py --episode outputs/demos/nav_clutter/NavigateKitchenClutter_s100_sesame.json

Over time, from the episode's receipts and ground truth: (a) the smallest barrier value h of the rows and the true
clearance (every robot sphere against the collision geometry: of all non-exempt objects, and of the rows' objects only;
and, for comparison only, against the oracle boxes), with contact ticks; (b) the nominal
and the filtered joint-velocity command (base and arm norms); (c) the QP outcome per tick; (d) the rows per tick (all
pairs, and those some u in U could violate); (e) the filter's time per tick.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman"], "font.weight": "normal", "axes.titleweight": "normal",
                     "axes.labelweight": "normal", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 600})
OUTCOMES = ["passive", "passthrough", "box", "filtered", "relaxed", "hold"]
COLOURS = {"passive": "#BBBBBB", "passthrough": "#4D4D4D", "box": "#8DA0CB", "filtered": "#1F4E9A", "relaxed": "#D95F02", "hold": "#C0392B"}

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True)
args = ap.parse_args()
path = Path(args.episode)
ep = json.loads(path.read_text())
res, rec, gt = ep["result"], ep["receipts"], ep["ground_truth"]
t = 0.05 * np.array([r["tick"] for r in rec])
h = np.array([np.nan if r["h_min"] is None else r["h_min"] for r in rec])
truth = np.array([np.nan if g["clearance"] is None else g["clearance"] for g in gt])
truth_rows = np.array([np.nan if g["clearance_rows"] is None else g["clearance_rows"] for g in gt])
box = np.array([np.nan if g.get("box_clearance") is None else g["box_clearance"] for g in gt])
contact = np.array([bool(g["touched"]) for g in gt])
u_nom, u = np.array([r["u_nominal"] for r in rec]), np.array([r["u_applied"] for r in rec])

fig, axs = plt.subplots(5, 1, figsize=(7.16, 6.4), sharex=True, layout="constrained", gridspec_kw=dict(height_ratios=[2.2, 1.6, 0.5, 1.0, 1.0]))
ax = axs[0]
ax.plot(t, 100 * box, color="#E8B4A0", lw=1.0, ls=":", label="clearance to the oracle\nboxes (comparison)")
ax.plot(t, 100 * truth, color="#D9D9D9", lw=2.4, label="true clearance, all\nnon-exempt objects")
ax.plot(t, 100 * truth_rows, color="#7F7F7F", lw=1.4, label="true clearance, the\nrows' objects")
ax.plot(t, 100 * h, color="black", lw=0.8, label="smallest barrier\nvalue h of the rows")
ax.axhline(0, color="#C0392B", lw=0.5, ls="--")
if contact.any():
    ax.scatter(t[contact], np.full(contact.sum(), np.nanmin(100 * np.r_[h, truth])), marker="|", color="#C0392B", s=30, label="contact (simulator)")
ax.set_ylabel("[cm]")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title(f"(a) barrier and truth ({res['task']}, seed {res['seed']}, {res['tag']}: {res['outcome']})", loc="left")
ax = axs[1]
for sl, name, ls in ((slice(0, 3), "base", "-"), (slice(3, 10), "arm", "--")):
    ax.plot(t, np.linalg.norm(u_nom[:, sl], axis=1), color="#9A9A9A", lw=1.8, ls=ls, label=f"nominal, {name}")
    ax.plot(t, np.linalg.norm(u[:, sl], axis=1), color="black", lw=0.7, ls=ls, label=f"filtered, {name}")
ax.set_ylabel(r"$|u|$ [m/s, rad/s]")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title("(b) nominal against filtered joint-velocity command (norms of the base and arm blocks)", loc="left")
ax = axs[2]
for k, o in enumerate(OUTCOMES):
    sel = np.array([r["outcome"] == o for r in rec])
    if sel.any():
        ax.scatter(t[sel], np.zeros(sel.sum()), marker="|", s=60, color=COLOURS[o], label=f"{o} ({int(sel.sum())})")
ax.set_yticks([])
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title("(c) QP outcome per tick (box: u_nom clipped to U, no row acting; filtered: a row moved u)", loc="left")
ax = axs[3]
ax.plot(t, [r["n_pairs"] for r in rec], color="#9A9A9A", lw=1.5, label="all rows\n(spheres x objects)")
ax.plot(t, [r["n_rows_qp"] for r in rec], color="black", lw=0.7, label="rows some u in U\ncould violate")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title("(d) rows per tick", loc="left")
ax = axs[4]
ax.plot(t, [r["tick_ms"] for r in rec], color="black", lw=0.7, label="filter")
ax.plot(t, [r["solve_ms"] for r in rec], color="#9A9A9A", lw=0.7, label="QP solve")
ax.set_yscale("log")
ax.set_ylabel("[ms]")
ax.set_xlabel("t [s]")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title("(e) time per tick", loc="left")
mission = ep["plan"].get("mission", res["task"])
out = Path("outputs/viz") / f"{mission}_s{res['seed']}"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{path.stem}_timeline.pdf")
print("wrote", out / f"{path.stem}_timeline.pdf")
