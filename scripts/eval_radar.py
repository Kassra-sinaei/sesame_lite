"""Radar plots of Table II (36 scenes), one per method, and a preview page with the table beside them.

    python scripts/eval_radar.py

Reads outputs/eval/results/v2/table2_36.csv. The axes are the five rates of Table II, in its column order, on one
0-100 % scale. One panel per full-pass-map method; Proposed is orange in every panel (dashed as the reference
outline), the panel's own method blue. Writes outputs/eval/figures/radar_table2_36.pdf/.png (one IEEE column, 3.5 in
wide) and preview_table2_radar.pdf/.png (a two-column page: Table II and the radar figure; a preview only, not the
paper).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "outputs/eval/results/v2/table2_36.csv"
OUT = REPO / "outputs/eval/figures"
AXES = [("obstacle_contact_free", "No obstacle\ncontact"), ("collision_free", "No\ncontact"), ("success", "Task\nsuccess"),
        ("safe_success_obstacle", "Safe success\n(obstacle)"), ("safe_success", "Safe\nsuccess")]
METHODS = ["Nominal", "End-effector filter", "Fixed margin", "Proposed without ISSf", "Proposed"]
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "axes.edgecolor": GRID})

rows = {r["row"]: r for r in csv.DictReader(open(TABLE))}
rate = {m: np.array([int(rows[m][k]) / int(rows[m]["runs"]) for k, _ in AXES]) for m in METHODS}
n = int(rows["Proposed"]["runs"])
theta = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)


def closed(v):
    return np.r_[v, v[:1]]


def panel(ax, method, labels=False):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(theta)
    ax.set_xticklabels([lab for _, lab in AXES] if labels else [], color=INK2, fontsize=5)
    ax.tick_params(axis="x", pad=1)
    ax.grid(color=GRID, linewidth=0.5)
    ax.spines["polar"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    t = closed(theta)
    if method != "Proposed":
        ax.plot(t, closed(rate["Proposed"]), color=ORANGE, linewidth=0.9, linestyle=(0, (3, 1.5)), zorder=3)
    colour = ORANGE if method == "Proposed" else BLUE
    ax.fill(t, closed(rate[method]), color=colour, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(t, closed(rate[method]), color=colour, linewidth=1.2, zorder=4)
    ss = int(rows[method]["safe_success"])
    ax.set_title(f"{method}\nsafe success {ss}/{n}", color=INK, fontsize=6, pad=3)


def radar_figure(fig, grid_spec):
    axes = [fig.add_subplot(grid_spec[i // 3, i % 3], projection="polar") for i in range(len(METHODS))]
    for ax, m in zip(axes, METHODS):
        panel(ax, m)
    key = fig.add_subplot(grid_spec[1, 2], projection="polar")  # the axis names, once
    key.set_theta_offset(np.pi / 2)
    key.set_theta_direction(-1)
    key.set_ylim(0, 1)
    key.set_yticks([0.5, 1.0])
    key.set_yticklabels(["50%", "100%"], color=INK2, fontsize=4.5)
    key.set_rlabel_position(180)  # between the two lower axes, away from the labels
    key.set_xticks(theta)
    key.set_xticklabels([lab for _, lab in AXES], color=INK, fontsize=5)
    key.tick_params(axis="x", pad=0)
    key.grid(color=GRID, linewidth=0.5)
    key.spines["polar"].set_color(GRID)
    key.set_title("Axes: rates of\nTable II", color=INK, fontsize=6, pad=7)
    return axes


OUT.mkdir(parents=True, exist_ok=True)
# 1. the figure, one column wide
fig = plt.figure(figsize=(3.5, 2.75), facecolor="white")
gs = fig.add_gridspec(2, 3, left=0.06, right=0.88, top=0.86, bottom=0.08, wspace=0.55, hspace=0.75)
radar_figure(fig, gs)
fig.legend(handles=[plt.Line2D([], [], color=BLUE, linewidth=1.2, label="method of the panel"),
                    plt.Line2D([], [], color=ORANGE, linewidth=0.9, linestyle=(0, (3, 1.5)), label="Proposed (reference)")],
           loc="upper center", ncol=2, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 1.0))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"radar_table2_36.{ext}", dpi=300)
plt.close(fig)

# 2. preview page: Table II (left column) and the radar figure (right column), US letter, IEEE-like margins
page = plt.figure(figsize=(8.5, 11), facecolor="white")
page.text(0.5, 0.965, "Preview only (not the paper): Table II, 36 scenes, and the radar plots", ha="center", fontsize=9, color=INK2)
col_w, gap, left = 3.5 / 8.5, 0.25 / 8.5, 0.625 / 8.5
tab_ax = page.add_axes([left, 0.62, col_w, 0.3])
tab_ax.axis("off")
cols = [("row", "Method")] + [(k, lab) for k, lab in AXES]
cells = [[rows[m]["row"].replace("Proposed without ISSf", "Without ISSf").replace("End-effector filter", "End-effector")]
         + [f"{rows[m][k]}/{rows[m]['runs']}" for k, _ in AXES[:]] for m in METHODS]
tbl = tab_ax.table(cellText=cells, colLabels=[c for _, c in cols], loc="upper center", cellLoc="center", colLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(5.5)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.4)
    if r == 0:
        cell.set_text_props(color=INK, fontweight="bold")
        cell.set_height(cell.get_height() * 2.2)
    if c == 0:
        cell.set_text_props(ha="left")
tab_ax.set_title("TABLE II\nCLOSED-LOOP RESULTS ON 36 SCENES (COUNTS)", fontsize=6.5, color=INK, pad=2)
gs2 = page.add_gridspec(2, 3, left=left + col_w + gap + 0.01, right=left + 2 * col_w + gap - 0.01, top=0.88, bottom=0.66, wspace=0.55, hspace=0.75)
radar_figure(page, gs2)
page.text(left + col_w + gap + col_w / 2, 0.905, "blue: method of the panel    orange dashed: Proposed (reference)", ha="center", fontsize=5.5, color=INK2)
page.text(left + col_w + gap + col_w / 2, 0.635, "Fig. X. Rates of Table II per method (36 scenes).", ha="center", fontsize=6.5, color=INK)
for ext in ("pdf", "png"):
    page.savefig(OUT / f"preview_table2_radar.{ext}", dpi=200)
print("wrote", OUT / "radar_table2_36.pdf", OUT / "preview_table2_radar.pdf")
