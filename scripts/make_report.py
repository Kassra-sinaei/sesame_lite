"""REPORT.md for one matched-seed demo: per-arm table, filter evidence, semantic-prunability table.

    python scripts/make_report.py outputs/demo_nav --task NavigateKitchenMats --seed 0 --arms A B C
"""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--task", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arms", nargs="+", default=["A", "C"])
args = ap.parse_args()
out = Path(args.out_dir)
eps = {a: json.load(open(out / f"{args.task}_s{a and args.seed}_{a.replace('+', 'plus')}.json")) for a in args.arms}
NAMES = {"A": "Baseline (no filter)", "B": "Naive CBF (all parts)", "B+": "All parts, task exempt", "C": "SESAME-3D"}
m = 0.005

lines = [f"# {args.task}, seed {args.seed}: baseline vs SESAME-3D", ""]
r0 = next(iter(eps.values()))["result"]
lines += [f"Task sentence: *{r0['task_description']}*. Layout {r0['layout'].get('layout_id')} / style {r0['layout'].get('style_id')}, {r0['n_nodes']} 3DSG object nodes. Scene snapshot `{r0['snapshot_id']}`" + (" (identical across arms)." if len({e['result']['snapshot_id'] for e in eps.values()}) == 1 else " **(MISMATCH across arms)**."), ""]

# per-arm table
cols = ["outcome", "success (plan goal)", "RoboCasa success", "steps", "contact ticks", "min true h [cm]", "worst penetration [cm]", "severity [m s]", "rows / tick", "tick ms p50 / p99", "filtered ticks", "holds / infeasible", "classifier [s]", "start h [cm]"]
lines.append("| | " + " | ".join(NAMES[a] for a in args.arms) + " |")
lines.append("|---|" + "---|" * len(args.arms))
def cell(r, c):
    return {
        "outcome": r["outcome"], "success (plan goal)": str(r["success"]), "RoboCasa success": str(r["task_success_robocasa"]), "steps": str(r["steps"]),
        "contact ticks": str(r["contact_ticks"]), "min true h [cm]": f"{100 * r['h_min_all']:.1f}" if r["h_min_all"] is not None else "-",
        "worst penetration [cm]": f"{100 * r['penetration_m']:.1f}", "severity [m s]": f"{r['severity_ms']:.3f}", "rows / tick": f"{r['n_rows_mean']:.0f}",
        "tick ms p50 / p99": f"{r['tick_ms_median']:.1f} / {r['tick_ms_p99']:.1f}", "filtered ticks": str(r["filtered_ticks"]), "holds / infeasible": f"{r['holds']} / {r['infeasible']}",
        "classifier [s]": ", ".join(f"{x:.1f}" for x in r["classifier_latency_s"] if x) or "-", "start h [cm]": f"{100 * r['start_h']:.1f}" if r["start_h"] is not None else "-",
    }[c]
for c in cols:
    lines.append(f"| {c} | " + " | ".join(cell(eps[a]["result"], c) for a in args.arms) + " |")
lines.append("")

# filter evidence: approach speed vs barrier for the SESAME arm
if "C" in eps:
    rec, gt = eps["C"]["receipts"], eps["C"]["ground_truth"]
    pairs = [(g["h_min_all"], np.linalg.norm(r["u_applied"][:2])) for r, g in zip(rec, gt) if g["h_min_all"] is not None and r["outcome"] == "filtered" and 0 < g["h_min_all"] < 0.3]
    if pairs:
        h, v = np.array(pairs).T
        lines += ["## Filter evidence (SESAME arm)", "", f"- Filtered ticks: {eps['C']['result']['filtered_ticks']}; holds {eps['C']['result']['holds']}; QP infeasible {eps['C']['result']['infeasible']}.",
                  f"- On filtered ticks with 0 < h < 0.3 m the applied base speed stays under alpha*h - d (alpha = 1/s, d = 0.015 m/s): max(v - (h - 0.015)) = {np.max(v - (h - 0.015)):+.3f} m/s over {len(pairs)} ticks (<= 0 means the row bound is respected on the base block).",
                  f"- Minimum true barrier after the start transient: {100 * eps['C']['result']['h_min_all']:.1f} cm (margin {100 * m:.1f} cm); recovery band entries: {eps['C']['result']['ticks_below_margin_all']} ticks below the margin.", ""]

# prunability table
if "C" in eps:
    e = eps["C"]
    bundle, graph = e["bundles"][0], {n["node_id"]: n for n in e["graph"]["nodes"]}
    seg = e["segments"][0]
    n_all = len(e["graph"]["nodes"])
    n_bplus = n_all - len([nid for nid in seg["exempt_ids"] if nid in graph])
    lines += ["## Semantic prunability", "", f"Stage 1 (corridor / disc): {r0['n_nodes']} nodes -> {len(bundle['candidate_ids'])} candidates. Stage 2 labels -> {len(bundle['admitted_ids'])} admitted nodes. Rows: naive {n_all} nodes -> task-exempt {n_bplus} -> SESAME {len(e['row_ids'])}.", "",
              "| candidate | name sent to the classifier | label | effect |", "|---|---|---|---|"]
    for cid in bundle["candidate_ids"]:
        n = graph[cid]
        x, y, z = (2 * h for h in n["half_extents"])
        lab = bundle["labels"].get(cid, "obstacle")
        effect = "override -> obstacle" if cid in bundle["overrides"] else ("task contact: exempt" if cid in seg["exempt_ids"] else ("" if lab == "obstacle" else "rows removed"))
        lines.append(f"| {cid} | {n['category']} (approx {x:.2f} x {y:.2f} x {z:.2f} m box) | {lab} | {effect} |")
    dropped = [cid for cid in bundle["candidate_ids"] if cid not in bundle["admitted_ids"]]
    lines += ["", f"Nodes removed by semantics: {', '.join(dropped) or 'none'}. Nodes outside the corridor (never rows): {r0['n_nodes'] - len(bundle['candidate_ids'])}.", ""]

lines += ["## Figures", "", f"![query](fig_{args.task}_s{args.seed}_query.pdf) ![traces](fig_{args.task}_s{args.seed}_traces.pdf) ![transcript](fig_{args.task}_s{args.seed}_transcript.pdf)", "", f"Videos: `{args.task}_s{args.seed}_<arm>.mp4`, side-by-side `{args.task}_s{args.seed}_side_by_side.mp4`."]
(out / "REPORT.md").write_text("\n".join(lines))
print("\n".join(lines[:40]))

videos = [out / f"{args.task}_s{args.seed}_{a.replace('+', 'plus')}.mp4" for a in args.arms]
if all(v.exists() for v in videos):
    from sesame_lite.viz import side_by_side

    side_by_side(videos, out / f"{args.task}_s{args.seed}_side_by_side.mp4")
    print("wrote", out / f"{args.task}_s{args.seed}_side_by_side.mp4")
