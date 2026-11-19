"""Table II metrics on one segment of the runs (comparison campaign v3, 2026-11-19). The same code scores every row.

    python -s scripts/score_segment.py --segment manipulation --out outputs/eval/results/v3/shared --name step0 \
        --row "Nominal=outputs/eval/runs/{scene}/nominal_full.json" --row "Proposed=outputs/eval/runs/{scene}/proposed_full.json"

A row is ``title=template``: the template is the path of a run file of scripts/eval_run.py, with {scene} for the scene.
Every receipt and ground-truth record of a run file carries its segment; only the ticks of the chosen segment count
(navigation: segment 0; manipulation: segment 1; all: every tick, which gives Table II as eval_report.py computes it).

- success: the segment was completed (manipulation and all: the task is done);
- obstacle contact: a tick with a simulator contact between the robot and the scene's designated obstacle;
- contact: a tick with a counted contact (robot and object on a pair the plan does not allow, as in Table II);
- safe success (obstacle) / safe success: success and no obstacle contact / no contact;
- smallest clearance: the smallest true clearance of the robot's spheres over the segment, to the obstacle and to
  anything (as SafetyFilterEnv.summary: if the segment starts below zero, the ticks until it first reaches zero are
  left out of the second);
- h < 0: ticks whose receipt has h_min < 0 (the Proposed barrier, evaluated in every row, also when it does not act);
- task time: the segment's ticks x 0.05 s, for a completed segment;
- QP time: median solve time over the segment's receipts with a QP (qp_status not "none"); rows: mean n_rows_qp over
  the receipts that are not "passive"; "-" if the segment has none;
- contacts by robot part (base, arm, hand, fingers): runs and ticks with a counted contact of that part.

A run without ticks in the segment, or rows of one scene on different scenes (snapshot id), stop the script with a list.
Rates come with 95% Wilson intervals; the reference row (--ref) is compared with every other row on the same scenes by
the exact McNemar test (two-sided binomial test on the discordant scenes).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DT_S = 0.05  # filter.CONTROL_DT_S
SEGMENTS = {"navigation": 0, "manipulation": 1, "all": None}
PARTS = ("base", "arm", "hand", "fingers")  # eval_scene.ROBOT_GROUPS
BINARY = ("obstacle_contact_free", "collision_free", "success", "safe_success_obstacle", "safe_success")
# the 14 scenes of the comparison (Kasra, 2026-11-19): the pick-and-place scenes with a designated obstacle
V3_SCENES = ("sink_to_counter_s7 sink_to_counter_s11 sink_to_counter_s12 sink_to_counter_s18 sink_to_counter_s19 counter_to_cabinet_s3 counter_to_cabinet_s10 "
             "counter_to_cabinet_s15 counter_to_cabinet_s17 counter_to_stove_s0 counter_to_stove_s7 counter_to_stove_s9 counter_to_stove_s10 counter_to_stove_s14").split()


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * max(c - h, 0):.0f}-{100 * min(c + h, 1):.0f}%"


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Discordant counts (a only, b only) and the exact two-sided p-value."""
    x, y = sum(u and not v for u, v in zip(a, b)), sum(v and not u for u, v in zip(a, b))
    n = x + y
    return x, y, 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2**n)


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def score_run(path: Path, segment: int | None) -> dict:
    """The metrics of one run file on one segment (None: every tick)."""
    d = json.loads(Path(path).read_text())
    r, gt, rec = d["result"], d["ground_truth"], d["receipts"]
    if len(gt) != len(rec) or any(g["segment"] != x["segment"] for g, x in zip(gt, rec)):
        raise ValueError(f"{path}: receipts and ground truth disagree")
    keep = [k for k, g in enumerate(gt) if segment is None or g["segment"] == segment]
    g1, r1 = [gt[k] for k in keep], [rec[k] for k in keep]
    ob = (r.get("safety_obstacle") or {}).get("oracle_id")
    first_ok = next((k for k, g in enumerate(g1) if g["clearance"] is not None and g["clearance"] >= 0.0), len(g1)) if g1 and (g1[0]["clearance"] or 0.0) < 0.0 else 0
    done = r["success"] if segment is None else r["segments_done"] > segment
    ob_ticks = sum(any(c.endswith(f"|{ob}") for c in g["contacts"]) for g in g1) if ob else 0
    contact_ticks = sum(bool(g["touched"]) for g in g1)
    qp = [x["solve_ms"] for x in r1 if x["qp_status"] != "none"]
    active = [x["n_rows_qp"] for x in r1 if x["outcome"] != "passive"]
    out = {"scene": r["scene"], "snapshot_id": r["snapshot_id"], "file": str(path), "ticks": len(g1), "first_tick": g1[0]["tick"] if g1 else None,
           "success": bool(done), "obstacle": bool(ob), "obstacle_contact_free": ob_ticks == 0, "collision_free": contact_ticks == 0,
           "safe_success_obstacle": bool(done and ob_ticks == 0), "safe_success": bool(done and contact_ticks == 0),
           "obstacle_contact_ticks": ob_ticks, "collision_ticks": contact_ticks,
           "clearance_obstacle_m": min((g["clearance_obstacle"] for g in g1 if g.get("clearance_obstacle") is not None), default=None),
           "clearance_m": min((g["clearance"] for g in g1[first_ok:] if g["clearance"] is not None), default=None),
           "negative_barrier_ticks": sum(1 for x in r1 if x["h_min"] is not None and x["h_min"] < 0.0),
           "task_time_s": len(g1) * DT_S if done else None, "qp_ms_median": med(qp), "rows_mean": float(np.mean(active)) if active else None}
    for part in PARTS:
        ticks = [any(c[0] == part for c in g["contact_points"]) for g in g1]
        out[f"contact_{part}_ticks"] = int(sum(ticks))
    out["contact_objects"] = sorted({f"{c[0]}>{c[1]}" for g in g1 for c in g["contact_points"]})
    return out


def table(runs: dict[str, list[dict]], ref: str | None) -> list[dict]:
    out = []
    for title, rs in runs.items():
        n = len(rs)
        row = {"row": title, "runs": n}
        for key in BINARY:
            k = sum(x[key] for x in rs)
            row[key], row[key + "_ci95"] = k, wilson(k, n)
            if ref and title != ref:
                a, b, p = mcnemar([x[key] for x in runs[ref]], [x[key] for x in rs])
                row[key + "_vs_ref"] = f"{a}:{b}, p={p:.3g}"
        row.update(clearance_obstacle_m_median=med([x["clearance_obstacle_m"] for x in rs]), clearance_m_median=med([x["clearance_m"] for x in rs]),
                   runs_negative_barrier=sum(x["negative_barrier_ticks"] > 0 for x in rs), negative_barrier_ticks=sum(x["negative_barrier_ticks"] for x in rs),
                   task_time_s_median=med([x["task_time_s"] for x in rs]), qp_ms_median=med([x["qp_ms_median"] for x in rs]), rows_median=med([x["rows_mean"] for x in rs]))
        for part in PARTS:
            row[f"contact_{part}_runs"] = sum(x[f"contact_{part}_ticks"] > 0 for x in rs)
            row[f"contact_{part}_ticks"] = sum(x[f"contact_{part}_ticks"] for x in rs)
        out.append(row)
    return out


def md(rows: list[dict], cols: list[tuple]) -> list[str]:
    out = ["| " + " | ".join(t for _, t in cols) + " |", "|" + "---|" * len(cols)]
    return out + ["| " + " | ".join(str(f(r)) if callable(f) else str(r.get(f, "")) for f, _ in cols) + " |" for r in rows]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True, choices=list(SEGMENTS))
    ap.add_argument("--row", action="append", required=True, help='"title=run-file template with {scene}"; the order of the table')
    ap.add_argument("--ref", default="Proposed", help="the row every other row is compared with (McNemar)")
    ap.add_argument("--scenes", nargs="+", default=V3_SCENES)
    ap.add_argument("--out", default=None, help="directory for <name>_table.csv, <name>_runs.csv and <name>.md")
    ap.add_argument("--name", default="table")
    args = ap.parse_args()
    seg = SEGMENTS[args.segment]
    rows = dict(x.split("=", 1) for x in args.row)
    ref = args.ref if args.ref in rows else None
    runs, missing, empty = {}, [], []
    for title, tmpl in rows.items():
        runs[title] = []
        for scene in args.scenes:
            p = Path(tmpl.format(scene=scene))
            if not p.exists():
                missing.append(str(p))
                continue
            x = score_run(p, seg)
            if x["ticks"] == 0:
                empty.append(str(p))
            runs[title].append(x)
    snap = {}
    for rs in runs.values():
        for x in rs:
            snap.setdefault(x["scene"], set()).add(x["snapshot_id"])
    mixed = [s for s, ids in snap.items() if len(ids) > 1]
    if missing or empty or mixed:
        sys.exit("not scored:\n" + "\n".join([f"  missing run file: {p}" for p in missing] + [f"  no ticks in the {args.segment} segment: {p}" for p in empty]
                                             + [f"  rows on different scenes (snapshot ids): {s}" for s in mixed]))
    tab = table(runs, ref)
    cm = lambda v: "-" if v is None else f"{100 * v:.1f} cm"  # noqa: E731
    frac = lambda key: (lambda r: f"{r[key]}/{r['runs']} ({r[key + '_ci95']})")  # noqa: E731
    cols = [("row", "Row"), (frac("obstacle_contact_free"), "No obstacle contact"), (frac("collision_free"), "No contact"), (frac("success"), "Task success"),
            (frac("safe_success_obstacle"), "Safe success (obstacle)"), (frac("safe_success"), "Safe success"),
            (lambda r: cm(r["clearance_obstacle_m_median"]), "Median clearance to the obstacle"), (lambda r: cm(r["clearance_m_median"]), "Median clearance"),
            (lambda r: f"{r['runs_negative_barrier']} ({r['negative_barrier_ticks']} ticks)", "Runs with h < 0 (ticks)"),
            (lambda r: "-" if r["task_time_s_median"] is None else f"{r['task_time_s_median']:.1f} s", "Median task time"),
            (lambda r: "-" if r["qp_ms_median"] is None else f"{r['qp_ms_median']:.2f} ms", "Median QP time"),
            (lambda r: "-" if r["rows_median"] is None else f"{r['rows_median']:.0f}", "Median rows")]
    L = [f"Segment: {args.segment}; {len(args.scenes)} scenes; one run per scene and row.", ""] + md(tab, cols)
    if ref:
        L += ["", f"Exact McNemar test, {ref} against each row on the same scenes: scenes where only {ref} has the outcome : scenes where only the other row has it, p.", ""]
        L += md([r for r in tab if r["row"] != ref], [("row", "Row")] + [((lambda k: lambda r: r[k + "_vs_ref"])(k), t) for k, (_, t) in zip(BINARY, cols[1:6])])
    L += ["", "Counted contacts by robot part: runs with a contact of that part (ticks).", ""]
    L += md(tab, [("row", "Row")] + [((lambda p: lambda r: f"{r[f'contact_{p}_runs']} ({r[f'contact_{p}_ticks']})")(p), p) for p in PARTS])
    print("\n".join(L))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / f"{args.name}_table.csv", tab)
        write_csv(out / f"{args.name}_runs.csv", [{"row": t, **x} for t, rs in runs.items() for x in rs])
        (out / f"{args.name}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
