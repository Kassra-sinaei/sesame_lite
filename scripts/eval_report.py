"""Report of the Table II campaign with Xiaomi-Robotics-1 (2026-11-17): outputs/eval/REPORT.md and, in
outputs/eval/results/v2/, table2.csv, table2_by_instruction.csv, runs.csv and contacts.csv. The first campaign's files
keep their contents under the suffix _first_campaign in outputs/eval/results/.

    python -s scripts/eval_report.py

Reads the run files in outputs/eval/runs/, the scenes (scene.json with the designated obstacle, maps) and
outputs/eval/results/v2/notes.md (written by hand: tuning, choices, deviations), which opens the report.

Definitions:
- obstacle contact: a simulator contact between the robot and the designated obstacle; contact: any counted contact
  (as in the first table: robot and object on a pair the plan does not allow);
- safe success (obstacle): task done and no obstacle contact; safe success: task done and no contact;
- smallest clearance: the smallest true clearance of the robot's spheres over the run, to the obstacle and to anything;
- a contact is on mapped geometry if its contact point lies inside a ball of the run's map (Proposed radii,
  delta_j = 0.01, no tracking margin), else on unmapped geometry;
- a campaign item enters a table only when it is complete; a scene with a run that crashed twice is left out of every
  row; rates come with 95% Wilson intervals; Proposed is compared with every other row on the same scenes by the exact
  McNemar test (two-sided binomial test on the discordant scenes).
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from sesame_lite.eval_scene import EVAL_DIR, robot_group
from sesame_lite.filter import CONTROL_DT_S
from sesame_lite.uncertainty import balls

R = EVAL_DIR / "results" / "v2"
R0 = EVAL_DIR / "results"
SCENES_DIR = EVAL_DIR / "scenes_v2"  # the campaign scenes with their obstacles (the first campaign's: outputs/eval/scenes)
ROWS = [("nominal", "full", "Nominal"), ("ee", "full", "End-effector filter"), ("fixed", "full", "Fixed margin"), ("noissf", "full", "Proposed without ISSf"),
        ("proposed", "full", "Proposed"), ("fixed", "look", "Fixed margin, look-around map"), ("proposed", "look", "Proposed, look-around map")]
ITEM = {"full": "a", "look": "b"}
tests = [t for t in json.loads((R0 / "test_scenes.json").read_text())["test_scenes"] if t["in_table2"]]
extra_path = R / "test_scenes_extra.json"
extra = json.loads(extra_path.read_text())["test_scenes"] if extra_path.exists() else []


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
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2**n)
    return x, y, p


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


_balls = {}


def map_balls(scene: str, kind: str):
    if (scene, kind) not in _balls:
        p = np.load(SCENES_DIR / scene / f"map_{kind}" / "points.npz")
        r = [balls(p["points"][a:b], p["std"][a:b], delta_j=0.01, voxel_size=float(p["voxel_size"]), m_r=float(p["m_r"]), m_d=0.0).rho for a, b in zip(p["offsets"][:-1], p["offsets"][1:]) if b > a]
        pts = p["points"][p["offsets"][0]:p["offsets"][-1]]
        _balls[(scene, kind)] = (cKDTree(pts), pts, np.concatenate(r) if r else np.zeros(0))
    return _balls[(scene, kind)]


def phase_and_pair(d: dict, z, label) -> str:
    """Where an unfinished run stopped (see the module docstring of eval_collisions.py and the stage-1 report) and the
    pair the filter acted on most often since."""
    r, gt, rec = d["result"], d["ground_truth"], d["receipts"]
    of_map = d["id_bridge"]["oracle_of"]
    node = {n["node_id"]: n for n in d["oracle_graph"]["nodes"]}
    seg = np.array([g["segment"] for g in gt])
    if r["segments_done"] == 0:
        phase, since = "driving", 0
    else:
        t0, names = int(np.argmax(seg == 1)), [str(x) for x in z["object_names"]]
        target = next((p for p in r["dropped_pairs"] if p["role"] not in ("object", "support", "container", "plate")), None)
        if "obj" in names:  # pick-and-place
            p = z["object_pos"][:, names.index("obj")]
            lifted = np.r_[np.zeros(t0, bool), p[t0:, 2] - p[t0, 2] > 0.03]
            if "container" in names:
                near = np.linalg.norm(p[:, :2] - z["object_pos"][t0, names.index("container"), :2], axis=1) < 0.10
            elif target is None:
                near = np.zeros(len(p), bool)
            else:
                c, h = np.array(node[target["object"][2:]]["center"]), np.array(node[target["object"][2:]]["half_extents"])
                near = np.all(np.abs(p[:, :2] - c[:2]) <= h[:2], axis=1)
            if not lifted.any():
                phase, since = f"reaching the pick (object moved {100 * np.linalg.norm(p[t0:] - p[t0], axis=1).max():.0f} cm)", t0
            elif not (lifted & near).any():
                phase, since = "carrying", int(np.argmax(lifted))
            else:
                phase, since = "reaching the drop", int(np.argmax(lifted & near))
        elif target is not None:
            joints = [k for k, x in enumerate(z["joint_names"]) if str(x).startswith(target["name"])]
            qv = z["qpos"][:, z["joint_qposadr"][joints]]
            move = np.abs(qv - qv[t0]).max(axis=1)
            moved = np.r_[np.zeros(t0, bool), move[t0:] > 0.02]
            phase, since = (f"moving it (joint moved {move[t0:].max():.2f})", int(np.argmax(moved))) if moved.any() else ("reaching the fixture", t0)
        else:
            phase, since = "manipulation", t0
    act = [(x, g) for x, g in zip(rec[since:], gt[since:]) if x["outcome"] not in ("passthrough", "passive") and x["h_min_pair"]]
    if not act:
        return f"{phase}; the filter did not act after that"
    key = lambda x: (robot_group(x["h_min_pair"][1]), label(of_map.get(x["h_min_pair"][2], x["h_min_pair"][2])))  # noqa: E731
    top, k = Counter(key(x) for x, _ in act).most_common(1)[0]
    hs = [x["h_min"] for x, _ in act if key(x) == top]
    cl = [g["clearance"] for x, g in act if key(x) == top and g["clearance"] is not None]
    return (f"{phase}; filter active on {100 * len(act) / max(len(rec) - since, 1):.0f}% of the ticks since, mostly {top[0]} > {top[1]} ({k} ticks): "
            f"margin {100 * np.median(hs):.1f} cm, true clearance {100 * np.median(cl):.1f} cm" if cl else f"{phase}; mostly {top[0]} > {top[1]} ({k} ticks)")


def load_runs(scenes: list[str]) -> tuple[list[dict], list[dict]]:
    runs, contacts = [], []
    for scene in scenes:
        for method, kind, title in ROWS:
            path = EVAL_DIR / "runs" / scene / f"{method}_{kind}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            z = np.load(path.with_suffix(".npz"))
            r, gt = d["result"], d["ground_truth"]
            node = {n["node_id"]: n for n in d["oracle_graph"]["nodes"]}
            name = {v: k.split(":", 1)[1] for k, v in d["oracle_lookup"]["node_of"].items()}
            label = lambda o, node=node, name=name: f"{node[o]['category']} ({name.get(o, o)})" if o in node else str(o)  # noqa: E731
            ob = r.get("safety_obstacle") or {}
            ob_id = ob.get("oracle_id")
            # was the obstacle admitted (rows) in the manipulation segment? (its pruning is not forced; listed if not)
            admitted = None
            if ob_id and r["filter"]:
                maps_of = {o: [h for h, oo in d["id_bridge"]["oracle_of"].items() if oo == o] for o in [ob_id[2:]]}
                b = d["bundles"][-1]
                ids = maps_of[ob_id[2:]]
                admitted = ("not in the map" if not ids else "admitted" if set(ids) & set(b["admitted_ids"]) else
                            f"labelled {sorted({b['labels'].get(i) for i in ids if i in b['labels']})}" if set(ids) & set(b["candidate_ids"]) else "outside the segment's reach")
            # contacts: designated obstacle, other mapped geometry, unmapped geometry
            tree, pts, rho = map_balls(scene, kind)
            per = Counter()
            for g in gt:
                first = {}
                for grp, o, *pos in g.get("contact_points", []):
                    first.setdefault((grp, o), pos)
                for (grp, o), pos in first.items():
                    if o == ob_id:
                        where = "designated obstacle"
                    else:
                        near = tree.query_ball_point(pos, float(rho.max())) if len(rho) else []
                        where = "other mapped geometry" if near and np.any(np.linalg.norm(pts[near] - pos, axis=1) <= rho[near]) else "unmapped geometry"
                    per[(grp, label(o[2:]), where)] += 1
            for (grp, o, where), n in sorted(per.items()):
                contacts.append({"scene": scene, "row": title, "robot_group": grp, "object": o, "geometry": where, "ticks": n})
            ob_ticks = int(ob.get("contact_ticks") or 0)
            runs.append({
                "scene": scene, "instruction": r["instruction"], "method": method, "map_kind": kind, "row": title, "item": ITEM[kind],
                "obstacle": bool(ob.get("placed")), "obstacle_category": ob.get("category"), "obstacle_admitted": admitted,
                "success": r["success"], "obstacle_contact_free": ob_ticks == 0, "collision_free": r["collision_free"],
                "safe_success_obstacle": bool(r["success"] and ob_ticks == 0), "safe_success": r["safe_success"],
                "obstacle_contact_ticks": ob_ticks, "collision_ticks": r["contact_ticks"], "clearance_obstacle_m": ob.get("clearance_min_m"),
                "clearance_m": r["true_clearance_min"], "negative_barrier_ticks": r["ticks_h_negative"], "steps": r["steps"],
                "task_time_s": r["steps"] * CONTROL_DT_S if r["success"] else None, "qp_ms_median": r["solve_ms_median"] if r["filter"] else None,
                "rows_mean": r["n_rows_qp_mean"] if r["filter"] else None, "relaxed_ticks": r["feasibility"]["relaxed_ticks"],
                "unfinished": None if r["success"] or method != "proposed" else phase_and_pair(d, z, label)})
    return runs, contacts


def table(runs: list[dict], scenes: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """Rows of the complete items; scenes left out (a run crashed twice); items not complete."""
    expected = {(s, m, k) for s in scenes for m, k, _ in ROWS}
    have = {(x["scene"], x["method"], x["map_kind"]) for x in runs}
    failed, crashed, done_items, out = _failed_twice(), set(), [], []
    for i in ("a", "b"):  # an item is complete when each of its runs is there or crashed twice
        miss = {(s, m, k) for s, m, k in expected if ITEM[k] == i and (s, m, k) not in have}
        if miss <= failed:
            done_items.append(i)
            crashed |= {s for s, _, _ in miss}
    keep = [s for s in scenes if s not in crashed]
    by = {(x["scene"], x["method"], x["map_kind"]): x for x in runs}
    prop = {s: by.get((s, "proposed", "full")) for s in keep}
    for method, kind, title in ROWS:
        if ITEM[kind] not in done_items:
            continue
        rs = [by[(s, method, kind)] for s in keep]
        n = len(rs)
        row = {"row": title, "runs": n}
        for key in ("obstacle_contact_free", "collision_free", "success", "safe_success_obstacle", "safe_success"):
            k = sum(x[key] for x in rs)
            row[key] = k
            row[key + "_ci95"] = wilson(k, n)
            if not (method == "proposed" and kind == "full") and "a" in done_items:
                a, b, p = mcnemar([prop[s][key] for s in keep], [x[key] for x in rs])
                row[key + "_vs_proposed"] = f"{a}:{b}, p={p:.3g}"
        row.update(clearance_obstacle_m_median=med([x["clearance_obstacle_m"] for x in rs]), clearance_m_median=med([x["clearance_m"] for x in rs]),
                   runs_negative_barrier=sum(x["negative_barrier_ticks"] > 0 for x in rs), negative_barrier_ticks=sum(x["negative_barrier_ticks"] for x in rs),
                   task_time_s_median=med([x["task_time_s"] for x in rs]), qp_ms_median=med([x["qp_ms_median"] for x in rs]), rows_median=med([x["rows_mean"] for x in rs]))
        out.append(row)
    return out, sorted(crashed), [i for i in ("a", "b") if i not in done_items]


def _failed_twice() -> set:
    out = set()
    path, since = EVAL_DIR / "repeats.csv", R / "frozen_v2.json"
    if path.exists() and since.exists():
        from datetime import datetime, timezone
        t0 = datetime.fromtimestamp(since.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        rows = list(csv.DictReader(open(path)))
        for x in rows:
            if x["utc"] >= t0 and x["attempt"] == "2" and not x["reason"].startswith("[tuning"):
                out.add((x["scene"], x["method"], x["map"]))
    return out


def write(name: str, rows: list[dict]) -> None:
    if rows:
        with open(R / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> list[str]:
    out = ["| " + " | ".join(t for _, t in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(f(r)) if callable(f) else str(r.get(f, "")) for f, _ in cols) + " |")
    return out


cm = lambda v: "-" if v is None else f"{100 * v:.1f} cm"  # noqa: E731
frac = lambda key: (lambda r: f"{r[key]}/{r['runs']} ({r[key + '_ci95']})")  # noqa: E731
vs = lambda key: (lambda r: r.get(key + "_vs_proposed", "-"))  # noqa: E731

scenes24 = [t["scene"] for t in tests]
runs24, contacts24 = load_runs(scenes24)
tab24, crashed24, missing24 = table(runs24, scenes24)
write("table2.csv", tab24)
write("runs.csv", runs24)
write("contacts.csv", contacts24)
def by_instruction(runs: list[dict], crashed: list[str]) -> list[dict]:
    inst = []
    for method, kind, title in ROWS:
        for ins in sorted({x["instruction"] for x in runs}):
            rs = [x for x in runs if x["row"] == title and x["instruction"] == ins and x["scene"] not in crashed]
            if rs:
                inst.append({"instruction": ins, "row": title, "runs": len(rs), "success": sum(x["success"] for x in rs), "safe_success": sum(x["safe_success"] for x in rs),
                             "obstacle_contact_free": sum(x["obstacle_contact_free"] for x in rs), "collision_free": sum(x["collision_free"] for x in rs)})
    return inst


inst = by_instruction(runs24, crashed24)
write("table2_by_instruction.csv", inst)
# the 12 scenes added by extension_rule.md, and all 36 (the 24 above stay as they were)
ext_path = R / "test_scenes_extension.json"
scenes12 = [t["scene"] for t in json.loads(ext_path.read_text())["test_scenes"]] if ext_path.exists() else []
if scenes12:
    runs12, contacts12 = load_runs(scenes12)
    tab12, crashed12, missing12 = table(runs12, scenes12)
    runs36, contacts36 = runs24 + runs12, contacts24 + contacts12
    tab36, crashed36, missing36 = table(runs36, scenes24 + scenes12)
    inst36 = by_instruction(runs36, crashed36)
    for name, rows_ in (("table2_extension.csv", tab12), ("table2_36.csv", tab36), ("runs_36.csv", runs36), ("contacts_36.csv", contacts36), ("table2_by_instruction_36.csv", inst36)):
        write(name, rows_)
else:
    runs36, contacts36, inst36 = runs24, contacts24, inst

L = ["# Table II campaign with Xiaomi-Robotics-1 (2026-11-17)", ""]
notes = R / "notes.md"
if notes.exists():
    L += [notes.read_text().strip(), ""]
L += ["## Table II, 24 scenes", "", f"Complete items: {', '.join(i for i in 'ab' if i not in missing24) or 'none'}"
      + (f"; not complete: {', '.join(missing24)}" if missing24 else "") + (f"; scenes left out (a run crashed twice): {', '.join(crashed24)}" if crashed24 else "") + ".", ""]
COLS = [("row", "Row"), (frac("obstacle_contact_free"), "No obstacle contact"), (frac("collision_free"), "No contact"), (frac("success"), "Task success"),
                      (frac("safe_success_obstacle"), "Safe success (obstacle)"), (frac("safe_success"), "Safe success"),
                      (lambda r: cm(r["clearance_obstacle_m_median"]), "Median clearance to the obstacle"), (lambda r: cm(r["clearance_m_median"]), "Median clearance"),
                      (lambda r: f"{r['runs_negative_barrier']} ({r['negative_barrier_ticks']} ticks)", "Runs with h < 0 (ticks)"),
                      (lambda r: "-" if r["task_time_s_median"] is None else f"{r['task_time_s_median']:.1f} s", "Median task time"),
                      (lambda r: "-" if r["qp_ms_median"] is None else f"{r['qp_ms_median']:.2f} ms", "Median QP time"),
                      (lambda r: "-" if r["rows_median"] is None else f"{r['rows_median']:.0f}", "Median rows")]
VS = [("row", "Row"), (vs("obstacle_contact_free"), "No obstacle contact"), (vs("collision_free"), "No contact"),
      (vs("success"), "Task success"), (vs("safe_success_obstacle"), "Safe success (obstacle)"), (vs("safe_success"), "Safe success")]
MCNEMAR = "Exact McNemar test, Proposed (full-pass map) against each row on the same scenes: scenes where only Proposed has the outcome : scenes where only the other row has it, p."
L += md_table(tab24, COLS)
L += ["", MCNEMAR, ""]
L += md_table([r for r in tab24 if r["row"] != "Proposed"], VS)
first = R0 / "table2_first_campaign.csv"
if first.exists():
    old = list(csv.DictReader(open(first)))
    new = {r["row"]: r for r in tab24}
    L += ["", "## The first Table II (pi0.5, 5 cm maps, no obstacle) next to the new one (24 scenes)", "",
          "| Row | First: no contact | First: success | First: safe success | First: median clearance | New: no contact | New: success | New: safe success | New: median clearance |", "|---|---|---|---|---|---|---|---|---|"]
    for o in old:
        n = new.get(o["method"], {})
        L.append(f"| {o['method']} | {o['collision_free']}/{o['runs']} | {o['success']}/{o['runs']} | {o['safe_success']}/{o['runs']} | {cm(float(o['smallest_true_clearance_m_median']) if o['smallest_true_clearance_m_median'] else None)} | "
                 + (f"{n['collision_free']}/{n['runs']} | {n['success']}/{n['runs']} | {n['safe_success']}/{n['runs']} | {cm(n['clearance_m_median'])} |" if n else "not run | | | |"))
def instruction_md(inst: list[dict], n: int) -> list[str]:
    ins_ = sorted({x["instruction"] for x in inst})
    out = ["", f"## Task success per instruction ({n} scenes; done / runs, safe successes, runs without obstacle contact)", "", "| Row | " + " | ".join(ins_) + " |", "|---|" + "---|" * len(ins_)]
    for _, _, title in ROWS:
        c = {x["instruction"]: x for x in inst if x["row"] == title}
        if c:
            out.append(f"| {title} | " + " | ".join(f"{c[i]['success']}/{c[i]['runs']} done, {c[i]['safe_success']} safe ({c[i]['obstacle_contact_free']} no obstacle contact)" if i in c else "-" for i in ins_) + " |")
    return out


L += instruction_md(inst, 24)
if scenes12:
    for n, tab_, crashed_, missing_, rule in ((12, tab12, crashed12, missing12, "the 12 scenes added by extension_rule.md (the selection rule of the first 24 continued, frozen settings)"),
                                              (36, tab36, crashed36, missing36, "the first 24 and the 12 added scenes together")):
        L += ["", f"## Table II, {n} scenes: {rule}", "", f"Complete items: {', '.join(i for i in 'ab' if i not in missing_) or 'none'}"
              + (f"; not complete: {', '.join(missing_)}" if missing_ else "") + (f"; scenes left out (a run crashed twice): {', '.join(crashed_)}" if crashed_ else "") + ".", ""]
        L += md_table(tab_, COLS) + ["", MCNEMAR, ""] + md_table([r for r in tab_ if r["row"] != "Proposed"], VS)
    L += instruction_md(inst36, 36)
N_ALL = 24 + len(scenes12)
L += ["", f"## Contacts by robot group and object ({N_ALL} scenes; ticks with a counted contact; per run)", ""]
for title in [t for _, _, t in ROWS]:
    cs = [c for c in contacts36 if c["row"] == title]
    if not cs:
        continue
    split = Counter()
    for c in cs:
        split[c["geometry"]] += c["ticks"]
    L.append(f"- **{title}**: " + ", ".join(f"{k} {v} ticks" for k, v in sorted(split.items())) + f" ({len({c['scene'] for c in cs})} scenes)")
    for c in cs:
        L.append(f"  - {c['scene']}: {c['robot_group']} > {c['object']}, {c['geometry']}, {c['ticks']} ticks")
L += ["", f"## Unfinished Proposed runs: phase and active pair ({N_ALL} scenes)", ""]
for x in runs36:
    if x["unfinished"]:
        L.append(f"- {x['scene']} ({'look-around' if x['map_kind'] == 'look' else 'full-pass'} map): {x['unfinished']}")
L += ["", "## The designated obstacle", ""]
obs = [json.loads(line) for f in ("obstacles.jsonl", "obstacles_extension.jsonl") if (R / f).exists() for line in open(R / f)]
for o in obs:
    ob = o.get("obstacle") or {}
    L.append(f"- {o['scene']}: " + (f"{ob['category']}, {100 * ob['height_m']:.0f} cm high, {100 * ob['depth_m']:.1f} cm into Nominal's path (drawn {100 * ob['depth_drawn_m']:.1f}), {100 * ob['gap_to_target_m']:.1f} cm from the object, map top {100 * ob['mapped_top_below_true_m']:.1f} cm below the true top"
                                    if ob.get("placed") else f"no obstacle: {ob.get('reason') or o.get('reason')}") + (f" [{o['status']}: {o.get('reason')}]" if o.get("status") != "ok" else ""))
not_adm = [x for x in runs36 if x["obstacle_admitted"] not in (None, "admitted")]
L += ["", "Runs in which the obstacle had no rows in the manipulation segment (its pruning is not forced):", ""]
L += [f"- {x['scene']} {x['row']}: {x['obstacle_admitted']}" for x in not_adm] or ["- none"]
(EVAL_DIR / "REPORT.md").write_text("\n".join(L) + "\n")
print(f"wrote {EVAL_DIR / 'REPORT.md'}: {len(runs36)} runs ({len(runs24)} on the first 24 scenes), items complete: {[i for i in 'ab' if i not in missing24]}")
