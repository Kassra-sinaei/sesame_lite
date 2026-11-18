"""One tuning round on one page (tuning procedure of 2026-11-16; not counted).

    python -s scripts/eval_tuning_report.py t0 [t1 ...]

For every run in outputs/eval_debug/tuning/<round>/: task done, ticks, counted contacts by robot group and object (on
the designated obstacle or not), the smallest true clearance and the filter's margin at it, the share of ticks the
filter changed the command while driving and while manipulating, how far the base moved during manipulation, ticks
with a negative barrier value, relaxed ticks, and for an unfinished run the phase and the pair the filter acted on most.
Writes outputs/eval_debug/tuning/<round>/report.md and prints it.
"""

import json
import sys
from collections import Counter

import numpy as np

from sesame_lite.eval_scene import DEBUG_DIR, robot_group


def label_fn(d):
    node = {n["node_id"]: n for n in d["oracle_graph"]["nodes"]}
    name = {v: k.split(":", 1)[1] for k, v in d["oracle_lookup"]["node_of"].items()}
    return lambda o: f"{node[o]['category']} ({name.get(o, o).replace('_group_1', '')})" if o in node else str(o)


def phase(d, z) -> tuple[str, int]:
    r, gt = d["result"], d["ground_truth"]
    node = {n["node_id"]: n for n in d["oracle_graph"]["nodes"]}
    seg = np.array([g["segment"] for g in gt])
    if r["segments_done"] == 0:
        return "driving", 0
    t0, names = int(np.argmax(seg == 1)), [str(x) for x in z["object_names"]]
    target = next((p for p in r["dropped_pairs"] if p["role"] not in ("object", "support", "container", "plate")), None)
    if "obj" in names:
        p = z["object_pos"][:, names.index("obj")]
        lifted = np.r_[np.zeros(t0, bool), p[t0:, 2] - p[t0, 2] > 0.03]
        if "container" in names:
            near = np.linalg.norm(p[:, :2] - z["object_pos"][t0, names.index("container"), :2], axis=1) < 0.10
        elif target is not None:
            c, h = np.array(node[target["object"][2:]]["center"]), np.array(node[target["object"][2:]]["half_extents"])
            near = np.all(np.abs(p[:, :2] - c[:2]) <= h[:2], axis=1)
        else:
            near = np.zeros(len(p), bool)
        if not lifted.any():
            return f"reaching the pick (object moved {100 * np.linalg.norm(p[t0:] - p[t0], axis=1).max():.0f} cm)", t0
        if not (lifted & near).any():
            return "carrying", int(np.argmax(lifted))
        return "reaching the drop", int(np.argmax(lifted & near))
    if target is not None:
        joints = [k for k, x in enumerate(z["joint_names"]) if str(x).startswith(target["name"])]
        qv = z["qpos"][:, z["joint_qposadr"][joints]]
        move = np.abs(qv - qv[t0]).max(axis=1)
        moved = np.r_[np.zeros(t0, bool), move[t0:] > 0.02]
        return (f"moving it (joint moved {move[t0:].max():.2f})", int(np.argmax(moved))) if moved.any() else ("reaching the fixture", t0)
    return "manipulation", t0


def run_line(path) -> str:
    d = json.loads(path.read_text())
    z = np.load(path.with_suffix(".npz"))
    r, gt, rec = d["result"], d["ground_truth"], d["receipts"]
    label, of_map = label_fn(d), d["id_bridge"]["oracle_of"]
    ob = (r.get("safety_obstacle") or {}).get("oracle_id")
    counted = Counter(c for g in gt for c in g["contacts"] if c not in g["exempt_pair_contacts"])
    contacts = "; ".join(f"{k.split('|')[0]}>{label(k.split(':', 1)[1])}{' [OBSTACLE]' if ob and k.endswith(ob) else ''} {v}" for k, v in counted.most_common()) or "none"
    seg = np.array([x["segment"] for x in rec])
    changed = np.array([x["outcome"] not in ("passthrough", "passive") for x in rec])
    share = lambda s: f"{100 * changed[seg == s].mean():.0f}%" if (seg == s).any() else "-"  # noqa: E731
    t0 = int(np.argmax(seg == 1)) if (seg == 1).any() else len(seg)
    base_move = float(np.linalg.norm(z["q"][t0:, :2] - z["q"][t0, :2], axis=1).max()) if t0 < len(seg) else 0.0
    t = r["true_clearance_min_tick"]
    close = "-"
    if t is not None and gt[t]["clearance_pair"]:
        link, o = gt[t]["clearance_pair"]
        mine = z["sphere_h"][t, [i for i, x in enumerate(d["sphere_links"]) if x == link]].min()
        close = f"{100 * r['true_clearance_min']:.1f} cm {robot_group(link)}>{label(o[2:])}, margin {100 * float(mine):.1f} cm" if np.isfinite(mine) else f"{100 * r['true_clearance_min']:.1f} cm {robot_group(link)}>{label(o[2:])}, no rows"
    why = ""
    if not r["success"]:
        ph, since = phase(d, z)
        act = [(x, g) for x, g in zip(rec[since:], gt[since:]) if x["outcome"] not in ("passthrough", "passive") and x["h_min_pair"]]
        why = ph
        if act:
            key = lambda x: (robot_group(x["h_min_pair"][1]), label(of_map.get(x["h_min_pair"][2], x["h_min_pair"][2])))  # noqa: E731
            top, k = Counter(key(x) for x, _ in act).most_common(1)[0]
            hs = [x["h_min"] for x, _ in act if key(x) == top]
            cl = [g["clearance"] for x, g in act if key(x) == top and g["clearance"] is not None]
            why += f"; held by {top[0]}>{top[1]} on {k} ticks, margin {100 * np.median(hs):.1f} cm, true {100 * np.median(cl):.1f} cm" if cl else f"; held by {top[0]}>{top[1]} on {k} ticks"
    so = r.get("safety_obstacle") or {}
    obst = "-" if not so else f"{so['contact_ticks']} ticks, closest {100 * so['clearance_min_m']:.1f} cm" if so.get("clearance_min_m") is not None else f"{so.get('contact_ticks')} ticks"
    return (f"| {d['result']['scene']} | {r['method']} | {'yes' if r['success'] else 'no'} | {r['steps']} | {obst} | {contacts} | {close} | {share(0)} / {share(1)} | {100 * base_move:.0f} cm | "
            f"{r['ticks_h_negative']} | {r['feasibility']['relaxed_ticks']} | {why} |")


for rnd in sys.argv[1:]:
    root = DEBUG_DIR / "tuning" / rnd
    lines = [f"# Tuning round {rnd}", "", "| Scene | Method | Done | Ticks | Obstacle: contact, closest | Counted contacts (ticks) | Closest approach | Filter active: driving / manipulation | Base moved in manipulation | h < 0 ticks | Relaxed ticks | If not done |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(root.glob("*/*_full.json")) + sorted(root.glob("*/*_look.json")):
        if not p.name.endswith(".dsg.json"):
            lines.append(run_line(p))
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
