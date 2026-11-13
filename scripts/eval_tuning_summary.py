"""Summary of one tuning round (outputs/eval/tuning/<round>/): one row per run, as tuning_procedure.md lists.

    python -s scripts/eval_tuning_summary.py r1

Writes outputs/eval/results/tuning_<round>.csv and prints it. When a run stalled or ran out of ticks, the pairs that
bound the QP most often over its last 100 ticks are listed (filter-side ids with their category).
"""

import csv
import json
import sys
from collections import Counter

from sesame_lite.eval_scene import EVAL_DIR

rnd = sys.argv[1]
rows = []
for path in sorted((EVAL_DIR / "tuning" / rnd).glob("*/*.json")):
    if path.name.endswith(".dsg.json"):
        continue
    d = json.loads(path.read_text())
    r, rec = d["result"], d["receipts"]
    cats = {n["node_id"]: n["category"] for n in d["graph"]["nodes"]}
    last = rec[-100:]
    binding = Counter()
    for x in last:
        if x["outcome"] in ("filtered", "relaxed", "hold") and x["h_min_pair"] is not None:
            obj = x["h_min_pair"][2]
            binding[f"{x['h_min_pair'][1]} -> {cats.get(obj.split('~')[0], '?')} {obj}"] += 1
    seg_ticks = Counter(g["segment"] for g in d["ground_truth"])
    rows.append({"scene": r["scene"], "method": r["method"], "outcome": r["outcome"], "success": r["success"], "ticks": r["steps"],
                 "ticks_driving": seg_ticks.get(0, 0), "ticks_manipulation": seg_ticks.get(1, 0), "contact_ticks": r["contact_ticks"],
                 "touched": " ".join(r["contact_nodes"]), "true_clearance_min_m": None if r["true_clearance_min"] is None else round(r["true_clearance_min"], 4),
                 "relaxed_ticks": r["feasibility"]["relaxed_ticks"], "hold_ticks": r["feasibility"]["hold_ticks"], "finger_patch_contact_ticks": r.get("finger_patch_contact_ticks"),
                 "tracking_error_max": r["tracking_error_max"], "unmapped_task_objects": " ".join(r["unmapped_task_objects"]),
                 "binding_last_100_ticks": "; ".join(f"{k} ({n})" for k, n in binding.most_common(3)) if r["outcome"] != "success" else ""})
out = EVAL_DIR / "results" / f"tuning_{rnd}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for row in rows:
    print(" | ".join(f"{k}={v}" for k, v in row.items() if v not in ("", None)))
print(f"wrote {out}")
