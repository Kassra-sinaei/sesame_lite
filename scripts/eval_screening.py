"""Summary of the 5th-instruction screening (EVAL_SPEC.md, section 4): outputs/eval/results/screening.csv and a table.

    python -s scripts/eval_screening.py

One row per screening seed (per candidate the first 6 seeds from 1000 whose scene could be built; a seed whose scene could
not be built is listed with its reason).
Contacts are split by what was touched: the floor obstacles, the counter objects, the support surface of the picked
object (its exemption is an open choice), and everything else.
"""

import csv
import json
from collections import defaultdict

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR

CANDIDATES = ("counter_to_cabinet", "counter_to_stove", "stove_to_counter")
N_SCENES = 6  # per candidate: seeds 1000, 1001, ... until this many scenes could be built (skipped seeds are listed)

rows = []
for cand in CANDIDATES:
    built = 0
    for seed in range(1000, 1040):
        if built == N_SCENES:
            break
        scene = f"{cand}_s{seed}"
        row = {"candidate": cand, "seed": seed, "scene": scene}
        failed = SCENES_DIR / scene / "failed.json"
        if failed.exists():
            f = json.loads(failed.read_text())
            rows.append(row | {"kitchen": f"{f['kitchen']['layout']}-{f['kitchen']['style']}", "built": False, "note": f["reason"]})
            continue
        if not (SCENES_DIR / scene / "scene.json").exists():
            break
        built += 1
        sc = json.loads((SCENES_DIR / scene / "scene.json").read_text())
        run = json.loads((EVAL_DIR / "runs" / scene / "nominal_full.json").read_text())
        res, gt = run["result"], run["ground_truth"]
        names = {v: k for k, v in run["oracle_lookup"]["node_of"].items()}  # oracle node -> RoboCasa name
        kind = defaultdict(int)
        for g in gt:
            ks = set()
            for o in g["touched"]:
                name = names.get(o[2:], "")
                ks.add("floor_obstacle" if "clutter_floor" in name else "counter_object" if "clutter_counter" in name
                       else "support" if o == res.get("support_of_object") else "other")
            for k in ks:
                kind[k] += 1
        rows.append(row | {"kitchen": f"{sc['kitchen']['layout']}-{sc['kitchen']['style']}", "built": True, "language": sc["language"], "success": res["success"],
                           "steps": res["steps"], "outcome": res["outcome"], "contact_ticks": res["contact_ticks"],
                           "contact_ticks_floor_obstacles": kind["floor_obstacle"], "contact_ticks_counter_objects": kind["counter_object"],
                           "contact_ticks_support": kind["support"], "contact_ticks_other": kind["other"],
                           "contact_ticks_if_support_exempt": sum(1 for g in gt if any(o != res.get("support_of_object") for o in g["touched"])),
                           "path_length_m": sc["start"]["path_length_m"], "wall_s": round(res["wall_s"], 1)})
out = EVAL_DIR / "results" / "screening.csv"
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
for cand in CANDIDATES:
    rs = [r for r in rows if r["candidate"] == cand]
    built = [r for r in rs if r["built"]]
    print(f"{cand:20s} scenes built {len(built)}/{len(rs)}  success {sum(r['success'] for r in built)}/{len(built)}  "
          f"collision-free {sum(r['contact_ticks'] == 0 for r in built)}/{len(built)}")
print(f"wrote {out}")
