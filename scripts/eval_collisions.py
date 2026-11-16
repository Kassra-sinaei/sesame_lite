"""Why each counted collision happened, read from the run logs (analysis only; changes no result).

    python -s scripts/eval_collisions.py

Writes outputs/eval/results/collisions.csv: one row per (run, robot part, object) with a counted contact, and prints the
runs per method and cause. The cause is decided at the first tick of the contact, by this fixed order:

  no filter                       the Nominal row
  object not in the map           no node of the map belongs to the object (it was not reconstructed)
  robot part without rows         End-effector filter, contact of the base or the arm
  outside the segment's reach     the object is in the map but not among the segment's candidates (A_g of eq. 20)
  not labelled obstacle           a candidate that the language model did not label an obstacle
  body outside its spheres        the object has rows and no robot sphere touches the true geometry at that tick
  spheres in contact, h > 0       a robot sphere touches the true geometry although the barrier on the map is positive
  barrier violated                the smallest barrier value of the tick is not positive
"""

import csv
import json
from collections import Counter

from sesame_lite.eval_scene import EVAL_DIR

R = EVAL_DIR / "results"
in_table = {t["scene"]: t.get("in_table2", True) for t in json.loads((R / "test_scenes.json").read_text())["test_scenes"]}
rows = []
for path in sorted((EVAL_DIR / "runs").glob("*/*.json")):
    if path.name.endswith(".dsg.json") or path.parent.name not in in_table:
        continue
    d = json.loads(path.read_text())
    r, gt, rec = d["result"], d["ground_truth"], d["receipts"]
    category = {n["node_id"]: n["category"] for n in d["oracle_graph"]["nodes"]}
    map_ids = {}  # oracle id -> the map's ids of the object
    for map_id, oracle_id in d["id_bridge"]["oracle_of"].items():
        map_ids.setdefault(oracle_id, []).append(map_id)
    bundle = {b["segment"]["index"]: b for b in d["bundles"]}
    kind = {s["index"]: s["kind"] for s in d["segments"]}
    counted = [[c for c in g["contacts"] if c not in g["exempt_pair_contacts"]] for g in gt]
    for pair, ticks in Counter(c for cs in counted for c in cs).items():
        part, obj = pair.split("|")
        oid = obj.split(":", 1)[1]
        t = next(i for i, cs in enumerate(counted) if pair in cs)
        g, x, b, ids = gt[t], rec[t], bundle.get(gt[t]["segment"]), map_ids.get(oid, [])
        if r["method"] == "nominal":
            cause = "no filter"
        elif not ids:
            cause = "object not in the map"
        elif r["method"] == "ee" and part in ("base", "arm"):
            cause = "robot part without rows"
        elif not set(ids) & set(b["candidate_ids"]):
            cause = "outside the segment's reach"
        elif not set(ids) & set(b["admitted_ids"]):
            cause = "not labelled obstacle"
        elif g["clearance"] is not None and g["clearance"] > 0:
            cause = "body outside its spheres"
        elif x["h_min"] is not None and x["h_min"] > 0:
            cause = "spheres in contact, h > 0"
        else:
            cause = "barrier violated"
        rows.append({"scene": r["scene"], "instruction": r["instruction"], "in_table2": in_table[r["scene"]], "method": r["method"], "map_kind": r["map"],
                     "robot_part": part, "object": category.get(oid, oid), "object_id": oid, "object_in_map": bool(ids), "collision_ticks": ticks, "first_tick": g["tick"],
                     "segment": kind[g["segment"]], "sphere_clearance_m": g["clearance"], "smallest_h_m": x["h_min"], "filter_outcome": x["outcome"], "cause": cause})
with open(R / "collisions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
runs = Counter((x["method"], x["map_kind"], x["cause"]) for x in {(x["scene"], x["method"], x["map_kind"], x["cause"]): x for x in rows if x["in_table2"]}.values())
for (method, map_kind, cause), n in sorted(runs.items()):
    print(f"{method:9s} {map_kind:5s} {cause:30s} {n} runs")
