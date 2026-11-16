"""Screening of the candidate instructions (EVAL_SPEC.md, section 4; rule in tuning_amendments.md): pi0.5 without the
filter on the first 6 eligible seeds from 1000 of every candidate.

    python -s scripts/eval_screening.py --need     seeds still to prepare ("<instruction> <seed>" per line; empty when complete)
    python -s scripts/eval_screening.py --list     writes outputs/eval/campaigns/screening.txt (the Nominal runs)
    python -s scripts/eval_screening.py            summary: outputs/eval/results/screening.csv and a table

A seed is eligible if its scene could be built and its kitchen is not the kitchen of a test scene of the first six
instructions (screening kitchens are never test scenes). Skipped seeds are listed with the reason.
"""

import csv
import json
import sys

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR

CANDIDATES = ("open_drawer", "close_fridge", "close_cabinet", "sink_to_counter", "counter_to_sink", "open_cabinet", "slide_dishwasher_rack",
              "counter_to_cabinet", "counter_to_stove", "stove_to_counter")  # the candidate list of tuning_amendments.md, in its order
N_SCENES = 6
LAST_SEED = 1060
tests = json.loads((EVAL_DIR / "results" / "test_scenes.json").read_text())["test_scenes"]
TEST_KITCHENS = {tuple(t["kitchen"]) for t in tests if t["seed"] < 1000}


def seeds(cand: str):
    """(rows of the seeds looked at, the eligible scenes, the first seed not prepared yet or None)."""
    rows, chosen = [], []
    for seed in range(1000, LAST_SEED):
        if len(chosen) == N_SCENES:
            return rows, chosen, None
        scene = f"{cand}_s{seed}"
        row = {"candidate": cand, "seed": seed, "scene": scene}
        if (SCENES_DIR / scene / "failed.json").exists():
            f = json.loads((SCENES_DIR / scene / "failed.json").read_text())
            rows.append(row | {"kitchen": f"{f['kitchen']['layout']}-{f['kitchen']['style']}", "built": False, "note": f["reason"]})
        elif (SCENES_DIR / scene / "scene.json").exists():
            k = json.loads((SCENES_DIR / scene / "scene.json").read_text())["kitchen"]
            if (k["layout"], k["style"]) in TEST_KITCHENS:
                rows.append(row | {"kitchen": f"{k['layout']}-{k['style']}", "built": False, "note": "kitchen of a test scene"})
            else:
                rows.append(row | {"kitchen": f"{k['layout']}-{k['style']}", "built": True})
                chosen.append(scene)
        else:
            return rows, chosen, seed
    return rows, chosen, None


if "--need" in sys.argv:
    for cand in CANDIDATES:
        _, chosen, first = seeds(cand)
        if first is not None:
            for k in range(N_SCENES - len(chosen)):
                if first + k < LAST_SEED:
                    print(cand, first + k)
    sys.exit(0)
if "--list" in sys.argv:
    lines = ["# screening of the candidate instructions (tuning_amendments.md): pi0.5 without the filter, the first 6 eligible seeds from 1000"]
    for cand in CANDIDATES:
        lines += [f"{s} nominal full" for s in seeds(cand)[1]]
    (EVAL_DIR / "campaigns" / "screening.txt").write_text("\n".join(lines) + "\n")
    print(len(lines) - 1, "runs")
    sys.exit(0)

rows = []
for cand in CANDIDATES:
    for row in seeds(cand)[0]:
        if row["built"]:
            sc = json.loads((SCENES_DIR / row["scene"] / "scene.json").read_text())
            res = json.loads((EVAL_DIR / "runs" / row["scene"] / "nominal_full.json").read_text())["result"]
            row |= {"language": sc["language"], "success": res["success"], "steps": res["steps"], "outcome": res["outcome"], "collision_ticks": res["contact_ticks"],
                    "path_length_m": sc["start"]["path_length_m"], "counter_objects": len(sc["counter_objects"]), "wall_s": round(res["wall_s"], 1)}
        rows.append(row)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(EVAL_DIR / "results" / "screening.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
for cand in CANDIDATES:
    built = [r for r in rows if r["candidate"] == cand and r["built"]]
    print(f"{cand:22s} scenes {len(built)}  success {sum(r['success'] for r in built)}/{len(built)}  seeds skipped {sum(r['candidate'] == cand and not r['built'] for r in rows)}")
