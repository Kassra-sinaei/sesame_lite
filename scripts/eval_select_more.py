"""Two more test scenes per Table II instruction (Kasra, 2026-11-17: Table II from 24 to 36 scenes), by the rule of
scripts/eval_select.py continued: outputs/eval/results/v2/test_scenes_extension.json.

    python -s scripts/eval_select_more.py

For each Table II instruction, in the order of eval_select.py, seeds after the last seed the first selection considered
(test_scenes.json), in increasing order: a seed is taken if its kitchen is not in excluded_kitchens.json, not a kitchen
of any screening or tuning scene (seeds >= 1000), not taken by the first selection or by an earlier take of this one,
and its scene could be built. The first 2 taken per instruction are the new scenes. Prints the seeds still to prepare
(PREPARE ...) or writes the file. The first 24 scenes are not touched.
"""

import json

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR

MORE = 2
RULE = ("per Table II instruction (order of eval_select.py), seeds after the last seed considered by test_scenes.json in increasing order; "
        "a seed is taken if its kitchen is not in excluded_kitchens.json, not a kitchen of any scene with seed >= 1000 (screening and tuning), "
        "not taken by test_scenes.json or an earlier take of this rule, and its scene could be built (scripts/eval_prepare.py --voxel 0.05, "
        "as the first 24); the first 2 taken are the new scenes")


def status(scene: str):
    d = SCENES_DIR / scene
    if (d / "scene.json").exists():
        k = json.loads((d / "scene.json").read_text())["kitchen"]
        return "ok", (k["layout"], k["style"]), ""
    if (d / "failed.json").exists():
        f = json.loads((d / "failed.json").read_text())
        return "failed", (f["kitchen"]["layout"], f["kitchen"]["style"]), f["reason"]
    return None


first = json.loads((EVAL_DIR / "results" / "test_scenes.json").read_text())
order = list(dict.fromkeys(r["instruction"] for r in first["considered"]))
table2 = [i for i in order if any(t["instruction"] == i and t["in_table2"] for t in first["test_scenes"])]
excluded = {(k["layout"], k["style"]) for k in json.loads((EVAL_DIR / "results" / "excluded_kitchens.json").read_text())["kitchens"]}
excluded |= {st[1] for d in SCENES_DIR.iterdir() if int(d.name.rsplit("_s", 1)[1]) >= 1000 and (st := status(d.name))}
taken = {tuple(r["kitchen"]) for r in first["considered"] if r["taken"]}
rows, missing = [], []
for instr in table2:
    seed = max(r["seed"] for r in first["considered"] if r["instruction"] == instr) + 1
    got = 0
    while got < MORE:
        st = status(f"{instr}_s{seed}")
        if st is None:
            missing.append(f"{instr} {seed}")
            break
        why = "kitchen excluded" if st[1] in excluded else "kitchen taken earlier" if st[1] in taken else st[2] if st[0] == "failed" else ""
        rows.append({"instruction": instr, "seed": seed, "kitchen": list(st[1]), "taken": not why, "skipped_because": why})
        if not why:
            got += 1
            taken.add(st[1])
        seed += 1
if missing:
    print("PREPARE " + " | ".join(missing))
else:
    test = [{"scene": f"{r['instruction']}_s{r['seed']}", **r, "in_table2": True} for r in rows if r["taken"]]
    out = EVAL_DIR / "results" / "v2" / "test_scenes_extension.json"
    out.write_text(json.dumps({"rule": RULE, "test_scenes": test, "considered": rows}, indent=1))
    for t in test:
        print("SELECTED", t["scene"], t["kitchen"])
