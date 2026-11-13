"""The 20 test scenes (EVAL_SPEC.md, section 4) by a fixed rule, from the scenes prepared so far.

    python -s scripts/eval_select.py

Instructions in the order below; for each, seeds 0, 1, 2, ...: a seed is taken if its kitchen (layout, style) is not
excluded (outputs/eval/results/excluded_kitchens.json, the kitchens of every screening scene, the kitchens taken by
an earlier instruction or an earlier seed) and its scene could be built. The first 4 taken are the instruction's test
scenes; the first of them is its tuning scene (section 3). Prints the seeds still to prepare, or, once all 20 are
taken, writes outputs/eval/results/test_scenes.json.
"""

import json

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR

ORDER = ("open_drawer", "close_fridge", "close_cabinet", "sink_to_counter", "counter_to_sink")
PER_INSTRUCTION = 4
RULE = ("instructions in the order " + ", ".join(ORDER) + "; per instruction seeds 0, 1, 2, ...; a seed is taken if its kitchen is not in "
        "excluded_kitchens.json, not a screening kitchen, not taken by an earlier instruction or seed, and its scene could be built; "
        "the first 4 taken are the test scenes, the first of them the tuning scene")


def status(scene: str):
    """("ok", kitchen) | ("failed", kitchen, reason) | None (not prepared yet)."""
    d = SCENES_DIR / scene
    if (d / "scene.json").exists():
        k = json.loads((d / "scene.json").read_text())["kitchen"]
        return "ok", (k["layout"], k["style"]), ""
    if (d / "failed.json").exists():
        f = json.loads((d / "failed.json").read_text())
        return "failed", (f["kitchen"]["layout"], f["kitchen"]["style"]), f["reason"]
    return None


excluded = {(k["layout"], k["style"]) for k in json.loads((EVAL_DIR / "results" / "excluded_kitchens.json").read_text())["kitchens"]}
SCREENING = [f"{c}_s{s}" for c in ("counter_to_cabinet", "counter_to_sink", "counter_to_stove") for s in range(1000, 1006)]
excluded |= {status(name)[1] for name in SCREENING}
taken, rows, missing = set(), [], []
for instr in ORDER:
    got, seed = [], 0
    while len(got) < PER_INSTRUCTION:
        st = status(f"{instr}_s{seed}")
        if st is None:
            missing.append(f"{instr} {seed}")
            break
        why = "kitchen excluded" if st[1] in excluded else "kitchen taken earlier" if st[1] in taken else st[2] if st[0] == "failed" else ""
        rows.append({"instruction": instr, "seed": seed, "kitchen": list(st[1]), "taken": not why, "skipped_because": why})
        if not why:
            got.append(seed)
            taken.add(st[1])
        seed += 1
if missing:
    print("PREPARE " + " | ".join(missing))
else:
    test = [{"scene": f"{r['instruction']}_s{r['seed']}", **r, "used_for_tuning": r["seed"] == min(x["seed"] for x in rows if x["instruction"] == r["instruction"] and x["taken"])}
            for r in rows if r["taken"]]
    (EVAL_DIR / "results" / "test_scenes.json").write_text(json.dumps({"rule": RULE, "test_scenes": test, "considered": rows}, indent=1))
    for t in test:
        print(t["scene"], t["kitchen"], "tuning" if t["used_for_tuning"] else "")
