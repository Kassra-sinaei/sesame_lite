"""The test scenes (EVAL_SPEC.md, section 4; six instructions since 2026-11-16) by a fixed rule, from the scenes prepared.

    python -s scripts/eval_select.py

Instructions in the order below; for each, seeds 0, 1, 2, ...: a seed is taken if its kitchen (layout, style) is not
excluded (outputs/eval/results/excluded_kitchens.json, the kitchens of every screening scene, the kitchens taken by
an earlier instruction or an earlier seed) and its scene could be built. The first 4 taken are the instruction's test
scenes. A test scene whose kitchen a tuning round ran on is marked used_for_tuning. Prints the seeds still to prepare,
or, once every instruction has its 4 scenes, writes outputs/eval/results/test_scenes.json.
"""

import json

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR

FIRST_SET = ("open_drawer", "close_fridge", "close_cabinet", "sink_to_counter", "counter_to_cabinet", "counter_to_stove")  # Kasra, 2026-11-16
chosen = json.loads((EVAL_DIR / "results" / "instructions.json").read_text())["table2"] if (EVAL_DIR / "results" / "instructions.json").exists() else list(FIRST_SET)
ORDER = FIRST_SET + tuple(c for c in chosen if c not in FIRST_SET)  # instructions added by the screening rule come after the first set
# kitchens the tuning rounds 1-6 ran on: a test scene in one of them is marked used_for_tuning, wherever it comes in the order
TUNING_KITCHENS = {(34, 43), (51, 28), (53, 37), (44, 17), (55, 26), (32, 32)}  # (32, 32): the stage-1 scene of the first freeze, whose results led to changes
PER_INSTRUCTION = 4
RULE = ("instructions in the order " + ", ".join(ORDER) + "; per instruction seeds 0, 1, 2, ...; a seed is taken if its kitchen is not in "
        "excluded_kitchens.json, not a screening kitchen, not taken by an earlier instruction or seed, and its scene could be built; "
        "the first 4 taken are the test scenes; used_for_tuning marks every test scene whose kitchen a tuning round ran on; "
        "the scenes of the first set stay as selected on 2026-11-16, later instructions are selected after them; "
        "in_table2: the instruction passed the screening rule (instructions.json)")


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
SCREENING = [d.name for d in SCENES_DIR.iterdir() if int(d.name.rsplit("_s", 1)[1]) >= 1000 and status(d.name)]  # screening seeds start at 1000
out_path = EVAL_DIR / "results" / "test_scenes.json"
prev = json.loads(out_path.read_text()) if out_path.exists() else {"test_scenes": [], "considered": []}
taken, rows, missing = set(), [], []
for instr in ORDER:
    kept = [r for r in prev["considered"] if r["instruction"] == instr]
    if sum(r["taken"] for r in kept) == PER_INSTRUCTION:  # selected before: stays as it is
        rows += kept
        taken |= {tuple(r["kitchen"]) for r in kept if r["taken"]}
        continue
    if instr not in FIRST_SET:  # (the first set was selected before the later screening scenes existed)
        excluded |= {status(name)[1] for name in SCREENING}
    elif not prev["considered"]:
        excluded |= {status(name)[1] for name in SCREENING}
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
    test = [{"scene": f"{r['instruction']}_s{r['seed']}", **r, "used_for_tuning": tuple(r["kitchen"]) in TUNING_KITCHENS, "in_table2": r["instruction"] in chosen} for r in rows if r["taken"]]
    out_path.write_text(json.dumps({"rule": RULE, "test_scenes": test, "considered": rows}, indent=1))
    for t in test:
        print(t["scene"], t["kitchen"], "used for tuning" if t["used_for_tuning"] else "", "" if t["in_table2"] else "not in Table II")
