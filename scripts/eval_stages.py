"""The campaign lists of the stages (EVAL_SPEC.md, section 7) from outputs/eval/results/test_scenes.json.

    python scripts/eval_stages.py

Stage 1: every method once on the full-pass map of one scene (the first test scene, in the order of the file, that no
tuning round ran on). Stage 2: Nominal on the other scenes. Stage 3: Proposed on the other scenes. Stage 4: the three
other methods on the other scenes, and the two look-around rows on all scenes.
"""

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "outputs" / "eval"
tests = json.loads((EVAL / "results" / "test_scenes.json").read_text())["test_scenes"]
scenes = [t["scene"] for t in tests]
first = next(t["scene"] for t in tests if not t["used_for_tuning"])
others = [s for s in scenes if s != first]
lists = {
    "stage1": (f"every method once on the full-pass map of {first}", [f"{first} {m} full" for m in ("nominal", "ee", "fixed", "noissf", "proposed")]),
    "stage2": ("Nominal on the other scenes", [f"{s} nominal full" for s in others]),
    "stage3": ("Proposed on the other scenes", [f"{s} proposed full" for s in others]),
    "stage4": ("end-effector filter, fixed margin and Proposed without ISSf on the other scenes; fixed margin and Proposed on the look-around map of all scenes",
               [f"{s} {m} full" for m in ("ee", "fixed", "noissf") for s in others] + [f"{s} {m} look" for m in ("fixed", "proposed") for s in scenes]),
}
for name, (title, rows) in lists.items():
    (EVAL / "campaigns" / f"{name}.txt").write_text(f"# {name} (EVAL_SPEC.md 7): {title}\n" + "".join(f"{r}\n" for r in rows))
    print(name, len(rows))
print("total", sum(len(r) for _, r in lists.values()), "runs on", len(scenes), "scenes")
