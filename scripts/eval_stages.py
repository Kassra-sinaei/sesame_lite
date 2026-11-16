"""The campaign lists from outputs/eval/results/test_scenes.json (EVAL_SPEC.md, section 7; order of 2026-11-16).

    python scripts/eval_stages.py

For the scenes of the Table II instructions (in_table2): table_main.txt (Nominal and Proposed on the full-pass map, scene by
scene), table_baselines.txt (end-effector filter, fixed margin, Proposed without ISSf) and table_look.txt (fixed margin
and Proposed on the look-around map). extra.txt: Nominal and Proposed on the scenes of the instructions that did not pass
the screening rule (reported outside Table II).
"""

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "outputs" / "eval"
tests = json.loads((EVAL / "results" / "test_scenes.json").read_text())["test_scenes"]
table = [t["scene"] for t in tests if t["in_table2"]]
extra = [t["scene"] for t in tests if not t["in_table2"]]
lists = {
    "table_main": ("Nominal and Proposed on the full-pass map", [f"{s} {m} full" for s in table for m in ("nominal", "proposed")]),
    "table_baselines": ("end-effector filter, fixed margin and Proposed without ISSf on the full-pass map", [f"{s} {m} full" for m in ("ee", "fixed", "noissf") for s in table]),
    "table_look": ("fixed margin and Proposed on the look-around map", [f"{s} {m} look" for m in ("fixed", "proposed") for s in table]),
    "extra": ("Nominal and Proposed on the scenes of the instructions outside Table II", [f"{s} {m} full" for s in extra for m in ("nominal", "proposed")]),
}
for old in (EVAL / "campaigns").glob("stage*.txt"):
    old.unlink()
for name, (title, rows) in lists.items():
    (EVAL / "campaigns" / f"{name}.txt").write_text(f"# {title}\n" + "".join(f"{r}\n" for r in rows))
    print(name, len(rows))
print("Table II:", len(table), "scenes,", 7 * len(table), "runs")
