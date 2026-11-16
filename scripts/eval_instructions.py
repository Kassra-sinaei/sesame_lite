"""The instructions of Table II by the rule of tuning_amendments.md, from outputs/eval/results/screening.csv.

    python -s scripts/eval_instructions.py

An instruction is in Table II if pi0.5 without the filter succeeds in at least 3 of its 6 screening scenes; if more
than six pass, the six with the most successes (ties: the instructions of the first set first, then the candidate
order). Writes outputs/eval/results/instructions.json.
"""

import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "outputs" / "eval" / "results"
FIRST_SET = ("open_drawer", "close_fridge", "close_cabinet", "sink_to_counter", "counter_to_cabinet", "counter_to_stove")
CANDIDATES = ("open_drawer", "close_fridge", "close_cabinet", "sink_to_counter", "counter_to_sink", "open_cabinet", "slide_dishwasher_rack",
              "counter_to_cabinet", "counter_to_stove", "stove_to_counter")
MIN_SUCCESS, N_SCENES, MAX_INSTRUCTIONS = 3, 6, 6
rows = [r for r in csv.DictReader(open(R / "screening.csv")) if r["built"] == "True"]
screen = {c: {"scenes": sum(r["candidate"] == c for r in rows), "success": sum(r["candidate"] == c and r["success"] == "True" for r in rows)} for c in CANDIDATES}
passing = [c for c in CANDIDATES if screen[c]["scenes"] == N_SCENES and screen[c]["success"] >= MIN_SUCCESS]
ranked = sorted(passing, key=lambda c: (-screen[c]["success"], c not in FIRST_SET, CANDIDATES.index(c)))[:MAX_INSTRUCTIONS]
table = [c for c in FIRST_SET if c in ranked] + [c for c in CANDIDATES if c in ranked and c not in FIRST_SET]
out = {"rule": f"in Table II if the unfiltered policy succeeds in at least {MIN_SUCCESS} of {N_SCENES} screening scenes; at most {MAX_INSTRUCTIONS}, by screening successes, "
               "ties: first set first, then candidate order", "screening": screen, "table2": table,
       "passed_but_not_taken": [c for c in passing if c not in table], "not_passed": [c for c in CANDIDATES if c not in passing]}
(R / "instructions.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
