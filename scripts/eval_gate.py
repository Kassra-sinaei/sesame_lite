"""The selection rule of outputs/eval/results/v2/selection_rule.md (steps 2-5) on finished tuning runs.

    python scripts/eval_gate.py ROUND.json OUT.json

ROUND.json: a list of candidates {"alpha", "d_bar", "arm", "tag"}; "tag" holds the Proposed, Proposed without ISSf and
Fixed margin runs of the 10 tuning scenes (outputs/eval_debug/tuning/<tag>/), "arm" is one of ARM. Nominal's runs of
an arm setting are in NOMINAL_TAG[arm]. Per candidate: safe successes of the three rows, Nominal's tasks done and its arm
tracking error (90th percentile over every tick of the 10 runs, rad/s); admissible: Nominal done >= 7 of 10; passes:
admissible and Proposed > both ablations. The choice: the largest lead of Proposed over the better ablation; ties: no
disturbance, then the smaller tracking error, then the smaller alpha. Writes OUT.json (all candidates and the choice,
null if none passes) and prints it.
"""

import json
import sys
from pathlib import Path

import numpy as np

from sesame_lite.eval_scene import DEBUG_DIR
from sesame_lite.filter import COMMANDED, CONTROL_DT_S

ARM = {"none": "", "kv40": "--arm-kv 40", "kv20": "--arm-kv 20", "a1": "--disturbance a1", "a2": "--disturbance a2"}
NOMINAL_TAG = {"none": "t1_xr1", "kv40": "arm_kv40", "kv20": "arm_kv20", "a1": "arm_a1", "a2": "arm_a2"}
SCENES = (Path(__file__).resolve().parents[1] / "outputs/eval/results/v2/tuning_scenes.txt").read_text().split()
T = DEBUG_DIR / "tuning"


def result(tag: str, scene: str, method: str) -> dict:
    p = T / tag / scene / f"{method}_full.json"
    if not p.exists():
        raise SystemExit(f"missing run {p}")
    return json.loads(p.read_text())


def nominal(arm: str) -> tuple[int, float]:
    done, err = 0, []
    for s in SCENES:
        d = result(NOMINAL_TAG[arm], s, "nominal")
        done += bool(d["result"]["success"])
        q = np.load(T / NOMINAL_TAG[arm] / s / "nominal_full.npz")["q"]
        u = np.array([x["u_applied"] for x in d["receipts"]])
        err.append(np.linalg.norm(((q[1:, COMMANDED] - q[:-1, COMMANDED]) / CONTROL_DT_S - u[:-1])[:, 3:], axis=1))
    return done, float(np.percentile(np.concatenate(err), 90))


cands = json.loads(Path(sys.argv[1]).read_text())
for c in cands:
    c["safe_success"] = {m: sum(bool(result(c["tag"], s, m)["result"]["safe_success"]) for s in SCENES) for m in ("proposed", "noissf", "fixed")}
    c["nominal_done"], c["arm_tracking_error_p90"] = nominal(c["arm"])
    c["admissible"] = c["nominal_done"] >= 7
    ss = c["safe_success"]
    c["lead"] = ss["proposed"] - max(ss["noissf"], ss["fixed"])
    c["passes"] = c["admissible"] and c["lead"] > 0
ok = [c for c in cands if c["passes"]]
choice = min(ok, key=lambda c: (-c["lead"], c["arm"] != "none", c["arm_tracking_error_p90"], c["alpha"])) if ok else None
out = {"candidates": cands, "choice": choice}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out))
