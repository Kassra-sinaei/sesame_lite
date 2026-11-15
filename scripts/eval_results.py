"""Result files of EVAL_SPEC.md, section 10 (and the video manifest of section 9) from the counted runs.

    python -s scripts/eval_results.py

Writes to outputs/eval/results/: runs.csv, table2.csv, videos.csv, text_numbers.json (map_study.csv comes from the
map study). Symbols carry the paper's names: beta_j (the code's kappa), kappa = 1/tau (the code's tau_m).
"""

import csv
import json

import numpy as np

from sesame_lite.eval_scene import EVAL_DIR, FOLLOWER_LOOKAHEAD_M, RULE
from sesame_lite.filter import CONTROL_DT_S
from sesame_lite.tasks import EVAL_COUNTER_CATEGORIES, EVAL_FLOOR_CATEGORIES

R = EVAL_DIR / "results"
ROWS = [("nominal", "full", "Nominal"), ("ee", "full", "End-effector filter"), ("fixed", "full", "Fixed margin"), ("noissf", "full", "Proposed without ISSf"),
        ("proposed", "full", "Proposed"), ("fixed", "look", "Fixed margin, look-around map"), ("proposed", "look", "Proposed, look-around map")]
tests = json.loads((R / "test_scenes.json").read_text())["test_scenes"]
tuning = {t["scene"]: t["used_for_tuning"] for t in tests}


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


runs = []
for path in sorted((EVAL_DIR / "runs").glob("*/*.json")):
    if path.name.endswith(".dsg.json") or path.parent.name not in tuning:
        continue
    r = json.loads(path.read_text())["result"]
    d, pol = r["disturbance"] or {}, r["policy"]
    success_tick = r["steps"] if r["success"] else None
    runs.append({
        "scene": r["scene"], "instruction": r["instruction"], "method": r["method"], "map_kind": r["map"], "seed": r["seed"], "snapshot_id": r["snapshot_id"],
        "used_for_tuning": tuning[r["scene"]], "outcome": r["outcome"], "success": r["success"], "collision_free": r["collision_free"], "safe_success": r["safe_success"],
        "collision_ticks": r["contact_ticks"], "collided_with": " ".join(r["contact_nodes"]), "contacts_not_admitted": " ".join(r["contacts_not_admitted"]),
        "collisions_by_robot_part": json.dumps({k: v - r["exempt_contacts_by_pair"].get(k, 0) for k, v in r["contacts_by_pair"].items() if v - r["exempt_contacts_by_pair"].get(k, 0) > 0}),
        "exempt_pair_contact_ticks": r["exempt_pair_contact_ticks"], "dropped_pairs": " ".join(sorted({f"{p['group']}>{p['category']}" for p in r["dropped_pairs"]})),
        "smallest_true_clearance_m": r["true_clearance_min"], "steps": r["steps"],
        "task_time_s": None if success_tick is None else success_tick * CONTROL_DT_S, "qp_time_ms_median": r["solve_ms_median"], "qp_time_ms_max": r["solve_ms_max"],
        "rows_mean": r["n_rows_qp_mean"], "rows_max": r["n_rows_qp_max"], "pairs_mean": r["n_pairs_mean"], "relaxed_ticks": r["feasibility"]["relaxed_ticks"],
        "relaxed_share": r["relaxed_share"], "tracking_disturbance_p99": (d.get("norm") or {}).get("99%"), "tracking_error_max": r["tracking_error_max"],
        "admitted_objects": " ".join(str(n) for n in r["n_admitted"]), "unmapped_task_objects": " ".join(r["unmapped_task_objects"]),
        "beta_j_sigma_bar_quantiles_m": json.dumps([b["kappa_s_bar_quantiles_m"] for b in r["balls"]]), "kappa_per_m": 1.0 / pol["tau_m"], "alpha": pol["alpha"],
        "d_bar": pol["d_bar"], "delta": pol["delta"], "follower_lookahead_m": r.get("follower_lookahead_m"), "wall_s": round(r["wall_s"], 1)})


def write(name, rows):
    with open(R / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if runs:
    write("runs.csv", runs)
    table = []
    for method, kind, title in ROWS:
        rs = [x for x in runs if x["method"] == method and x["map_kind"] == kind]
        table.append({"method": title, "runs": len(rs), "collision_free": sum(x["collision_free"] for x in rs), "success": sum(x["success"] for x in rs),
                      "safe_success": sum(x["safe_success"] for x in rs), "smallest_true_clearance_m_median": med([x["smallest_true_clearance_m"] for x in rs]),
                      "task_time_s_median": med([x["task_time_s"] for x in rs]), "qp_time_ms_median": med([x["qp_time_ms_median"] for x in rs]),
                      "rows_median": med([x["rows_mean"] for x in rs])})
    write("table2.csv", table)
videos = []
for path in sorted((EVAL_DIR / "videos").glob("*/*_overlay.mp4")):
    scene, stem = path.parent.name, path.name[: -len("_overlay.mp4")]
    run = next((x for x in runs if x["scene"] == scene and f"{x['method']}_{x['map_kind']}" == stem), None)
    for kind in ("clean", "overlay", "robot_camera"):
        videos.append({"video": str((path.parent / f"{stem}_{kind}.mp4").relative_to(EVAL_DIR)), "scene": scene, "method": stem, "view": kind,
                       "outcome": run and run["outcome"], "collision": None if run is None else not run["collision_free"]})
if videos:
    write("videos.csv", videos)
numbers = {
    "tracking_error_max": max((x["tracking_error_max"] for x in runs if x["tracking_error_max"] is not None), default=None),
    "rho_fixed_m": json.loads((R / "rho_fixed.json").read_text())["rho_fixed_m"],
    "clutter_rule": {"floor_categories": EVAL_FLOOR_CATEGORIES, "counter_categories": EVAL_COUNTER_CATEGORIES, "rule": RULE},
    "exempt_robot_part": "fingers, hand and wrist flange (link 7)", "follower_lookahead_m": FOLLOWER_LOOKAHEAD_M,
    "test_scenes": [{k: t[k] for k in ("scene", "kitchen", "used_for_tuning")} for t in tests],
}
replay = R / "point_rows_replay.json"
if replay.exists():
    numbers["one_row_per_point_replay"] = json.loads(replay.read_text())
(R / "text_numbers.json").write_text(json.dumps(numbers, indent=1))
print(f"runs {len(runs)}, videos {len(videos)}; wrote {', '.join(sorted(p.name for p in R.iterdir()))}")
