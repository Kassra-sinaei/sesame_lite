"""Aggregate episode receipts into the per-arm table (episode-level statistics, matched seeds).

    python scripts/summarize.py outputs/demo_nav [--csv table.csv]
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
from lerobot.utils.eval_stats import wilson_interval

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--csv", default=None)
args = ap.parse_args()

rows = [json.load(open(f))["result"] for f in sorted(glob.glob(os.path.join(args.out_dir, "*.json")))]
arms = sorted({r["arm"] for r in rows}, key=lambda a: ("A", "B", "B+", "C").index(a))

# matched scenes: every arm of a seed must have seen the same oracle graph
by_seed = collections.defaultdict(dict)
for r in rows:
    by_seed[(r["task"], r["seed"])][r["arm"]] = r["snapshot_id"]
mismatch = [k for k, v in by_seed.items() if len(set(v.values())) > 1]
if mismatch:
    print("WARNING: scene mismatch across arms for", mismatch)


def rate(rs, pred):
    n, k = len(rs), sum(1 for r in rs if pred(r))
    lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
    return f"{k}/{n} [{100 * lo:.0f}-{100 * hi:.0f}%]"


cols = [
    ("episodes", lambda rs: len(rs)),
    ("success (plan goal)", lambda rs: rate(rs, lambda r: r["success"])),
    ("success (RoboCasa check)", lambda rs: rate(rs, lambda r: r.get("task_success_robocasa", False))),
    ("success, no violation", lambda rs: rate(rs, lambda r: r["outcome"] == "success")),
    ("success but unsafe", lambda rs: rate(rs, lambda r: r["outcome"] == "success_but_unsafe")),
    ("unsafe fail", lambda rs: rate(rs, lambda r: r["outcome"] == "unsafe_fail")),
    ("safe fail", lambda rs: rate(rs, lambda r: r["outcome"] == "safe_fail")),
    ("stalled", lambda rs: rate(rs, lambda r: r["outcome"] == "stalled")),
    ("start violation", lambda rs: rate(rs, lambda r: r["start_violation"])),
    ("episodes w/ contact", lambda rs: rate(rs, lambda r: r["contact_ticks"] > 0)),
    ("worst penetration [cm]", lambda rs: f"{100 * max(r['penetration_m'] for r in rs):.1f}"),
    ("severity [m*s] (sum)", lambda rs: f"{sum(r['severity_ms'] for r in rs):.3f}"),
    ("steps (successes)", lambda rs: f"{np.mean([r['steps'] for r in rs if r['success']]):.0f}" if any(r["success"] for r in rs) else "-"),
    ("rows/tick", lambda rs: f"{np.mean([r['n_rows_mean'] for r in rs]):.0f}"),
    ("tick ms p50 / p99", lambda rs: f"{np.median([r['tick_ms_median'] for r in rs]):.1f} / {np.median([r['tick_ms_p99'] for r in rs]):.1f}"),
    ("holds / infeasible", lambda rs: f"{sum(r['holds'] for r in rs)} / {sum(r['infeasible'] for r in rs)}"),
    ("classifier s (mean)", lambda rs: f"{np.mean([x for r in rs for x in r['classifier_latency_s'] if x]):.1f}" if any(x for r in rs for x in r["classifier_latency_s"]) else "-"),
    ("classifier fallbacks", lambda rs: sum(r["classifier_fallback"] for r in rs)),
]
table = {name: {arm: fn([r for r in rows if r["arm"] == arm]) for arm in arms} for name, fn in cols}
width = max(len(n) for n, _ in cols)
print(f"{'':{width}s} " + " ".join(f"{a:>18s}" for a in arms))
for name, _ in cols:
    print(f"{name:{width}s} " + " ".join(f"{str(table[name][a]):>18s}" for a in arms))
if args.csv:
    with open(args.csv, "w") as f:
        f.write("metric," + ",".join(arms) + "\n")
        for name, _ in cols:
            f.write(name + "," + ",".join(str(table[name][a]) for a in arms) + "\n")
