"""Resumable campaign (EVAL_SPEC.md, section 3): run a list of (scene, method, map), a few processes at a time.

    python scripts/eval_campaign.py outputs/eval/campaigns/screening.txt -j 3

The list has one run per line: ``<scene> <method> <map>``. A run whose result file exists is skipped, so a stopped
campaign continues from the first unfinished run. A run that ends without a result (crash, out of memory) is repeated
once, and the repeat is logged in outputs/eval/repeats.csv with the reason; a second failure is logged and left.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "outputs" / "eval"

ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("-j", type=int, default=3, help="processes at once (pi0.5 needs ~11 GB of GPU memory each)")
args = ap.parse_args()
runs = [line.split() for line in Path(args.runs).read_text().splitlines() if line.strip() and not line.startswith("#")]


def result_path(scene, method, kind) -> Path:
    return EVAL / "runs" / scene / f"{method}_{kind}.json"


def log_repeat(run, attempt, reason) -> None:
    path = EVAL / "repeats.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "scene", "method", "map", "attempt", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), *run, attempt, reason])


def run_one(run) -> str:
    scene, method, kind = run
    if result_path(*run).exists():
        return f"done already  {' '.join(run)}"
    log = EVAL / "logs" / f"{scene}_{method}_{kind}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        t0 = time.time()
        with open(log, "a") as f:
            code = subprocess.call([sys.executable, "-s", str(REPO / "scripts" / "eval_run.py"), "--scene", scene, "--method", method, "--map", kind],
                                   stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env={**os.environ, "MUJOCO_GL": "egl"})
        if result_path(*run).exists():
            return f"finished      {' '.join(run)}  ({time.time() - t0:.0f} s)"
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        log_repeat(run, attempt, f"exit {code}: {tail[0][:200]}")
    return f"FAILED twice  {' '.join(run)}  (see {log})"


with ThreadPoolExecutor(args.j) as pool:
    for line in pool.map(run_one, runs):
        print(line, flush=True)
