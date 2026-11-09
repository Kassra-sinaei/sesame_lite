"""Explain one recorded episode: what the safety filter did and which constraint held the robot back.

    .venv/bin/python -s scripts/diagnose_episode.py outputs/arm_c/OpenDrawer_s0_C.json

Writes <episode>.diagnosis.txt and <episode>.timeline.png next to the episode file. Reads only the episode's own
records (filter receipts and the ground-truth log written by run_episode); nothing is recomputed.
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def node_of_row(row_id: str | None) -> str | None:
    """Row ids look like "<family>:<protected body>:<node>"; return the node (e.g. "O(42)")."""
    return None if row_id is None else row_id.split(":")[-1]


def main(path: Path) -> None:
    episode = json.loads(path.read_text())
    receipts, truth, result = episode["receipts"], episode["ground_truth"], episode["result"]
    label = {n["node_id"]: n["category"] for n in episode["graph"]["nodes"]}
    exempt = sorted({e for seg in episode["segments"] for e in seg.get("exempt_ids", [])})

    def name(node):
        return f"{node} ({label.get(node, '?')})"

    ticks = np.arange(len(receipts))
    nominal = np.array([r["u_nominal"] for r in receipts])
    applied = np.array([r["u_applied"] for r in receipts])
    speed_nominal = np.linalg.norm(nominal, axis=1)
    speed_applied = np.linalg.norm(applied, axis=1)
    h_min = np.array([np.nan if r["h_min"] is None else r["h_min"] for r in receipts])
    eef = np.array([t["eef"] for t in truth])
    outcomes = collections.Counter(r["outcome"] for r in receipts)
    binding = collections.Counter(node_of_row(r["h_min_row"]) for r in receipts if r["outcome"] != "passthrough")

    lines = [
        f"episode: {path.name}",
        f"task: {result.get('task')}  seed: {result.get('seed')}  arm: {result.get('arm')}",
        f"success: {result['success']}  steps: {result['steps']}  stalled: {result['stalled']}  "
        f"start inside a margin: {result['start_violation']}",
        f"contact ticks (with non-exempt nodes): {result['contact_ticks']}  nodes touched: {result.get('contact_nodes')}",
        f"exempt (target) nodes: {', '.join(name(e) for e in exempt) or 'none'}",
        "",
        "filter outcome per tick: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()),
        "nodes that set the closest constraint on non-passthrough ticks:",
        *[f"  {name(n)}: {c} ticks" for n, c in binding.most_common(5)],
        "",
        "progress in quarters of the episode (end-effector net displacement / path length, applied/nominal speed):",
    ]
    for q in range(4):
        a, b = q * len(ticks) // 4, max((q + 1) * len(ticks) // 4, q * len(ticks) // 4 + 1)
        net = np.linalg.norm(eef[b - 1] - eef[a])
        path_len = np.sum(np.linalg.norm(np.diff(eef[a:b], axis=0), axis=1))
        ratio = np.median(speed_applied[a:b] / np.maximum(speed_nominal[a:b], 1e-9))
        lines.append(f"  ticks {a:4d}-{b - 1:4d}: net {net:.3f} m, path {path_len:.3f} m, speed ratio {ratio:.2f}")
    text = "\n".join(lines)
    path.with_suffix(".diagnosis.txt").write_text(text + "\n")
    print(text)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(ticks, speed_nominal, label="VLA command", color="0.6")
    axes[0].plot(ticks, speed_applied, label="after filter", color="C0")
    axes[0].set_ylabel("command norm")
    axes[0].legend(loc="upper right")
    axes[1].plot(ticks, h_min, color="C3")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("closest clearance h (m)")
    top = [n for n, _ in binding.most_common(4)]
    rows = [top.index(node_of_row(r["h_min_row"])) if node_of_row(r["h_min_row"]) in top else np.nan for r in receipts]
    axes[2].scatter(ticks, rows, s=4, color="C2")
    axes[2].set_yticks(range(len(top)), [name(n) for n in top])
    axes[2].set_ylabel("closest node")
    axes[2].set_xlabel("tick")
    fig.suptitle(f"{path.stem}: success={result['success']}")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".timeline.png"), dpi=110)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(Path(arg))
