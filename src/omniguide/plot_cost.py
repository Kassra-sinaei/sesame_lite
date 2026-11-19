"""The guidance cost over one run, from its query record (<run>.omniguide.json, written by scripts/eval_run.py).

    .venv/bin/python src/omniguide/plot_cost.py <run>.omniguide.json [<out>.png]

Per query: the collision value (collision_weight * sum of log d over the robot points within 0.15 m; 0: no point
within 0.15 m) of the standard sampler's chunk and of the chunk sent to the robot, and the smallest distance of
their robot points to the field; the queries where the noise pick replaced the first noise are marked.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
d = json.loads(src.read_text())
q = [x for x in d["queries"] if "value_standard" in x]
t = [x["tick"] for x in q]
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, [x["value_standard"] for x in q], ".-", label="standard chunk")
ax[0].plot(t, [x.get("value_out", x["value_standard"]) for x in q], ".-", label="chunk sent")
picked = [x for x in q if x.get("picked", -1) >= 0]
ax[0].plot([x["tick"] for x in picked], [x["value_out"] for x in picked], "kx", label="noise pick replaced the first noise")
ax[0].set_ylabel("collision value")
ax[0].legend()
ax[1].plot(t, [x["d_min_standard"] for x in q], ".-", label="standard chunk")
ax[1].plot(t, [x.get("d_min_out", x["d_min_standard"]) for x in q], ".-", label="chunk sent")
ax[1].axhline(0.15, color="gray", lw=0.8, ls="--")
ax[1].set_ylabel("smallest distance of the robot points [m]")
ax[1].set_xlabel("tick (0.05 s)")
fig.suptitle(f"{src.name}: guidance_factor {d['guidance_factor']}, {len(q)} queries")
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
