"""TODO (spec v4, section 11): the Qwen planner on the Hydra map is out of scope for this version; not run since v4.

Plan one mission with the LangGraph planner (Qwen3-14B, tool calling) and write it as a runnable plan file.

The scene is the mission's kitchen at the given seed, exactly as run_episode builds it (for nav_clutter: with the
flour bag already on the floor). The planner sees only the Hydra scene graph and the task instruction.

    bash scripts/llm/serve_qwen.sh &      # once
    MUJOCO_GL=egl .venv/bin/python -s scripts/plan_mission.py --plan plans/open_drawer.json --seed 100
    MUJOCO_GL=egl .venv/bin/python -s scripts/run_episode_pi05.py --plan outputs/plans_llm/open_drawer_s100.json --seed 100

Writes outputs/plans_llm/<mission>_s<seed>.json (the plan, in plans/*.json format) and
<mission>_s<seed>.transcript.json (every model message and tool result). The hand-written plan's roles, resolved
on the same scene, are stored under "reference" for scoring; the planner never sees them.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # same scene as run_episode (see sesame_lite/determinism.py)
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402  (registers NavigateKitchenClutter)
from sesame_lite.agents.planner_agent import run_planner  # noqa: E402
from sesame_lite.env import SafetyFilterEnv, SesameRoboCasaEnv  # noqa: E402
from sesame_lite.planner import load_plan, resolve_role  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True, help="the mission file in plans/ (task, kitchen and the hand-written reference plan)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/plans_llm")
args = ap.parse_args()

mission = load_plan(args.plan)
base = SesameRoboCasaEnv(mission["task"], env_kwargs=dict(mission.get("env_kwargs", {})), episode_length=mission.get("horizon"))
env = SafetyFilterEnv(base, filter=False)  # no classifier call; reset builds the scene and its scene graph
env.reset(seed=args.seed, plan=mission)
task_instruction = env.env.task_description

t0 = time.time()
result = run_planner(env.graph, task_instruction)
wall_s = time.time() - t0

# the hand-written plan resolved on the same scene: kept for scoring only
reference = {"roles": {name: resolve_role(spec, env.rs, env.lookup) for name, spec in mission["roles"].items()},  # filter-side ids
             "segment_kinds": [s["kind"] for s in mission["segments"]],
             "exempt_ids": [list(s.exempt_ids) for s in env.segments]}

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"{mission['mission']}_s{args.seed}"
plan_file = {key: mission[key] for key in ("task", "env_kwargs", "horizon", "injections") if key in mission}
plan_file.update({"mission": mission["mission"], "seed": args.seed, "task_instruction": task_instruction, "graph_source": "hydra",
                  **(result["plan"] or {"roles": {}, "segments": []}),
                  "planner": {"accepted": result["plan"] is not None, "failure": result["failure"], "turns": result["turns"],
                              "wall_s": round(wall_s, 1), "committed": result["committed"], "settings": result["settings"]},
                  "snapshot_id": env.graph.snapshot_id,
                  "reference": reference})
(out_dir / f"{stem}.json").write_text(json.dumps(plan_file, indent=1))
(out_dir / f"{stem}.transcript.json").write_text(json.dumps({"system_prompt": result["system_prompt"], "messages": result["transcript"]}, indent=1))
env.graph.save(out_dir / f"{stem}.dsg.json")
print("PLAN " + json.dumps({"mission": mission["mission"], "seed": args.seed, "accepted": result["plan"] is not None, "turns": result["turns"],
                            "wall_s": round(wall_s, 1), "roles": (result["plan"] or {}).get("roles"), "reference_roles": reference["roles"]}), flush=True)
base.close()
