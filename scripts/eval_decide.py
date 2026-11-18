"""The language model's decisions of one scene and map kind, asked once before the campaign (EVAL_SPEC.md: all methods
on the same scene; tuning_amendments.md A7): the object labels of both segments (obstacle, passable or irrelevant).

    MUJOCO_GL=egl python -s scripts/eval_decide.py --scene close_fridge_s4 --map full

Needs the Qwen server (scripts/llm/serve_qwen.sh). Writes outputs/eval/scenes/<scene>/model_<map>.json and marks it
complete; the runs then only read it (the server is not needed during the campaign).
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import SCENES_DIR, EvalFilterEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--map", default="full", choices=("full", "look"))
ap.add_argument("--scenes-root", default=None, help="a directory of scene variants (tuning)")
args = ap.parse_args()
scene_dir = (Path(args.scenes_root) if args.scenes_root else SCENES_DIR) / args.scene
scene = json.loads((scene_dir / "scene.json").read_text())
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, map_kind=args.map, pair_pruning="rule", classifier="qwen")
env.reset(scene_dir=scene_dir)  # the driving segment's labels
env.begin_segment(1)  # the manipulation segment's labels
env.shared.store("complete", True)
labels = {k: v for k, v in env.shared.load().items() if k.startswith("labels:")}
counts = {}
for v in labels.values():
    for lab in v["labels"].values():
        counts[lab] = counts.get(lab, 0) + 1
print("DECIDED " + json.dumps({"scene": args.scene, "map": args.map, "allowed_for_the_gripper": sorted({f"{p['role']}:{p['category']}" for p in env.dropped_pairs}),
                               "labels": counts, "label_requests": len(labels), "label_fallbacks": sum(bool(v["fallback_used"]) for v in labels.values()),
                               "label_s": [v["latency_s"] for v in labels.values()], "unmapped_task_objects": env.unmapped_roles}), flush=True)
base.close()
