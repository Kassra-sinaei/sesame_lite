"""VLA validation of lerobot/pi05_robocasa on the protocol REKI-2B was validated with on the laptop (HANDOFF §8.1):
pretrain-split kitchens, robocasa 1.0.1 registry horizons, episode seeds drawn from default_rng(42) (as in
eval_nav.py), the task's own success check, no safety filter, no navigation. One task per process.

    PYTHONPATH=deps/robocasa-1.0.1 MUJOCO_GL=egl python -s scripts/eval_vla_pi05.py --task OpenCabinet --episodes 10 --video 2

Laptop REKI reference: OpenCabinet 7/10, PickPlaceCounterToCabinet 8/10.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import imageio.v2 as imageio
import numpy as np
from lerobot.envs.robocasa import RoboCasaEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_nominal import POLICY_PATH, Pi05Policy  # noqa: E402
from sesame_lite.determinism import begin_episode, seed_fallback_generators  # noqa: E402
import sesame_lite.render_fix  # noqa: F401,E402  (stops freed render contexts from corrupting camera frames)

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split", default="pretrain")
ap.add_argument("--video", type=int, default=2, help="record the first N episodes")
ap.add_argument("--out", default="outputs/vla_eval")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

seed_fallback_generators()
env = RoboCasaEnv(args.task, split=args.split, obj_registries=("objaverse", "lightwheel"))
horizon = env._max_episode_steps
policy = Pi05Policy(args.task)
rng = np.random.default_rng(args.seed)
results = []


def frame_noise(img) -> float:
    """Mean |difference| of horizontally adjacent pixels: ~3-15 for a rendered kitchen, ~75+ for a corrupt frame."""
    return float(np.abs(np.diff(img.astype(np.int16), axis=1)).mean())


for ep in range(args.episodes):
    seed = int(rng.integers(0, 2**31 - 1))
    begin_episode(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    meta = env._env.env.get_ep_meta()
    lang = env.task_description
    frames, success, steps, t0, noise = [], False, 0, time.time(), 0.0
    for steps in range(1, horizon + 1):
        if steps % 10 == 1:
            noise = max(noise, *(frame_noise(v) for v in obs["pixels"].values()))
        if ep < args.video and steps % 2 == 1:
            frames.append(np.concatenate([obs["pixels"][c] for c in ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right")], axis=1))
        obs, _, terminated, _, info = env.step(policy(obs, lang))
        if info.get("is_success"):
            success = True
            break
        if terminated:
            break
    r = {"episode": ep, "seed": seed, "instruction": lang, "success": success, "steps": steps, "horizon": horizon,
         "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"), "wall_s": round(time.time() - t0, 1), "max_frame_noise": round(noise, 1)}
    results.append(r)
    if frames:
        imageio.mimsave(out / f"{args.task}_ep{ep:02d}_{'success' if success else 'fail'}.mp4", frames, fps=10)
    print(f"{args.task} ep {ep:2d} seed {seed}: success={success} steps={steps}/{horizon} layout={r['layout_id']} style={r['style_id']} noise={r['max_frame_noise']} '{lang}' {r['wall_s']}s", flush=True)

summary = {"task": args.task, "policy": POLICY_PATH, "split": args.split, "horizon": horizon, "seed": args.seed,
           "episodes": len(results), "success": sum(r["success"] for r in results)}
print("SUMMARY", json.dumps(summary))
(out / f"{args.task}_results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
env.close()
