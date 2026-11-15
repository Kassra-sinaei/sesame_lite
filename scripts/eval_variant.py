"""Stage 0b round 4 (Kasra, 2026-11-13): a tuning scene with the path planned on the full-pass map that contains the
clutter, at a given voxel size. Written to outputs/eval_debug/scenes_v<voxel>/<scene>/ (tuning only, not counted).

    MUJOCO_GL=egl python -s scripts/eval_variant.py --scene counter_to_sink_s0 --voxel 0.025

The clutter stays where the scene's seeded rule put it (relative to the path planned without clutter). The full-pass
map is built at ``--voxel`` (the scene's own map is reused at 5 cm); the shared path is then planned on it from the
start to the standoff with the two-layer planner (base band with inflation = base half-width + 0.10 m; the rigid column
and shoulder in their own height band with their own inflation; around every ball), and replaces the plan's path. Prints one VARIANT line with the mapping time, peak memory and
the planner's margin; ``no_path`` if the planner finds none.
"""

import argparse
import json
import os
import resource
import shutil
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import DEBUG_DIR, SCENES_DIR, EvalFilterEnv, apply_scene  # noqa: E402
from sesame_lite.mapping import MappingParams, build_map  # noqa: E402
from sesame_lite.planner import NoPathError, plan_base_path  # noqa: E402
from sesame_lite.scene_graph import capture_scene_graph  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--voxel", type=float, required=True)
ap.add_argument("--suffix", default="", help="written to scenes_v<voxel><suffix>/; the maps are linked from scenes_v<voxel>/ if they exist there")
args = ap.parse_args()
src = SCENES_DIR / args.scene
built = DEBUG_DIR / f"scenes_v{args.voxel:g}" / args.scene
out = DEBUG_DIR / f"scenes_v{args.voxel:g}{args.suffix}" / args.scene
out.mkdir(parents=True, exist_ok=True)
for kind in ("map_full", "map_look"):  # maps already built for this voxel size are reused
    if args.suffix and (built / kind).exists() and not (out / kind).exists():
        (out / kind).symlink_to((built / kind).resolve(), target_is_directory=True)
scene = json.loads((src / "scene.json").read_text())
base = SesameRoboCasaEnv(scene["task"], env_kwargs=dict(scene["env_kwargs"]), episode_length=scene["horizon"])
env = EvalFilterEnv(base, filter=False)
env.env.reset(seed=scene["seed"])
apply_scene(env, scene)
oracle, lookup = capture_scene_graph(env.rs)
env.oracle_graph, env.oracle_lookup = oracle, lookup
assert oracle.snapshot_id == scene["snapshot_id"]
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t0 = time.time()
if (out / "map_full").exists():
    pass
elif abs(args.voxel - MappingParams().voxel_size_m) < 1e-9:
    for kind in ("map_full", "map_look"):
        if not (out / kind).exists():
            shutil.copytree(src / kind, out / kind)
else:
    build_map(env.rs, oracle, lookup, out / "map_full", seed=scene["seed"], params=MappingParams(voxel_size_m=args.voxel))
map_s, rss1 = time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
# the shared path on the map with the clutter (planner unchanged)
env._load_map(out / "map_full")
env._base_geometry()
start, goal = np.asarray(scene["start"]["xy"]), np.asarray(scene["standoff"]["xy"])
info = {"scene": args.scene, "voxel_m": args.voxel, "mapping_s": round(map_s, 1), "peak_rss_mb_before": round(rss0), "peak_rss_mb_after": round(rss1),
        "planner_inflation_m": round(env.inflation_m, 3), "base_band_m": [round(v, 2) for v in env.z_band],
        "upper_band_m": [round(v, 2) for v in env.upper[0]], "upper_inflation_m": round(env.upper[1], 3)}
try:
    path = plan_base_path(env.graph, env.plan_balls, start, goal, inflation_m=env.inflation_m, z_band=env.z_band, upper=env.upper)
    scene["plan"]["path_xy_without_clutter"] = scene["plan"]["path_xy"]
    scene["plan"]["path_xy"] = np.round(path, 4).tolist()
    scene["plan"]["planned_on"] = f"map_full with the clutter, voxel {args.voxel:g} m, two-layer planner (base; column and shoulder)"
    info.update(path_length_m=round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 3), status="ok")
except NoPathError as exc:
    info.update(status="no_path", reason=str(exc))
scene["map_voxel_m"] = args.voxel
(out / "scene.json").write_text(json.dumps(scene, indent=1))
print("VARIANT " + json.dumps(info), flush=True)
base.close()
