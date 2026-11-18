"""A video of a finished run, rendered from its logged simulator states (no policy, no filter is run again).

    MUJOCO_GL=egl python -s scripts/eval_replay_video.py RUN.json [--scenes-root DIR] [--out FILE.mp4] [--every 2]

The run file's .npz holds the whole joint state of every tick (qpos) and the robot's spheres' smallest barrier values
(sphere_h). Every ``every``-th tick is set into the scene and rendered from an outside camera that follows the robot,
with the robot's collision spheres coloured by their barrier value (green far, red at the boundary, as in the evaluation
videos), the designated obstacle tinted, and a caption: method, time, task done, contact. 1920x1080, 20 fps / every.
"""

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import sesame_lite.tasks  # noqa: F401,E402
from sesame_lite.env import SesameRoboCasaEnv  # noqa: E402
from sesame_lite.eval_scene import SCENES_DIR, EvalFilterEnv  # noqa: E402
from sesame_lite.scene_graph import _raw_data, _raw_model, _subtree_body_ids  # noqa: E402
from sesame_lite.viz import WorldRenderer, h_colour  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--scenes-root", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--every", type=int, default=2)
args = ap.parse_args()
run = Path(args.run)
d = json.loads(run.read_text())
r = d["result"]
z = np.load(run.with_suffix(".npz"))
root = Path(args.scenes_root) if args.scenes_root else SCENES_DIR
sc = json.loads((root / r["scene"] / "scene.json").read_text())
base = SesameRoboCasaEnv(sc["task"], env_kwargs=dict(sc["env_kwargs"]), episode_length=sc["horizon"])
env = EvalFilterEnv(base, filter=False)
env.reset(scene_dir=root / r["scene"])
m, data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
world = WorldRenderer(env, width=1920, height=1080, view="three_quarter")
ob = sc.get("safety_obstacle") or {}
tint = _subtree_body_ids(m, env.rs.obj_body_id[ob["name"]]) if ob.get("placed") else set()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
title = {"nominal": "Nominal (no filter)", "ee": "End-effector filter", "fixed": "Fixed margin", "noissf": "Proposed without ISSf", "proposed": "Proposed"}[r["method"]]
out = Path(args.out) if args.out else run.with_name(run.stem + "_replay.mp4")
yaw_s = None
contacts = [bool([c for c in g["contacts"] if c not in g["exempt_pair_contacts"]]) for g in d["ground_truth"]]
with av.open(str(out), "w") as video:
    stream = video.add_stream("h264", rate=max(1, 20 // args.every), options={"crf": "21", "preset": "medium"})
    stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
    for t in range(0, len(z["qpos"]), args.every):
        data.qpos[:] = z["qpos"][t]
        mujoco.mj_forward(m, data)
        bp, yaw = env.robot_pose()[0], env.base_yaw()
        yaw_s = yaw if yaw_s is None else yaw_s + 0.1 * float(np.arctan2(np.sin(yaw - yaw_s), np.cos(yaw - yaw_s)))
        world.cam.lookat[:] = [bp[0] + 0.35 * np.cos(yaw_s), bp[1] + 0.35 * np.sin(yaw_s), 0.8]
        world.cam.distance, world.cam.azimuth, world.cam.elevation = 3.0, np.degrees(yaw_s) + 30.0, -28.0
        centres, radii = env.robot.spheres(z["q"][t])
        sph = [h_colour(float(h), 0.5) for h in z["sphere_h"][t]]
        world.renderer.update_scene(data, camera=world.cam, scene_option=world.opt)
        scn = world.renderer.scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and int(m.geom_bodyid[g.objid]) in tint:
                g.rgba[:3] = 0.35 * np.asarray(g.rgba[:3]) + 0.65 * np.array([1.0, 0.55, 0.0])
        for c, rad, col in zip(centres, radii, sph):
            world._add(mujoco.mjtGeom.mjGEOM_SPHERE, [rad, 0, 0], c, np.eye(3), col)
        img = Image.fromarray(world.renderer.render())
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, 0, 1920, 64], fill=(0, 0, 0, 150))
        done = r["success"] and t >= r["steps"] - 1
        txt = f"{title}   |   t = {t * 0.05:5.1f} s   |   {'TASK DONE' if done else sc['language']}" + ("   |   CONTACT" if t < len(contacts) and contacts[t] else "")
        draw.text((24, 14), txt, font=font, fill=(255, 90, 90, 255) if t < len(contacts) and contacts[t] else (255, 255, 255, 255))
        if ob.get("placed"):
            draw.text((24, 1080 - 52), "orange: the safety obstacle", font=font, fill=(255, 190, 90, 255))
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")):
            video.mux(packet)
    for packet in stream.encode():
        video.mux(packet)
print("wrote", out)
base.close()
