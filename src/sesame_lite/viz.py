"""Demo videos (spec v4, section 12, items 1 and 6), drawn with MuJoCo's renderer on the live simulation.

World view (free camera; the policy's cameras are untouched), per tick:
  - the robot's collision spheres coloured by their barrier value h (smallest over the admitted objects): red at
    h <= 0, through yellow, to green at h >= GREEN_AT_M;
  - a line from the sphere with the smallest h to its closest map point, and that point's ball (radius rho_m);
  - the balls of the admitted objects near the robot (within NEAR_M of a sphere), small and translucent;
  - the planned path and its corridor (navigation) or the manipulation disc;
  - a status strip (time, QP outcome, rows, smallest h, true clearance, closest robot link and object) and a
    marker on every relaxed or held tick; the agent-view frame the policy received as an inset.
A second stream follows the robot from close by with its spheres over its geometry (robot_spheres.mp4, item 1).
Walls are drawn translucent.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sesame_lite.planner import NAVIGATION
from sesame_lite.scene_graph import _raw_data, _raw_model

GREEN_AT_M = 0.20
NEAR_M = 0.35
RUN_TITLES = {"sesame": "SESAME-3D", "nofilter": "No filter"}


def h_colour(h: float, alpha: float = 0.55) -> tuple:
    """red (h <= 0) -> yellow (GREEN_AT_M / 2) -> green (h >= GREEN_AT_M)."""
    t = float(np.clip(h / GREEN_AT_M, 0.0, 1.0))
    r, g = (1.0, 2 * t) if t < 0.5 else (2 * (1 - t), 1.0)
    return (r, min(g, 1.0) * 0.85, 0.1, alpha)


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", size)
    except OSError:
        return ImageFont.load_default()


class WorldRenderer:
    """Free-camera renderer over the live kitchen with overlay geoms."""

    def __init__(self, env, *, width=960, height=720, view="bird"):
        self.env = env
        self.model, self.data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(self.model, height, width, max_geom=40000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.view = view
        self.frame_mission(width / height)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(self.model.ngeom)]
        bodies = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or "" for g in range(self.model.ngeom)]
        self.wall_geoms = {g for g in range(self.model.ngeom) if "wall" in names[g] or "wall" in bodies[g]}
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1  # visual geoms only

    def frame_mission(self, aspect: float) -> None:
        """Navigation: a bird view over the planned path and its corridor. Manipulation: a 3/4 view of the robot."""
        env, seg = self.env, self.env.segments[0]
        base = env.robot_pose()[0]
        fovy = np.deg2rad(float(self.model.vis.global_.fovy))
        if self.view == "bird":
            pts = np.vstack([base[:2], seg.path_polyline_xy]) if seg.path_polyline_xy is not None else base[None, :2]
            pad = (env.bundle.corridor_radius_m or 1.0) + 0.5
            lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
            self.cam.lookat[:] = [*((lo + hi) / 2), 0.4]
            self.cam.distance = max(hi[1] - lo[1], (hi[0] - lo[0]) / aspect) / (2 * np.tan(fovy / 2))
            self.cam.azimuth, self.cam.elevation = -90.0, -89.0  # world +x to the right, +y up
        else:
            yaw = env.base_yaw()  # from behind the robot (the floor it stands on), looking where it faces
            self.cam.lookat[:] = base + np.array([0.45 * np.cos(yaw), 0.45 * np.sin(yaw), 0.6])
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.2, np.degrees(yaw) + 25.0, -32.0

    def _add(self, gtype, size, pos, mat, rgba):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            raise RuntimeError(f"render scene full ({scn.maxgeom} geoms)")
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64), np.asarray(mat, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1

    def _line(self, a, b, rgba, width=4.0):
        scn = self.renderer.scene
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def render(self, *, robot_only: bool = False) -> np.ndarray:
        env = self.env
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        scn = self.renderer.scene
        for i in range(scn.ngeom):
            gg = scn.geoms[i]
            if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                if robot_only and int(self.model.geom_bodyid[gg.objid]) not in env.robot_bodies:
                    gg.rgba[3] = 0.0
                elif gg.objid in self.wall_geoms:
                    gg.rgba[3] = 0.12
        bar = env.last_barrier
        h_sphere = bar.h.min(axis=1) if bar.h.shape[1] else np.full(len(bar.radii), np.inf)
        for c, r, h in zip(bar.centres, bar.radii, h_sphere):
            self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], c, np.eye(3), h_colour(h, 0.35 if robot_only else 0.55))
        if robot_only:
            return self.renderer.render().copy()
        flt = env.filter
        if len(flt.points_np):
            near = np.min(np.linalg.norm(flt.points_np[:, None, :] - bar.centres[None], axis=2) - bar.radii[None], axis=1) < NEAR_M
            rho = flt.rho.cpu().numpy()
            for p in flt.points_np[near]:
                self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], p, np.eye(3), (0.75, 0.2, 0.2, 0.5))
            i = int(np.argmin(h_sphere))
            k = int(bar.closest[i])
            self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [rho[k], 0, 0], flt.points_np[k], np.eye(3), (0.85, 0.1, 0.1, 0.25))
            self._line(bar.centres[i], flt.points_np[k], (0.05, 0.05, 0.05, 1.0))
        seg = env.segment
        if seg.kind == NAVIGATION and env.bundle.corridor_radius_m:
            path = seg.path_polyline_xy
            for a, b in zip(path[:-1], path[1:]):
                for t in np.linspace(0, 1, max(1, int(np.linalg.norm(b - a) / 0.25)), endpoint=False):
                    p = a + t * (b - a)
                    self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [env.bundle.corridor_radius_m, 0.004, 0], [p[0], p[1], 0.01], np.eye(3), (0.20, 0.45, 0.90, 0.035))
                for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05)), endpoint=False):
                    p = a + t * (b - a)
                    self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], [p[0], p[1], 0.02], np.eye(3), (0.10, 0.30, 0.90, 0.9))
            self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12, 0.02, 0], [path[-1][0], path[-1][1], 0.03], np.eye(3), (0.10, 0.80, 0.90, 0.9))
        elif seg.center_xy is not None and env.bundle.region_radius_m:
            self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [env.bundle.region_radius_m, 0.004, 0], [seg.center_xy[0], seg.center_xy[1], 0.01], np.eye(3), (0.20, 0.45, 0.90, 0.035))
        return self.renderer.render().copy()


class EpisodeRecorder:
    """Per-tick frames of the world view (item 6) and of the robot close-up (item 1); outcome banner at the end."""

    def __init__(self, env, *, width=960, height=720):
        self.env, self.frames, self.robot_frames = env, [], []
        self.world = WorldRenderer(env, width=width, height=height, view="bird" if env.segments[0].kind == NAVIGATION else "three_quarter")
        self.close = WorldRenderer(env, width=480, height=480, view="three_quarter")
        self.font, self.small = _font(30), _font(21)

    def __call__(self, env):
        yaw = env.base_yaw()
        base = env.robot_pose()[0]
        self.close.cam.lookat[:] = np.r_[base[:2], 0.6]
        self.close.cam.distance, self.close.cam.azimuth, self.close.cam.elevation = 3.0, np.degrees(yaw) + 25.0, -18.0  # from behind, kitchen hidden
        self.robot_frames.append(self.close.render(robot_only=True))
        frame = Image.fromarray(self.world.render())
        cam = getattr(env.env, "last_raw_obs", {}).get("video.robot0_agentview_left")
        if cam is not None:
            inset = Image.fromarray(np.ascontiguousarray(cam)).resize((frame.width // 4, frame.width // 4))
            frame.paste(inset, (frame.width - inset.width - 8, 52))
        rec, gt = env.receipts[-1], env.ground_truth[-1]
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rectangle([0, 0, frame.width, 46], fill=(0, 0, 0, 170))
        draw.text((12, 6), f"{RUN_TITLES[env.tag]}  |  Hydra map  |  t = {rec['tick'] * 0.05:4.1f} s", font=self.font, fill=(255, 255, 255))
        if rec["outcome"] in ("relaxed", "hold"):  # the relaxed-tick marker
            label = f"QP {rec['outcome'].upper()}" + (f": {len(rec['slack_rows'])} rows, slack {100 * rec['max_slack']:.1f} cm/s" if rec["outcome"] == "relaxed" else f" ({rec['qp_status']})")
            w = draw.textlength(label, font=self.small)
            draw.rectangle([8, 54, 24 + w, 86], fill=(200, 40, 40, 230))
            draw.text((16, 58), label, font=self.small, fill=(255, 255, 255))
        h, ht = rec["h_min"], gt["clearance"]
        line1 = f"QP {rec['outcome']}   rows {rec['n_pairs']} ({rec['n_rows_qp']} can bind)   smallest h {'-' if h is None else f'{100 * h:+5.1f} cm'}   true clearance {'-' if ht is None else f'{100 * ht:+5.1f} cm'}"
        closest = "-"
        if rec["h_min_pair"] is not None:
            _, link, obj = rec["h_min_pair"]
            closest = f"{link.replace('robot0_', '').replace('mobilebase0_', 'base ').replace('gripper0_right_', 'gripper ')}  ->  {env.graph.nodes[obj].category} ({obj})"
        line2 = f"closest: {closest}" + ("   CONTACT: " + ", ".join(gt["touched"]) if gt["touched"] else "")
        draw.rectangle([0, frame.height - 62, frame.width, frame.height], fill=(0, 0, 0, 170))
        bad = bool(gt["touched"]) or (ht is not None and ht < 0)
        draw.text((12, frame.height - 58), line1, font=self.small, fill=(255, 95, 95) if bad else (255, 255, 255))
        draw.text((12, frame.height - 30), line2, font=self.small, fill=(255, 95, 95) if gt["touched"] else (255, 255, 255))
        self.frames.append(np.asarray(frame))

    def finish(self, result: dict, hold_s: float = 2.0, fps: int = 20) -> None:
        """Hold the last frame with a banner stating the outcome."""
        if not self.frames:
            return
        oracle_category = {f"O:{oid}": n.category for oid, n in self.env.oracle_graph.nodes.items()}  # contacts are oracle ids
        nodes = sorted({oracle_category.get(n, n) for n in result["contact_nodes"]})
        if result["outcome"] == "success":
            text, color = "GOAL REACHED  -  no contact", (40, 160, 90)
        elif result["outcome"] == "success_but_unsafe":
            text, color = f"GOAL REACHED  -  UNSAFE: {result['contact_ticks']} contact ticks ({', '.join(nodes[:2]) or 'clearance below zero'})", (200, 60, 50)
        else:
            text, color = result["outcome"].replace("_", " ").upper(), (210, 130, 30)
        last = Image.fromarray(self.frames[-1])
        draw = ImageDraw.Draw(last, "RGBA")
        w = draw.textlength(text, font=self.font)
        y = last.height // 2 - 30
        draw.rectangle([0, y, last.width, y + 60], fill=(*color, 220))
        draw.text(((last.width - w) / 2, y + 12), text, font=self.font, fill=(255, 255, 255))
        self.frames += [np.asarray(last)] * int(hold_s * fps)

    def save(self, path: Path, robot_path: Path | None = None, fps: int = 20):
        from lerobot.utils.io_utils import write_video

        if self.frames:
            write_video(str(path), np.stack(self.frames), fps=fps)
        if robot_path is not None and self.robot_frames:
            write_video(str(robot_path), np.stack(self.robot_frames), fps=fps)
