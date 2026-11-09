"""Demo rendering: world cameras with 3DSG overlays, per-episode videos, side-by-side cuts.

Everything is drawn with MuJoCo's own renderer on the live simulation (``mujoco.Renderer`` with a
free camera) so the policy's observation cameras are untouched: boxes of the 3DSG are added as
scene geoms coloured by their role in the filter (constraint row, passable, irrelevant, exempt,
outside the corridor), the navigation corridor / manipulation disc as translucent discs, the
end-effector sphere, and the base goal. Front walls are made translucent per frame.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sesame_lite.planner import NAVIGATION
from sesame_lite.scene_graph import _raw_data, _raw_model

RGBA = {
    "row": (0.85, 0.15, 0.15, 0.40),  # constraint row in this arm
    "passable": (0.15, 0.75, 0.30, 0.55),
    "irrelevant": (0.55, 0.55, 0.55, 0.35),
    "exempt": (0.15, 0.40, 0.95, 0.50),
    "outside": (1.0, 1.0, 1.0, 0.06),  # not a candidate (outside corridor / disc)
    "passive": (0.80, 0.80, 0.80, 0.22),  # unfiltered arm: geometry shown, nothing enforced
    "corridor": (0.20, 0.45, 0.90, 0.035),
    "path": (0.10, 0.30, 0.90, 0.9),
    "eef": (0.95, 0.85, 0.10, 0.9),
    "goal": (0.10, 0.80, 0.90, 0.9),
}
ARM_TITLES = {"A": "Baseline: no safety filter", "B": "Naive CBF: every object", "B+": "All objects, target exempt", "C": "SESAME-3D (ours)"}


def _quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fragment_roles(env) -> dict[str, str]:
    """Role of every scene-graph box in the current segment, for colouring."""
    seg, bundle = env.segment, env.bundle
    roles = {}
    rows = set(env.row_ids)
    for nid in env.graph.nodes:
        label = bundle.labels.get(nid)
        if nid in rows:
            roles[nid] = "passive" if env.arm == "A" else "row"
        elif nid in seg.exempt_ids:
            roles[nid] = "exempt"
        elif label == "passable":
            roles[nid] = "passable"
        elif label == "irrelevant":
            roles[nid] = "irrelevant"
        elif nid in bundle.candidate_ids:
            roles[nid] = "row" if env.arm != "A" else "passive"
        else:
            roles[nid] = "outside"
    return roles


class WorldRenderer:
    """Free-camera renderer over the live kitchen with overlay geoms."""

    def __init__(self, env, *, width=960, height=720, view="bird"):
        self.env = env
        self.model, self.data = _raw_model(env.rs.sim), _raw_data(env.rs.sim)
        # the kitchen model declares a small offscreen buffer (robosuite's camera size); enlarge it for the world view
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(self.model, height, width, max_geom=40000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.frame_mission(width / height, view)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(self.model.ngeom)]
        bodies = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or "" for g in range(self.model.ngeom)]
        self.wall_geoms = {g for g in range(self.model.ngeom) if "wall" in names[g] or "wall" in bodies[g]}
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1  # visual geoms only

    def frame_mission(self, aspect: float, view: str) -> None:
        """Frame the area the mission uses (planned path with its corridor, or the manipulation disc) plus a margin."""
        env, seg = self.env, self.env.segments[0]
        pts = [env.robot_state().base_position[:2]]
        if seg.path_polyline_xy is not None:
            pts += list(seg.path_polyline_xy)
            pad = (env.bundle.corridor_radius_m or 1.0) + 0.5
        else:
            pad = (env.bundle.region_radius_m or 1.0) + 0.6
        pts = np.asarray(pts)
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        span_x, span_y = hi - lo
        fovy = np.deg2rad(float(self.model.vis.global_.fovy))
        self.cam.lookat[:] = [*((lo + hi) / 2), 0.4]
        dist = max(span_y, span_x / aspect) / (2 * np.tan(fovy / 2))
        if view == "bird":  # camera on the -y side looking down: world +x to the right, +y up, as in the figures
            self.cam.distance, self.cam.azimuth, self.cam.elevation = dist, -90.0, -89.0
        else:
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.2 * dist, -60.0, -40.0

    def _add(self, gtype, size, pos, mat, rgba):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64), np.asarray(mat, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1

    def render(self, *, overlays=True) -> np.ndarray:
        env = self.env
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        scn = self.renderer.scene
        for i in range(scn.ngeom):
            if scn.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM and scn.geoms[i].objid in self.wall_geoms:
                scn.geoms[i].rgba[3] = 0.12
        if overlays:
            roles = fragment_roles(env)
            for node in env.graph.obstacle_nodes():
                frag = node.fragment()
                self._add(mujoco.mjtGeom.mjGEOM_BOX, frag.half_extents, frag.center, frag.rotation, RGBA[roles[node.node_id]])
            seg = env.segment
            if seg.kind == NAVIGATION and env.bundle.corridor_radius_m:
                path = seg.path_polyline_xy
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    n = max(1, int(np.linalg.norm(b - a) / 0.25))
                    for t in np.linspace(0, 1, n, endpoint=False):
                        p = a + t * (b - a)
                        self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [env.bundle.corridor_radius_m, 0.004, 0], [p[0], p[1], 0.01], np.eye(3), RGBA["corridor"])
                    for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05)), endpoint=False):
                        p = a + t * (b - a)
                        self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], [p[0], p[1], 0.02], np.eye(3), RGBA["path"])
                self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12, 0.02, 0], [path[-1][0], path[-1][1], 0.03], np.eye(3), RGBA["goal"])
            elif seg.center_xy is not None and env.bundle.region_radius_m:
                self._add(mujoco.mjtGeom.mjGEOM_CYLINDER, [env.bundle.region_radius_m, 0.004, 0], [seg.center_xy[0], seg.center_xy[1], 0.01], np.eye(3), RGBA["corridor"])
            st = env.robot_state()
            self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [env.policy.eef_radius_m, 0, 0], st.eef_position, np.eye(3), RGBA["eef"])
            for pt in st.arm_points:
                self._add(mujoco.mjtGeom.mjGEOM_SPHERE, [env.policy.arm_radius_m, 0, 0], pt, np.eye(3), (0.95, 0.85, 0.10, 0.35))
        return self.renderer.render().copy()


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", size)
    except OSError:
        return ImageFont.load_default()


class EpisodeRecorder:
    """Per-step frame grabber: framed world view with 3DSG overlays + agent-view inset + status strip; outcome banner at the end."""

    def __init__(self, env, *, every: int = 1, width=960, height=720):
        self.env, self.every, self.frames = env, every, []
        self.world = WorldRenderer(env, width=width, height=height, view="bird")
        self.font, self.small = _font(30), _font(23)

    def __call__(self, env):
        if env.tick % self.every:
            return
        frame = Image.fromarray(self.world.render())
        # the exact frame the policy received at the last step (no second render)
        cam = getattr(env.env, "last_raw_obs", {}).get("video.robot0_agentview_left")
        if cam is not None:
            inset = Image.fromarray(np.ascontiguousarray(cam)).resize((frame.width // 4, frame.width // 4))
            frame.paste(inset, (frame.width - inset.width - 8, 52))
        rec, gt = env.receipts[-1], env.ground_truth[-1]
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rectangle([0, 0, frame.width, 46], fill=(0, 0, 0, 170))
        draw.text((12, 6), ARM_TITLES[env.arm], font=self.font, fill=(255, 255, 255))
        h = gt["h_min_all"]
        rows = "unfiltered" if env.arm == "A" else f"constraints {len(env.row_ids)} parts (active {rec['n_rows']})"
        status = f"t = {env.tick * 0.05:4.1f} s   {rows}   clearance {100 * h:+5.1f} cm" if h is not None else f"t = {env.tick * 0.05:4.1f} s   {rows}"
        hit = bool(gt["touched"]) or (h is not None and h < -env.policy.policy_margin_m)
        if gt["touched"]:
            status += "   CONTACT"
        draw.rectangle([0, frame.height - 36, frame.width, frame.height], fill=(0, 0, 0, 170))
        draw.text((12, frame.height - 32), status, font=self.small, fill=(255, 95, 95) if hit else (255, 255, 255))
        self.frames.append(np.asarray(frame))

    def finish(self, result: dict, hold_s: float = 2.0, fps: int = 20) -> None:
        """Hold the last frame with a banner stating the outcome."""
        if not self.frames:
            return
        category = {nid: n.category for nid, n in self.env.graph.nodes.items()}
        n_contact, nodes = result["contact_ticks"], [category.get(n, n) for n in result["contact_nodes"]]
        if result["outcome"] == "success":
            text, color = "GOAL REACHED  -  no contact", (40, 160, 90)
        elif result["outcome"] == "success_but_unsafe":
            text, color = f"GOAL REACHED  -  UNSAFE: {n_contact} contact ticks ({', '.join(sorted(set(nodes))[:2]) or 'margin violated'})", (200, 60, 50)
        elif result["outcome"] == "stalled":
            blocker = category.get((self.env.receipts[-1]["h_min_row"] or "").split(":", 2)[-1], "?")  # row id = family:protected:node
            text, color = f"STALLED  -  blocked by '{blocker}'", (210, 130, 30)
        else:
            text, color = result["outcome"].replace("_", " ").upper(), (120, 120, 120)
        last = Image.fromarray(self.frames[-1])
        draw = ImageDraw.Draw(last, "RGBA")
        w = draw.textlength(text, font=self.font)
        y = last.height // 2 - 30
        draw.rectangle([0, y, last.width, y + 60], fill=(*color, 220))
        draw.text(((last.width - w) / 2, y + 12), text, font=self.font, fill=(255, 255, 255))
        self.frames += [np.asarray(last)] * int(hold_s * fps / self.every)

    def save(self, path: Path, fps: int = 20):
        from lerobot.utils.io_utils import write_video

        if self.frames:
            write_video(str(path), np.stack(self.frames), fps=max(1, fps // self.every))


def side_by_side(paths: list[Path], out: Path, titles: list[str] | None = None, fps: int = 20) -> None:
    """Horizontal cut of matched-seed videos (shorter ones hold their last frame)."""
    import av
    from lerobot.utils.io_utils import write_video

    clips = []
    for p in paths:
        with av.open(str(p)) as c:
            clips.append([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    n = max(len(c) for c in clips)
    frames = []
    sep = np.full((clips[0][0].shape[0], 6, 3), 255, dtype=np.uint8)
    for i in range(n):
        parts = []
        for c in clips:
            parts += [c[min(i, len(c) - 1)], sep]
        frames.append(np.concatenate(parts[:-1], axis=1))
    write_video(str(out), np.stack(frames), fps=fps)
