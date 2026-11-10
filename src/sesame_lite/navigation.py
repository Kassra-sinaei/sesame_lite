"""Point-to-point navigation of the PandaOmron holonomic base.

The planner sees a static map: the collision geoms of the kitchen fixtures (a prior map, as on the real
robot; movable objects are left to the safety filter). The map and the robot's own collision geoms (at the
current arm configuration) are sliced into height bands, so an arm reaching over a counter is not a
collision while a base touching the counter plinth is. The configuration-space obstacle of each yaw bin is
the union over bands of the band map dilated by the rotated band footprint (Lozano-Perez C-space), A* on the
(x, y, yaw) lattice with a clearance cost gives the path, and a holonomic carrot tracker follows it. A missing
path is reported, never replaced by a straight line.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from shapely.geometry import MultiPoint
from skimage.draw import polygon as raster_polygon
from skimage.draw import polygon_perimeter

from .filter import MODE_INDEX, BaseModel, FilterPolicy
from .scene_graph import _collision_geoms, _geom_corners_local, _raw_data, _raw_model, _subtree_body_ids

BASE_BODY = "mobilebase0_base"  # NavigateKitchen's success check and RoboCasa's placement poses use this frame
Z_BANDS = (0.005, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # band edges, m; below the first edge is floor (fridge housings have 2 cm floor plates)
FREE_JOINT = 0


class NoPathError(RuntimeError):
    pass


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def base_pose(rs_env) -> np.ndarray:
    """(x, y, yaw) of the mobile base, read from MuJoCo (robosuite's observation buffer lags after reset)."""
    m, d = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    R = d.xmat[b].reshape(3, 3)
    return np.array([d.xpos[b][0], d.xpos[b][1], math.atan2(R[1, 0], R[0, 0])])


def _robot_body_ids(m) -> set[int]:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    while int(m.body_parentid[b]) != 0:
        b = int(m.body_parentid[b])
    return _subtree_body_ids(m, b)


def _movable_body_ids(m) -> set[int]:
    roots = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == FREE_JOINT}
    return set().union(*(_subtree_body_ids(m, r) for r in roots)) if roots else set()


def _geom_boxes(m, d, gids) -> list[tuple[float, float, np.ndarray]]:
    """(z_lo, z_hi, 8x3 world corners of the geom's local AABB) per geom."""
    out = []
    for g in gids:
        c = _geom_corners_local(m, g) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g][None, :]
        out.append((float(c[:, 2].min()), float(c[:, 2].max()), c))
    return out


def _bands_of(z_lo: float, z_hi: float) -> list[int]:
    return [b for b in range(len(Z_BANDS) - 1) if z_hi > Z_BANDS[b] and z_lo < Z_BANDS[b + 1]]


def _hull(points_xy: np.ndarray) -> np.ndarray | None:
    h = MultiPoint([tuple(p) for p in points_xy]).convex_hull
    return np.asarray(h.exterior.coords) if h.geom_type == "Polygon" else None


@dataclass
class StaticMap:
    """Occupancy of the fixture collision geoms, one boolean grid per height band (x index first)."""

    origin: np.ndarray  # world xy of cell (0, 0)
    resolution: float
    occupancy: np.ndarray  # (n_bands, nx, ny)

    @classmethod
    def from_sim(cls, rs_env, resolution: float = 0.04, margin_m: float = 0.5) -> "StaticMap":
        m, d = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
        skip = _robot_body_ids(m) | _movable_body_ids(m)
        boxes = [b for b in _geom_boxes(m, d, _collision_geoms(m, set(range(m.nbody)) - skip)) if b[1] > Z_BANDS[0]]
        xy = np.vstack([c[:, :2] for _, _, c in boxes])
        origin = xy.min(axis=0) - margin_m
        nx, ny = (np.ceil((xy.max(axis=0) + margin_m - origin) / resolution).astype(int) + 1).tolist()
        occ = np.zeros((len(Z_BANDS) - 1, nx, ny), dtype=bool)
        for z_lo, z_hi, c in boxes:
            bands = _bands_of(z_lo, z_hi)
            hull = _hull(c[:, :2]) if bands else None
            if hull is None:
                continue
            ij = (hull - origin) / resolution
            rr, cc = raster_polygon(ij[:, 0], ij[:, 1], shape=(nx, ny))
            pr, pc = polygon_perimeter(ij[:, 0], ij[:, 1], shape=(nx, ny), clip=True)  # thin geoms (walls, panels) still block
            for b in bands:
                occ[b, rr, cc] = True
                occ[b, pr, pc] = True
        return cls(origin, resolution, occ)

    def cell(self, xy) -> tuple[int, int]:
        c = np.round((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        return int(c[0]), int(c[1])

    def world(self, i, j) -> np.ndarray:
        return self.origin + self.resolution * np.array([i, j], dtype=float)


def robot_footprint(rs_env) -> list[np.ndarray | None]:
    """Convex hull, in the base frame, of the robot's collision geoms in each height band (None if empty)."""
    m, d = _raw_model(rs_env.sim), _raw_data(rs_env.sim)
    x, y, yaw = base_pose(rs_env)
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    per_band: list[list[np.ndarray]] = [[] for _ in range(len(Z_BANDS) - 1)]
    for z_lo, z_hi, c in _geom_boxes(m, d, _collision_geoms(m, _robot_body_ids(m))):
        local = (c[:, :2] - [x, y]) @ R
        for b in _bands_of(z_lo, z_hi):
            per_band[b].append(local)
    return [_hull(np.vstack(p)) if p else None for p in per_band]


class ConfigurationSpace:
    """C-space obstacle and clearance (m) of every yaw bin over a static map, for one robot footprint.

    A lattice state is usable when its clearance is at least ``min_clearance_m``: the map is certified only to a
    cell and a yaw bin, and the tracker stops within its tolerance, so a zero-clearance state can still touch.
    """

    def __init__(self, smap: StaticMap, footprint: list[np.ndarray | None], n_yaw: int = 32, min_clearance_m: float = 0.06):
        self.map, self.n_yaw = smap, n_yaw
        self.yaws = 2 * np.pi * np.arange(n_yaw) / n_yaw - np.pi
        res = smap.resolution
        reach = max(np.linalg.norm(f, axis=1).max() for f in footprint if f is not None)
        half = int(math.ceil(reach / res)) + 1
        self.obstacle = np.zeros((n_yaw,) + smap.occupancy.shape[1:], dtype=bool)
        for k, yaw in enumerate(self.yaws):
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            for b, fp in enumerate(footprint):
                if fp is None or not smap.occupancy[b].any():
                    continue
                kernel = np.zeros((2 * half + 1, 2 * half + 1))
                ij = fp @ R.T / res + half
                rr, cc = raster_polygon(ij[:, 0], ij[:, 1], shape=kernel.shape)
                kernel[rr, cc] = 1.0
                # robot at cell c hits the band iff some occupied cell lies under its footprint: correlate
                self.obstacle[k] |= fftconvolve(smap.occupancy[b].astype(float), kernel[::-1, ::-1], mode="same") > 0.5
        self.clearance = np.stack([distance_transform_edt(~o) * res for o in self.obstacle])
        self.min_clearance = min_clearance_m
        self.blocked = self.clearance < min_clearance_m

    def yaw_bin(self, yaw: float) -> int:
        return int(np.round((wrap(yaw) + np.pi) / (2 * np.pi / self.n_yaw))) % self.n_yaw

    def state(self, pose) -> tuple[int, int, int]:
        i, j = self.map.cell(pose[:2])
        return self.yaw_bin(pose[2]), i, j

    def inside(self, s) -> bool:
        return 0 <= s[1] < self.obstacle.shape[1] and 0 <= s[2] < self.obstacle.shape[2]

    def clearance_at(self, pose) -> float:
        s = self.state(pose)
        return float(self.clearance[s]) if self.inside(s) else 0.0


def project_free(cspace: ConfigurationSpace, pose, radius_m: float = 0.15) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """Nearest collision-free lattice state to ``pose`` at its yaw bin, within ``radius_m``, and its world pose.

    RoboCasa places the robot a centimetre or two from the fixture it faces (and sometimes from a side wall),
    closer than the lattice can certify. The projection is the smallest displacement to a state with the
    required clearance; None if there is none within ``radius_m``.
    """
    k, i, j = cspace.state(pose)
    n = int(math.ceil(radius_m / cspace.map.resolution))
    cells = [(di * di + dj * dj, i + di, j + dj) for di in range(-n, n + 1) for dj in range(-n, n + 1) if di * di + dj * dj <= n * n]
    for _, ci, cj in sorted(cells):
        s = (k, ci, cj)
        if cspace.inside(s) and not cspace.blocked[s]:
            xy = np.asarray(pose[:2], dtype=float) if (ci, cj) == (i, j) else cspace.map.world(ci, cj)
            return np.r_[xy, pose[2]], s
    return None


def plan_path(cspace: ConfigurationSpace, start, goal, *, rotation_radius_m: float = 0.5, clearance_ref_m: float = 0.25, clearance_weight: float = 4.0, projection_radius_m: float = 0.15) -> np.ndarray:
    """Lowest-cost (x, y, yaw) path on the C-space lattice: 8-connected moves at fixed yaw and +-1 yaw-bin turns.

    Step cost: SE(2) length (turns weighted by ``rotation_radius_m``) times 1 + w (1 - c / c_ref)^2 below the
    reference clearance c_ref, which keeps the path away from furniture where it can. Start and goal are first
    projected onto the free lattice (``project_free``); the path starts at the true start and ends at the
    projected goal. Raises ``NoPathError`` when a projection or the search fails.
    """
    projected = {}
    for name, pose in (("start", start), ("goal", goal)):
        p = project_free(cspace, pose, projection_radius_m)
        if p is None:
            raise NoPathError(f"{name} pose {np.round(pose, 3).tolist()} has no collision-free lattice state within {projection_radius_m} m")
        projected[name] = p
    (start_free, s0), (goal_free, g) = projected["start"], projected["goal"]
    res, dyaw = cspace.map.resolution, 2 * np.pi / cspace.n_yaw
    pen = 1.0 + clearance_weight * np.clip(1.0 - cspace.clearance / clearance_ref_m, 0.0, 1.0) ** 2
    moves = [(0, di, dj, res * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    moves += [(dk, 0, 0, rotation_radius_m * dyaw) for dk in (-1, 1)]
    gxy = cspace.map.world(g[1], g[2])

    def h(s):
        dk = abs(s[0] - g[0]) % cspace.n_yaw
        return float(np.linalg.norm(cspace.map.world(s[1], s[2]) - gxy)) + rotation_radius_m * dyaw * min(dk, cspace.n_yaw - dk)

    best, came, heap = {s0: 0.0}, {}, [(h(s0), 0.0, s0)]
    nk, nx, ny = cspace.obstacle.shape
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == g:
            break
        if cost > best[cur]:
            continue
        for dk, di, dj, length in moves:
            nxt = ((cur[0] + dk) % nk, cur[1] + di, cur[2] + dj)
            if not (0 <= nxt[1] < nx and 0 <= nxt[2] < ny) or cspace.blocked[nxt]:
                continue
            c = cost + length * 0.5 * (pen[cur] + pen[nxt])
            if c < best.get(nxt, math.inf):
                best[nxt], came[nxt] = c, cur
                heapq.heappush(heap, (c + h(nxt), c, nxt))
    else:
        raise NoPathError("no collision-free path between start and goal on the static map")
    states = [g]
    while states[-1] != s0:
        states.append(came[states[-1]])
    states.reverse()
    path = np.array([[*cspace.map.world(i, j), cspace.yaws[k]] for k, i, j in states])
    path[0], path[-1] = start_free, goal_free
    path[:, 2] = np.unwrap(path[:, 2])
    path = _shortcut(cspace, path, clearance_ref_m)
    if np.linalg.norm(start_free[:2] - np.asarray(start)[:2]) > 1e-9:  # ease off the fixture the robot is parked at
        path = np.vstack([np.asarray(start, dtype=float), path])
        path[:, 2] = np.unwrap(path[:, 2])
    return path


def _segment_ok(cspace: ConfigurationSpace, a, b, floor_m: float) -> bool:
    n = max(2, int(math.ceil(max(np.linalg.norm(b[:2] - a[:2]) / (0.5 * cspace.map.resolution), abs(b[2] - a[2]) / (0.5 * 2 * np.pi / cspace.n_yaw)))) + 1)
    for t in np.linspace(0.0, 1.0, n):
        s = cspace.state(a + t * (b - a))
        if not cspace.inside(s) or cspace.blocked[s] or cspace.clearance[s] < floor_m:
            return False
    return True


def _shortcut(cspace: ConfigurationSpace, path: np.ndarray, clearance_ref_m: float) -> np.ndarray:
    """Greedy line-of-sight shortcuts that keep the clearance the lattice path had (or c_ref, whichever is smaller)."""
    clear = np.array([cspace.clearance_at(p) for p in path])
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _segment_ok(cspace, path[i], path[j], min(clearance_ref_m, clear[i : j + 1].min()) - 1e-9):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


class HolonomicTracker:
    """Carrot tracker for the holonomic base on an (x, y, yaw) path; returns the 12-D RoboCasa action.

    The robot is projected onto the path (monotone in arc length), the carrot sits ``lookahead_m`` ahead in the
    SE(2) metric, and the commanded twist points at the carrot with the speed of a trapezoidal profile
    (``v_max``, ``a_max``) on the remaining length. Inside ``lookahead_m`` of the goal it regulates the pose.
    Velocities are realized m/s and rad/s, mapped through the base actuator model (deadband, legacy frame).
    """

    def __init__(self, path: np.ndarray, *, init_yaw: float, policy: FilterPolicy = FilterPolicy(), rotation_radius_m: float = 0.5,
                 v_max: float = 0.6, a_max: float = 0.8, lookahead_m: float = 0.25, k_goal: float = 2.0, tol_xy_m: float = 0.03, tol_yaw_rad: float = 0.03):
        self.path, self.init_yaw, self.policy, self.r = np.asarray(path, dtype=float), init_yaw, policy, rotation_radius_m
        self.base = BaseModel()
        self.v_max, self.a_max, self.lookahead, self.k_goal, self.tol_xy, self.tol_yaw = v_max, a_max, lookahead_m, k_goal, tol_xy_m, tol_yaw_rad
        w = self.path * [1.0, 1.0, self.r]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(w, axis=0), axis=1))]
        self.progress = 0.0

    def _at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.path[:, c]) for c in range(3)])

    def _distance(self, q, pose) -> float:
        return math.hypot(np.linalg.norm(q[:2] - pose[:2]), self.r * wrap(q[2] - pose[2]))

    def _project(self, pose) -> float:
        """Closest arc length within 1 m ahead of the current progress (no backward jumps)."""
        candidates = np.linspace(self.progress, min(self.s[-1], self.progress + 1.0), 41)
        return float(min(candidates, key=lambda s: self._distance(self._at(s), pose)))

    def deviation(self, pose) -> float:
        q = self._at(self._project(pose))
        return float(np.linalg.norm(q[:2] - pose[:2]))

    def at_goal(self, pose) -> bool:
        g = self.path[-1]
        return np.linalg.norm(g[:2] - pose[:2]) <= self.tol_xy and abs(wrap(g[2] - pose[2])) <= self.tol_yaw

    def __call__(self, pose) -> np.ndarray:
        self.progress = self._project(pose)
        remaining = self.s[-1] - self.progress
        target = self._at(min(self.s[-1], self.progress + self.lookahead))
        e = np.r_[target[:2] - pose[:2], self.r * wrap(target[2] - pose[2])]
        if remaining <= self.lookahead:
            e = np.r_[self.path[-1, :2] - pose[:2], self.r * wrap(self.path[-1, 2] - pose[2])]
            speed = min(self.v_max, self.k_goal * np.linalg.norm(e), math.sqrt(2 * self.a_max * np.linalg.norm(e)))
        else:
            speed = min(self.v_max, math.sqrt(2 * self.a_max * remaining))
        v = np.zeros(3) if self.at_goal(pose) else speed * e / max(np.linalg.norm(e), 1e-9)
        c, s_ = math.cos(pose[2]), math.sin(pose[2])
        twist = np.array([c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1], v[2] / self.r])  # world -> base frame
        a = np.zeros(12, dtype=np.float32)
        a[0:3] = self.base.steady_command(twist, pose[2] - self.init_yaw)
        a[MODE_INDEX] = 1.0
        return a
