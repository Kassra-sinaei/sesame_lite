"""Per-row hard CBF-QP for the PandaOmron (base rectangle + end-effector sphere).

Port of the DEC-075 per-row structure of signal_3d ``safety/pink_filter.py``
without pink/pinocchio: the decision variable is the commanded velocity

    u = [vx, vy, wz, ex, ey, ez]

with the base twist (vx, vy, wz) and the end-effector linear velocity relative
to the base (ex, ey, ez), both in the robot base frame. Each admitted obstacle
box contributes one hard row per protected body, in the input-to-state-safe form
(Kolathaya and Ames; uncertainty_cvar.pdf eq. 8):

    grad_u h . u >= -alpha(h) h + eps |grad_u h|^2     (alpha = 1, or 10 when h < 0)

The boxes are the confidence boxes of the scene graph (faces pushed out by kappa*sigma + m_r + m_d, see
scene_graph.SceneGraph.with_confidence). The end effector and the arm spheres use the smooth point-box barrier, the
base its rectangle-box barrier. There is no slack and no cross-obstacle aggregation. Infeasibility falls back to a
penalised relaxation; a solver failure holds the tick (zero velocities) and the receipt says so. The evaluation side
uses the exact signed distances (``exact=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
from qpsolvers import solve_qp

from sesame_lite.geometry.box_geometry import (
    box_faces,
    signed_point_box_barrier,
    smooth_point_box_barrier,
    symmetric_polygon_separation_barrier,
)
from sesame_lite.geometry.fragments import OrientedBoxFragment, RawBarrierSample

CONTROL_DT_S = 0.05  # RoboCasa runs at 20 Hz
ACTION_DIM = 12
# lerobot flat action layout for RoboCasa (see lerobot.envs.robocasa.convert_action)
BASE_SLICE = slice(0, 3)  # forward, side, yaw velocity commands in [-1, 1]
TORSO_INDEX = 3
MODE_INDEX = 4  # < 0.5 arm mode, >= 0.5 base mode
EE_POS_SLICE = slice(5, 8)  # OSC_POSE delta in the base frame, output_max 0.05 m per step
EE_ROT_SLICE = slice(8, 11)
GRIPPER_INDEX = 11
EE_DELTA_MAX_M = 0.05
RELAXATION_PENALTY = 1e3  # per-row slack price of the best-effort fallback (velocity units: m/s per unit slack)


@dataclass(frozen=True)
class FilterPolicy:
    """Preregistered filter constants (same values as the paper's per-row policy)."""

    barrier_gain: float = 1.0  # alpha
    recovery_gain: float = 10.0
    recovery_cap_mps: float = 0.5  # a single row may demand at most this escape speed (keeps deep start-state penetrations feasible)
    disturbance_margin_mps: float = 0.05  # d_bar: measured steady-state base tracking residual (p50 0.04-0.07 m/s on the PandaOmron base)
    issf_eps: float = 0.025  # eps = d_bar / 2: the ISSf row term eps |grad_u h|^2 (makes m_d + the rest offset smallest)
    delta: float = 0.01  # risk of the confidence boxes, split over the admitted nodes, then over six faces
    smooth_tau_m: float = 0.005  # temperature of the smooth point-box barrier
    clearance_margin_m: float = 0.001
    uncertainty_margin_m: float = 0.0  # replaced by the per-face confidence box (kappa*sigma + m_r per face)
    goal_standoff_m: float = 0.065  # base goal clearance from the graph's boxes (the value used before the ISSf rows)
    eef_radius_m: float = 0.05
    smoothing_bias_m: float = 0.0005
    # RoboCasa's base is driven by MuJoCo velocity actuators (ctrlrange x kv) against a
    # joint frictionloss: steady speed = command*ctrlrange - frictionloss/kv, zero below
    # the deadband. The QP works in realized velocity; the mapping below inverts this.
    base_ctrlrange: tuple[float, float, float] = (1.0, 1.0, 1.5)  # m/s, m/s, rad/s per unit action
    base_deadband: tuple[float, float, float] = (0.25, 0.25, 250.0 / 1500.0)  # joint frictionloss / kv, per joint (verified in steady state)
    base_command_frame: str = "base"  # robosuite rotates the base command by the current yaw
    # Acceleration box bounds (as in the paper's per-row policy): the QP may only ask for velocities reachable
    # within one tick from the *realized* base velocity, so the escape it relies on is physically available.
    base_accel_limits: tuple[float, float, float] = (3.0, 3.0, 6.0)  # m/s^2, m/s^2, rad/s^2 (measured step response: ~3.7 m/s^2, ~8.8 rad/s^2)
    arm_radius_m: float = 0.08  # spheres on the Panda links (elbow to hand), protected while the arm is carried in base mode
    base_z_interval_m: tuple[float, float] = (0.0, 0.40)  # absolute world-z band of the base footprint (navigation family)
    weight_base: float = 1.0
    weight_eef: float = 1.0

    @property
    def policy_margin_m(self) -> float:
        return self.clearance_margin_m + self.uncertainty_margin_m

    @property
    def disturbance_offset_m(self) -> float:
        """m_d = d_bar^2 / (4 eps alpha): added to every confidence face, it restores the d = 0 guarantee (PDF Thm. 2)."""
        return self.disturbance_margin_mps**2 / (4.0 * self.issf_eps * self.barrier_gain)

    def smoothing_gain(self, n_rows: int) -> float:
        # kappa = log(6 * 10 * capacity) / bias, as in pink_filter.py
        return math.log(6.0 * 10.0 * max(1, n_rows)) / self.smoothing_bias_m


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def command_to_base_velocity(cmd_unit: np.ndarray, policy: FilterPolicy, theta: float = 0.0) -> np.ndarray:
    """Steady base velocity (vx, vy, wz), base frame, realized for a unit command in [-1, 1].

    RoboCasa's legacy base controller maps the (x, y) command through R(theta), theta = yaw - yaw at episode start,
    onto two world-fixed slide joints; each joint (and the yaw joint) loses ``base_deadband`` to friction.
    """
    cmd = np.clip(np.asarray(cmd_unit, dtype=float), -1.0, 1.0) * np.asarray(policy.base_ctrlrange)
    target = np.r_[_rot(theta) @ cmd[:2], cmd[2]]  # joint-space targets
    joint = np.sign(target) * np.maximum(np.abs(target) - np.asarray(policy.base_deadband), 0.0)
    return np.r_[_rot(theta).T @ joint[:2], joint[2]]


def base_velocity_to_command(v_base: np.ndarray, policy: FilterPolicy, theta: float = 0.0) -> np.ndarray:
    """Inverse of ``command_to_base_velocity`` (deadband compensation per joint); zero stays zero."""
    v = np.asarray(v_base, dtype=float)
    joint = np.r_[_rot(theta) @ v[:2], v[2]]
    target = np.where(np.abs(joint) > 1e-4, joint + np.sign(joint) * np.asarray(policy.base_deadband), 0.0)
    cmd = np.r_[_rot(theta).T @ target[:2], target[2]]
    return np.clip(cmd / np.asarray(policy.base_ctrlrange), -1.0, 1.0)


@dataclass(frozen=True)
class RobotState:
    base_position: np.ndarray  # world xyz of the base frame origin
    base_rotation: np.ndarray  # 3x3 world <- base
    eef_position: np.ndarray  # world xyz of the protected end-effector point
    base_rectangle_xy: np.ndarray  # 4x2 CCW footprint vertices in the base frame
    base_mode: bool
    base_velocity: np.ndarray | None = None  # realized (vx, vy, wz) in the base frame at this tick (diagnostics)
    yaw_axis_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))  # yaw joint axis in the base frame, relative to base_position
    arm_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # world centres of the arm-link spheres
    command_theta: float = 0.0  # yaw - yaw at episode start (frame of RoboCasa's base joints)

    @property
    def eef_offset_base(self) -> np.ndarray:
        return self.base_rotation.T @ (self.eef_position - self.base_position)


@dataclass
class FilterReceipt:
    tick: int
    outcome: str  # "passthrough" | "filtered" | "hold" | "passive"
    n_rows: int
    h_min: float | None
    h_min_row: str | None
    n_rows_negative: int
    below_margin: bool
    qp_status: str
    solve_ms: float
    tick_ms: float
    u_nominal: list[float]
    u_applied: list[float]
    correction_norm: float
    rows: list[dict] = field(default_factory=list)
    active_rows: list[tuple] = field(default_factory=list)  # (obstacle id, protected body, h, nearest face index)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["rows"] = list(self.rows)
        d.pop("active_rows")
        return d


class PerRowFilter:
    """Build one hard row per (protected body, admitted fragment) and solve the QP."""

    def __init__(self, policy: FilterPolicy = FilterPolicy(), *, record_rows: bool = False, couple_by_mode: bool = True):
        self.policy = policy
        self.record_rows = record_rows
        # Hybrid controller semantics: in base mode the arm is carried rigidly (its block is pinned to the nominal),
        # in arm mode the base is pinned; ``couple_by_mode=False`` frees both blocks (ablation).
        self.couple_by_mode = couple_by_mode
        self.hold_count = 0
        self.infeasible_count = 0
        self.relaxed_count = 0
        self.u_prev = np.zeros(6)  # last dispatched velocity (the robot starts at rest)

    # ----------------------------------------------------------------- kinematics

    def _base_vertex_jacobians(self, state: RobotState) -> list[np.ndarray]:
        # (vx, vy) translate the yaw joint; wz rotates about it (the PandaOmron yaw axis is 0.21 m behind the base center)
        R2 = state.base_rotation[:2, :2]
        jac = []
        for r in state.base_rectangle_xy:
            rr = r - state.yaw_axis_xy
            J = np.zeros((2, 6))
            J[:, 0] = R2[:, 0]
            J[:, 1] = R2[:, 1]
            J[:, 2] = R2 @ np.array([-rr[1], rr[0]])
            jac.append(J)
        return jac

    def _point_jacobian(self, state: RobotState, point: np.ndarray, *, arm_block: bool) -> np.ndarray:
        """d(point)/du for a point carried by the base (and, for the end effector, also moved by the arm block)."""
        R = state.base_rotation
        r = R.T @ (point - state.base_position) - np.array([*state.yaw_axis_xy, 0.0])
        J = np.zeros((3, 6))
        J[:, 0] = R[:, 0]
        J[:, 1] = R[:, 1]
        J[:, 2] = R @ np.array([-r[1], r[0], 0.0])
        if arm_block:
            J[:, 3:6] = R
        return J

    def _eef_jacobian(self, state: RobotState) -> np.ndarray:
        return self._point_jacobian(state, state.eef_position, arm_block=True)

    def base_vertices_world(self, state: RobotState) -> np.ndarray:
        R2 = state.base_rotation[:2, :2]
        return state.base_position[:2][None, :] + state.base_rectangle_xy @ R2.T

    # ----------------------------------------------------------------- rows

    def build_rows(
        self,
        state: RobotState,
        fragments: Mapping[str, OrientedBoxFragment],
        *,
        base_rows_for: Iterable[str],
        eef_rows_for: Iterable[str],
        speed_bound: dict[str, float] | None = None,
        skip_beyond_m: float | None = None,
        exact: bool = False,
    ) -> tuple[list[RawBarrierSample], np.ndarray, np.ndarray, list[int]]:
        """Return samples, the stacked constraint matrix G (rows x 6), the RHS b with G u <= b, and per row the index
        of the box face nearest to the protected body.

        Active set (exact): a row is skipped when a cheap lower bound h_lb on its barrier proves it slack for every
        velocity in the box, gain*h_lb - eps*Jn^2 >= the largest speed the protected body can reach (``speed_bound``),
        with Jn the spectral norm of the body's Jacobian (for the base, the largest over its four vertex Jacobians).
        h_lb is the AABB distance minus radius and margin (base and exact point rows), or for the smooth point barrier
        aabb_gap / sqrt(3) - tau log 6 - radius - margin. The QP solution is unchanged. ``skip_beyond_m`` instead skips
        rows whose lower bound exceeds a distance (ground-truth evaluation only needs nearby values). ``exact`` uses the
        exact signed point-box distance instead of the smooth barrier (the evaluation side).
        """

        p = self.policy
        base_ids = [i for i in base_rows_for if i in fragments]
        eef_ids = [i for i in eef_rows_for if i in fragments]
        kappa = p.smoothing_gain(len(base_ids) + len(eef_ids))
        samples: list[RawBarrierSample] = []
        faces: list[int] = []
        G_rows: list[np.ndarray] = []
        b_rows: list[float] = []

        def skip(lb_h: float, body: str, jac_norm: float) -> bool:
            if skip_beyond_m is not None:
                return lb_h > skip_beyond_m
            if speed_bound is not None:
                return p.barrier_gain * lb_h - p.issf_eps * jac_norm**2 >= speed_bound[body] + 1e-6
            return False

        def aabb_gap(points: np.ndarray, frag) -> float:
            lo, hi = frag.aabb(points.shape[1])
            return float(np.min(np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0), axis=1)))

        if base_ids:
            verts = self.base_vertices_world(state)
            vjac = self._base_vertex_jacobians(state)
            base_jn = max(float(np.linalg.norm(J, 2)) for J in vjac)
            c = verts.mean(axis=0)
            r_disc = float(np.max(np.linalg.norm(verts - c, axis=1)))
            for fid in base_ids:
                frag = fragments[fid]
                lo, hi = frag.aabb(3)
                z0, z1 = p.base_z_interval_m
                if hi[2] < z0 or lo[2] > z1:
                    continue  # vertically disjoint from the base band
                # lower bound: distance from the footprint's bounding disc to the obstacle's AABB
                if skip(aabb_gap(c[None, :], frag) - r_disc - p.policy_margin_m, "base", base_jn):
                    continue
                sample, grad = symmetric_polygon_separation_barrier(verts, vjac, frag, policy_margin=p.policy_margin_m, smoothing_gain=kappa, protected_id="base")
                A, b = box_faces(frag)
                samples.append(sample)
                faces.append(int(np.argmax(A[:, :2] @ c - b)))  # the face the footprint centre is furthest outside of
                grad_u = grad.reshape(6)
                G_rows.append(-grad_u)
                b_rows.append(self._rhs(sample.value, grad_u))

        points = [("eef", state.eef_position, p.eef_radius_m, True)]
        if state.base_mode:  # arm carried rigidly: its link spheres move with the base twist only
            points += [(f"arm{k}", pt, p.arm_radius_m, False) for k, pt in enumerate(state.arm_points)]
        for name, point, radius, arm_block in points if eef_ids else ():
            J = self._point_jacobian(state, point, arm_block=arm_block)
            jn = float(np.linalg.norm(J, 2))
            for fid in eef_ids:
                frag = fragments[fid]
                gap = aabb_gap(point[None, :], frag)
                if exact:
                    if skip(gap - radius - p.policy_margin_m, name, jn):
                        continue
                    sample = signed_point_box_barrier(point, frag, protected_radius=radius, policy_margin=p.policy_margin_m, protected_id=name)
                    A, b = box_faces(frag)
                    face = int(np.argmax(A @ point - b))
                else:
                    if skip(gap / math.sqrt(3.0) - p.smooth_tau_m * math.log(6.0) - radius - p.policy_margin_m, name, jn):
                        continue
                    sample, face = smooth_point_box_barrier(point, frag, protected_radius=radius, policy_margin=p.policy_margin_m, tau=p.smooth_tau_m, protected_id=name)
                samples.append(sample)
                faces.append(face)
                grad_u = np.asarray(sample.point_jacobian) @ J
                G_rows.append(-grad_u)
                b_rows.append(self._rhs(sample.value, grad_u))

        if not G_rows:
            return samples, np.zeros((0, 6)), np.zeros(0), faces
        return samples, np.vstack(G_rows), np.asarray(b_rows), faces

    def speed_bounds(self, state: RobotState, lb: np.ndarray, ub: np.ndarray) -> dict[str, float]:
        """Largest speed each protected body can reach for any velocity in the box [lb, ub]."""
        vlin = float(np.hypot(max(abs(lb[0]), abs(ub[0])), max(abs(lb[1]), abs(ub[1]))))
        w = max(abs(lb[2]), abs(ub[2]))
        vee = float(np.linalg.norm(np.maximum(np.abs(lb[3:6]), np.abs(ub[3:6]))))
        axis = np.array([*state.yaw_axis_xy, 0.0])
        r_base = float(np.max(np.linalg.norm(state.base_rectangle_xy - state.yaw_axis_xy, axis=1)))
        out = {"base": vlin + w * r_base}
        for name, point, arm_block in [("eef", state.eef_position, True)] + [(f"arm{k}", pt, False) for k, pt in enumerate(state.arm_points)]:
            r = state.base_rotation.T @ (point - state.base_position) - axis
            out[name] = vlin + w * float(np.hypot(r[0], r[1])) + (vee if arm_block else 0.0)
        return out

    def _rhs(self, h: float, grad_u: np.ndarray) -> float:
        """ISSf row grad_u . u >= -gain*h + eps |grad_u|^2, as -grad_u . u <= b: b = max(gain*h - eps |grad_u|^2, -cap)."""
        p = self.policy
        gain = p.barrier_gain if h >= 0.0 else p.recovery_gain
        return max(gain * h - p.issf_eps * float(np.dot(grad_u, grad_u)), -p.recovery_cap_mps)

    # ----------------------------------------------------------------- action <-> velocity

    def nominal_velocity(self, action: np.ndarray, state: RobotState) -> np.ndarray:
        p = self.policy
        a = np.asarray(action, dtype=float)
        v_base = command_to_base_velocity(a[BASE_SLICE], p, state.command_theta)
        if p.base_command_frame == "world":
            R2 = state.base_rotation[:2, :2]
            v_base = np.concatenate([R2.T @ v_base[:2], v_base[2:3]])
        e = np.clip(a[EE_POS_SLICE], -1.0, 1.0) * EE_DELTA_MAX_M / CONTROL_DT_S
        return np.concatenate([v_base, e])

    def velocity_to_action(self, u: np.ndarray, action: np.ndarray, state: RobotState) -> np.ndarray:
        p = self.policy
        out = np.asarray(action, dtype=float).copy()
        v_base = np.asarray(u[:3], dtype=float)
        if p.base_command_frame == "world":
            R2 = state.base_rotation[:2, :2]
            v_base = np.concatenate([R2 @ v_base[:2], v_base[2:3]])
        out[BASE_SLICE] = base_velocity_to_command(v_base, p, state.command_theta)
        out[EE_POS_SLICE] = np.clip(np.asarray(u[3:6]) * CONTROL_DT_S / EE_DELTA_MAX_M, -1.0, 1.0)
        return out

    def _bounds(self, state: RobotState, u_nom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity box; the block the hybrid controller is not steering is pinned to its nominal value.

        In base mode the arm goal is carried along with the base, so only the
        base twist is a decision variable; in arm mode only the end-effector
        velocity is. The pinned block keeps whatever the policy commanded.
        """

        p = self.policy
        reach = np.asarray(p.base_ctrlrange) - np.asarray(p.base_deadband)
        # the slide joints are world-fixed: a body-frame box inscribed in the joint box for every heading
        ub = np.array([reach[0] / np.sqrt(2), reach[1] / np.sqrt(2), reach[2]] + [EE_DELTA_MAX_M / CONTROL_DT_S] * 3)
        lb = -ub
        if self.u_prev is not None:  # acceleration box around the previously *commanded* base velocity (confirmed dispatch, as in the paper)
            dv = np.asarray(p.base_accel_limits) * CONTROL_DT_S
            lb[:3] = np.maximum(lb[:3], self.u_prev[:3] - dv)
            ub[:3] = np.minimum(ub[:3], self.u_prev[:3] + dv)
            lb[:3], ub[:3] = np.minimum(lb[:3], ub[:3]), np.maximum(lb[:3], ub[:3])
        if self.couple_by_mode:
            fixed = slice(3, 6) if state.base_mode else slice(0, 3)
            lb[fixed] = u_nom[fixed]
            ub[fixed] = u_nom[fixed]
        return lb, ub

    # ----------------------------------------------------------------- main entry

    def filter_action(
        self,
        *,
        tick: int,
        action: np.ndarray,
        state: RobotState,
        fragments: Mapping[str, OrientedBoxFragment],
        base_rows_for: Sequence[str],
        eef_rows_for: Sequence[str],
        passive: bool = False,
    ) -> tuple[np.ndarray, FilterReceipt]:
        t0 = time.perf_counter()
        p = self.policy
        u_nom = self.nominal_velocity(action, state)
        lb, ub = self._bounds(state, u_nom)
        samples, G, b, nearest_faces = self.build_rows(state, fragments, base_rows_for=base_rows_for, eef_rows_for=eef_rows_for, speed_bound=None if passive else self.speed_bounds(state, lb, ub))
        hs = [s.value for s in samples]
        h_min = min(hs) if hs else None
        h_min_row = samples[int(np.argmin(hs))].constraint_id if hs else None
        n_neg = sum(1 for h in hs if h < 0.0)
        below_margin = bool(h_min is not None and h_min < -p.policy_margin_m)
        rows = [s.as_dict() for s in samples] if self.record_rows else []

        if passive or G.shape[0] == 0:
            u = u_nom
            outcome = "passive" if passive else "passthrough"
            status = "none"
            solve_ms = 0.0
        else:
            W = np.diag([p.weight_base] * 3 + [p.weight_eef] * 3)
            t1 = time.perf_counter()
            # Rows with h < 0 carry the recovery gain, so the QP itself drives the
            # state back above the margin; the tick is held only when no velocity
            # satisfies every row (fail closed).
            try:
                sol = solve_qp(W, -W @ u_nom, G, b, lb=lb, ub=ub, solver="daqp")
            except Exception as exc:  # solver-level failure
                sol, status = None, f"solver_error:{type(exc).__name__}"
            else:
                status = "optimal" if sol is not None else "infeasible"
            if sol is None:
                # Best-effort recovery: the strict rows cannot all be met within the acceleration box (typically an
                # escape demand from rest). Minimise the row violations with penalised slack instead of freezing,
                # so the robot still accelerates out of the unsafe set; the tick is reported as "relaxed".
                self.infeasible_count += 1
                m = G.shape[0]
                P = np.block([[W, np.zeros((6, m))], [np.zeros((m, 6)), 1e-6 * np.eye(m)]])
                q = np.concatenate([-W @ u_nom, RELAXATION_PENALTY * np.ones(m)])
                try:
                    sol = solve_qp(P, q, np.hstack([G, -np.eye(m)]), b, lb=np.concatenate([lb, np.zeros(m)]), ub=np.concatenate([ub, np.full(m, np.inf)]), solver="daqp")
                except Exception:
                    sol = None
                if sol is None:
                    u, status = np.zeros(6), status if status != "infeasible" else "relaxation_failed"
                else:
                    u, status = np.asarray(sol[:6], dtype=float), "relaxed"
            else:
                u = np.asarray(sol, dtype=float)
                # Post-check: every row satisfied up to a small tolerance.
                if np.any(G @ u - b > 1e-6):
                    u, status = np.zeros(6), "postcheck_failed"
            solve_ms = (time.perf_counter() - t1) * 1e3
            if status == "optimal":
                outcome = "filtered" if np.linalg.norm(u - u_nom) > 1e-9 else "passthrough"
            elif status == "relaxed":
                outcome = "relaxed"
                self.relaxed_count += 1
            else:
                outcome = "hold"
                self.hold_count += 1

        applied = self.velocity_to_action(u, action, state) if not passive else np.asarray(action, dtype=float)
        self.u_prev = np.asarray(u, dtype=float).copy()
        receipt = FilterReceipt(
            tick=tick,
            outcome=outcome,
            n_rows=int(G.shape[0]),
            h_min=h_min,
            h_min_row=h_min_row,
            n_rows_negative=n_neg,
            below_margin=below_margin,
            qp_status=status,
            solve_ms=solve_ms,
            tick_ms=(time.perf_counter() - t0) * 1e3,
            u_nominal=[float(v) for v in u_nom],
            u_applied=[float(v) for v in u],
            correction_norm=float(np.linalg.norm(u - u_nom)),
            rows=rows,
            active_rows=[(smp.obstacle_parent_id, smp.protected_id, float(smp.value), f) for smp, f in zip(samples, nearest_faces)],
        )
        return applied, receipt
