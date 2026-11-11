"""The safety filter of spec v4 (PDF eq. 5, 6, 8, 9): robot spheres against the objects' confidence balls.

For every robot sphere i (centre s_i(q), radius r_i, Jacobian J_i; robot_spheres.py) and every admitted object j
(points p_m with radii rho_m; uncertainty.balls):

    g_m    = |s_i - p_m| - rho_m
    psi    = -tau ln sum_m exp(-g_m / tau)                     eq. (5)
    h      = psi - r_i,   grad_q h = (sum_m pi_m nu_m) J_i      eq. (6), pi = softmax(-g / tau), nu_m = (s_i - p_m) / |.|
    row:   grad_c h . u + grad_k h . q_dot_k - eps |grad_c h|^2 >= -alpha h                    eq. (8)
    QP:    u* = argmin |u - u_nom|^2  s.t. every row, u in U                                    eq. (9)

u = the joint velocities of the base (forward slide, side slide, yaw) and the seven arm joints (columns c);
the torso and the two fingers are not commanded: they enter each row with their nominal velocity (columns k).
U is a box: for the base the mean joint velocities its controller can reach within the tick from the current velocity
(``BaseModel.bounds``); for the arm the joint velocity limits, narrowed so that no joint passes its position limit
within the tick (``SphereFilter.box``). All pairs are computed on the GPU. A row that no u in U
can violate is left out of the solver; this changes no solution and is counted (``n_rows_qp``).

Feasibility (spec 6.3): if the QP is infeasible, it is solved again with one slack per row, priced by
``relaxation_penalty``; every row that needed slack is recorded. If the relaxed QP fails as well, or a solution
violates a row it was solved for, the tick holds (u = 0) and the receipt says why. Nothing else is done to a tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import daqp
import numpy as np
import torch

CONTROL_DT_S = 0.05  # RoboCasa runs at 20 Hz
ACTION_DIM = 12
# lerobot flat action layout for RoboCasa (see lerobot.envs.robocasa.convert_action)
BASE_SLICE = slice(0, 3)  # forward, side, yaw velocity commands in [-1, 1]
TORSO_INDEX = 3
MODE_INDEX = 4  # < 0.5 arm mode, >= 0.5 base mode
EE_POS_SLICE = slice(5, 8)  # OSC_POSE delta in the base frame, output_max 0.05 m per step
EE_ROT_SLICE = slice(8, 11)
GRIPPER_INDEX = 11
COMMANDED = np.r_[0:3, 4:11]  # columns of u in q = [base 3, torso, arm 7, fingers 2]
KNOWN = np.r_[3, 11, 12]  # torso and fingers
ARM_Q = np.r_[4:11]  # the arm joints in q
DAQP_EXIT = {1: "optimal", 2: "soft_optimal", -1: "infeasible", -2: "cycling", -3: "unbounded", -4: "iteration_limit", -5: "nonconvex", -6: "overdetermined_start"}


@dataclass(frozen=True)
class FilterPolicy:
    """Filter constants. The first block is the PDF's (Numbers, Theorem 2), tuned; the rest describes the robot."""

    # tuned (spec step 4, Kasra 2026-11-11; PDF start values alpha 1, d_bar 0.05, tau 2 mm): the filter was too
    # conservative; d_bar from the measured tracking disturbance (99 % below 0.035 in pick_place_far)
    alpha: float = 2.0  # class-K gain of eq. (7)-(9)
    d_bar: float = 0.03  # bound on the tracking disturbance |d|, q_dot = u + d
    tau_m: float = 0.0005  # temperature of the log-sum-exp (eq. 5)
    delta: float = 0.01  # map risk, split over the admitted objects, then over 2 |P_j| (eq. 4)
    relaxation_penalty: float = 1e3  # price per unit slack of a relaxed row (infeasible QP, section 6.3)
    goal_standoff_m: float = 0.065  # base goal clearance from the balls (the value used before; planning only)
    arm_velocity_limits: tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)  # rad/s, Franka datasheet

    @property
    def eps(self) -> float:
        """eps = d_bar / 2 makes the rest offset smallest (PDF, Theorem 2)."""
        return self.d_bar / 2.0

    @property
    def m_d(self) -> float:
        """m_d = d_bar^2 / (4 eps alpha), added to every ball radius (Theorem 2): 0.75 cm with the tuned values."""
        return self.d_bar**2 / (4.0 * self.eps * self.alpha)


# ---------------------------------------------------------------------- the base: command <-> joint velocities

BASE_RESPONSE = Path(__file__).resolve().parents[2] / "config" / "base_response.json"


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class BaseModel:
    """RoboCasa's base as scripts/calibrate_base.py measured it (config/base_response.json).

    robosuite's base controller turns the command (base frame) by theta = yaw - yaw at episode start onto the two slide
    joints (axes fixed in the start frame) and the yaw joint; each joint's velocity actuator gets the target
    command x ctrlrange. Per joint, the measured steady velocity steady(target) and the lag beta over one tick:

        mean joint velocity over the tick = beta * velocity at its start + (1 - beta) * steady(target)

    ``nominal`` is that mean for the policy's command (u_nom of the base), ``bounds`` the means reachable in one tick
    (slide targets within ctrlrange / sqrt(2), which every heading can command; yaw within ctrlrange), and
    ``command`` the command whose predicted mean is u (u within the bounds)."""

    def __init__(self, path: Path = BASE_RESPONSE):
        cal = json.loads(Path(path).read_text())["joints"]
        self.cr = np.array([j["ctrlrange"] for j in cal.values()])
        self.beta = np.array([j["beta"] for j in cal.values()])
        self.targets = [np.asarray(j["targets"]) for j in cal.values()]
        self.steady_v = [np.maximum.accumulate(np.asarray(j["steady"])) for j in cal.values()]
        self.target_max = np.r_[self.cr[:2] / np.sqrt(2.0), self.cr[2]]

    def steady(self, target: np.ndarray) -> np.ndarray:
        t = np.asarray(target, float)
        return np.array([np.sign(t[j]) * np.interp(abs(t[j]), self.targets[j], self.steady_v[j]) for j in range(3)])

    def target_for(self, v_steady: np.ndarray) -> np.ndarray:
        """Inverse of ``steady`` (the smallest target that reaches the velocity)."""
        v = np.asarray(v_steady, float)
        out = np.zeros(3)
        for j in range(3):
            if abs(v[j]) > self.steady_v[j][-1] + 1e-9:
                raise ValueError(f"base joint {j}: steady velocity {v[j]} is beyond the measured range {self.steady_v[j][-1]}")
            k = int(np.searchsorted(self.steady_v[j], abs(v[j]), side="left"))  # first grid point at or above |v|
            if k == 0:
                continue
            v0, v1, t0, t1 = self.steady_v[j][k - 1], self.steady_v[j][k], self.targets[j][k - 1], self.targets[j][k]
            out[j] = np.sign(v[j]) * (t0 + (t1 - t0) * (abs(v[j]) - v0) / max(v1 - v0, 1e-12))
        return out

    def joint_targets(self, cmd: np.ndarray, theta: float) -> np.ndarray:
        cmd = np.clip(np.asarray(cmd, float), -1.0, 1.0) * self.cr
        return np.clip(np.r_[_rot(theta) @ cmd[:2], cmd[2]], -self.cr, self.cr)

    def nominal(self, cmd: np.ndarray, theta: float, v0: np.ndarray) -> np.ndarray:
        return self.beta * v0 + (1.0 - self.beta) * self.steady(self.joint_targets(cmd, theta))

    def bounds(self, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hi = self.beta * v0 + (1.0 - self.beta) * self.steady(self.target_max)
        lo = self.beta * v0 - (1.0 - self.beta) * self.steady(self.target_max)
        return lo, hi

    def command(self, u: np.ndarray, theta: float, v0: np.ndarray) -> np.ndarray:
        target = self.target_for((np.asarray(u, float) - self.beta * v0) / (1.0 - self.beta))
        if np.any(np.abs(target) > self.target_max + 1e-9):
            raise ValueError(f"base joint velocity {u} is outside the reachable box (targets {target})")
        return np.r_[_rot(theta).T @ target[:2], target[2]] / self.cr

    def steady_command(self, v_base: np.ndarray, theta: float) -> np.ndarray:
        """Command for a steady base-frame velocity (vx, vy, wz): the navigation follower's output, clipped to [-1, 1]."""
        v = np.asarray(v_base, float)
        target = self.target_for(np.clip(np.r_[_rot(theta) @ v[:2], v[2]], -self.steady(self.cr), self.steady(self.cr)))
        return np.clip(np.r_[_rot(theta).T @ target[:2], target[2]] / self.cr, -1.0, 1.0)


# ---------------------------------------------------------------------- the filter


@dataclass
class Barrier:
    """Every (robot sphere, object) pair at one configuration."""

    h: np.ndarray  # [S, O] smooth barrier h = psi - r_i (eq. 6)
    exact: np.ndarray  # [S, O] min_m g_m - r_i: the exact distance to the union of the balls, minus the radius
    grad: np.ndarray  # [S, O, 13] grad_q h
    closest: np.ndarray  # [S] index (into the stacked points) of each sphere's closest ball, by g
    centres: np.ndarray  # [S, 3]
    radii: np.ndarray  # [S]


@dataclass
class FilterReceipt:
    tick: int
    outcome: str  # "passthrough" | "box" (u_nom clipped to U) | "filtered" (a row acted) | "relaxed" | "hold" | "passive"
    qp_status: str  # DAQP's exit of the QP (and of the relaxed QP), or "none"
    n_pairs: int  # rows of eq. (9): robot spheres x admitted objects
    n_rows_qp: int  # the rows some u in U could violate (given to the solver)
    h_min: float | None
    h_min_pair: list | None  # [sphere, link, object]
    exact_min: float | None
    solve_ms: float
    tick_ms: float
    u_nominal: list[float]
    u_applied: list[float]
    correction_norm: float
    known_velocity: list[float]
    slack_rows: list[dict] = field(default_factory=list)  # relaxed ticks: every row with slack > 0
    max_slack: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SphereFilter:
    """Rows of eq. (8) for every robot sphere against every admitted object, and the QP of eq. (9)."""

    def __init__(self, robot, policy: FilterPolicy = FilterPolicy(), device: str = "cuda"):
        self.robot, self.policy, self.device = robot, policy, device
        self.arm_limit = np.asarray(policy.arm_velocity_limits, float)
        self.set_objects({})

    def set_objects(self, balls: dict) -> None:
        """The admitted objects of a segment and their balls (uncertainty.Balls)."""
        self.object_ids = list(balls)
        t = dict(device=self.device, dtype=torch.float64)
        pts = [balls[o].points for o in self.object_ids]
        self.points_np = np.vstack(pts) if pts else np.zeros((0, 3))
        self.points = torch.tensor(self.points_np, **t)
        self.rho = torch.tensor(np.concatenate([balls[o].rho for o in self.object_ids]) if pts else np.zeros(0), **t)
        self.point_object = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)]) if pts else np.zeros(0, dtype=np.int64)
        self.obj = torch.tensor(self.point_object, device=self.device, dtype=torch.int64)

    def barrier(self, q: np.ndarray) -> Barrier:
        """eq. (5)-(6) for all sphere-object pairs at q (one GPU pass)."""
        c_np, r_np, J_np = self.robot.jacobians(q)
        S, O = len(r_np), len(self.object_ids)
        if O == 0:
            return Barrier(np.zeros((S, 0)), np.zeros((S, 0)), np.zeros((S, 0, J_np.shape[2])), np.full(S, -1), c_np, r_np)
        t = dict(device=self.device, dtype=torch.float64)
        tau = self.policy.tau_m
        with torch.no_grad():
            c, r, J = torch.tensor(c_np, **t), torch.tensor(r_np, **t), torch.tensor(J_np, **t)
            D = c[:, None, :] - self.points[None]  # [S, N, 3]
            dist = torch.linalg.norm(D, dim=2)
            if not bool(torch.all(dist > 0)):
                raise FloatingPointError("a robot sphere centre lies on a map point: nu_m is undefined there (eq. 5)")
            g = dist - self.rho
            a = -g / tau
            idx = self.obj[None].expand(S, -1)
            amax = torch.full((S, O), -torch.inf, **t).scatter_reduce(1, idx, a, reduce="amax")
            e = torch.exp(a - amax.gather(1, idx))
            Z = torch.zeros((S, O), **t).scatter_add(1, idx, e)
            psi = -tau * (amax + torch.log(Z))
            pi = e / Z.gather(1, idx)
            w = torch.zeros((S, O, 3), **t).scatter_add(1, idx[..., None].expand(-1, -1, 3), pi[..., None] * D / dist[..., None])
            grad = torch.einsum("sox,sxq->soq", w, J)
            exact = torch.full((S, O), torch.inf, **t).scatter_reduce(1, idx, g, reduce="amin")
            out = Barrier((psi - r[:, None]).cpu().numpy(), (exact - r[:, None]).cpu().numpy(), grad.cpu().numpy(), torch.argmin(g, dim=1).cpu().numpy(), c_np, r_np)
        if not (np.all(np.isfinite(out.h)) and np.all(np.isfinite(out.grad))):
            raise FloatingPointError("non-finite barrier value or gradient")
        return out

    def rows(self, bar: Barrier, known_qdot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """G u <= b for every pair (eq. 8 written as -grad_c . u <= alpha h + grad_k . q_dot_k - eps |grad_c|^2), and
        each row's pair (sphere, object)."""
        p = self.policy
        S, O = bar.h.shape
        gc = bar.grad[..., COMMANDED].reshape(S * O, -1)
        gk = bar.grad[..., KNOWN].reshape(S * O, -1)
        b = p.alpha * bar.h.reshape(-1) + gk @ np.asarray(known_qdot, float) - p.eps * np.einsum("ij,ij->i", gc, gc)
        pairs = np.stack(np.unravel_index(np.arange(S * O), (S, O)), axis=1)
        return -gc, b, pairs

    def box(self, q: np.ndarray, base_bounds: tuple[np.ndarray, np.ndarray], q_lo: np.ndarray, q_hi: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """U: the base's reachable mean velocities (BaseModel.bounds); for each arm joint its velocity limit and the
        velocities that keep it within its position limits over the tick (a joint at a limit cannot move into it)."""
        qa = np.asarray(q, float)[ARM_Q]
        lo = np.maximum(-self.arm_limit, (q_lo - qa) / dt)
        hi = np.minimum(self.arm_limit, (q_hi - qa) / dt)
        lo = np.minimum(lo, hi)  # a joint already past a limit (MuJoCo's limits are soft): only motion back is allowed
        return np.r_[base_bounds[0], lo], np.r_[base_bounds[1], hi]

    def filter(self, *, tick: int, q: np.ndarray, u_nom: np.ndarray, known_qdot: np.ndarray, bounds: tuple[np.ndarray, np.ndarray],
               passive: bool = False) -> tuple[np.ndarray, FilterReceipt, Barrier]:
        """One tick: rows at q, then the QP over u in U = [bounds] (``box``)."""
        t0 = time.perf_counter()
        p = self.policy
        u_nom = np.asarray(u_nom, float)
        self.lb, self.ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        n = len(u_nom)
        bar = self.barrier(q)
        G, b, pairs = self.rows(bar, known_qdot)
        can_bind = np.maximum(G * self.lb, G * self.ub).sum(axis=1) > b  # the others hold for every u in U
        G, b, pairs = G[can_bind], b[can_bind], pairs[can_bind]
        status, solve_ms, slack_rows, max_slack = "none", 0.0, [], 0.0
        if passive:
            u, outcome = u_nom, "passive"
        else:
            t1 = time.perf_counter()
            u, flag = self._solve(np.eye(n), -u_nom, G, b, self.lb, self.ub)
            status = DAQP_EXIT.get(flag, f"exit_{flag}")
            if flag > 0 and np.all(G @ u - b <= 1e-6):
                if np.linalg.norm(u - u_nom) <= 1e-9:
                    outcome = "passthrough"
                else:  # "box": u is u_nom clipped to U, no row acted; "filtered": a row moved u
                    outcome = "box" if np.allclose(u, np.clip(u_nom, self.lb, self.ub), rtol=0.0, atol=1e-7) else "filtered"
            elif flag > 0:
                u, outcome, status = np.zeros(n), "hold", status + "/postcheck_failed"
            elif flag == -1:  # infeasible: the relaxed QP, one slack per row
                m = G.shape[0]
                A = np.hstack([G, -np.eye(m)])
                x, flag2 = self._solve(np.diag(np.r_[np.ones(n), np.full(m, 1e-6)]), np.r_[-u_nom, np.full(m, p.relaxation_penalty)], A, b,
                                       np.r_[self.lb, np.zeros(m)], np.r_[self.ub, np.full(m, 1e30)])
                status += "/relaxed_" + DAQP_EXIT.get(flag2, f"exit_{flag2}")
                if flag2 > 0 and np.all(A @ x - b <= 1e-6):
                    u, s, outcome = x[:n], x[n:], "relaxed"
                    for k in np.flatnonzero(s > 1e-9):
                        i, j = pairs[k]
                        slack_rows.append({"sphere": int(i), "link": self.robot.links[i], "object": self.object_ids[j], "h": float(bar.h[i, j]), "slack": float(s[k])})
                    max_slack = float(s.max())
                else:
                    u, outcome = np.zeros(n), "hold"
                    status += "/postcheck_failed" if flag2 > 0 else ""
            else:
                u, outcome = np.zeros(n), "hold"
            solve_ms = (time.perf_counter() - t1) * 1e3
        if bar.h.size:
            i, j = np.unravel_index(int(np.argmin(bar.h)), bar.h.shape)
            h_min, pair, exact_min = float(bar.h[i, j]), [int(i), self.robot.links[i], self.object_ids[j]], float(bar.exact.min())
        else:
            h_min, pair, exact_min = None, None, None
        rec = FilterReceipt(tick=tick, outcome=outcome, qp_status=status, n_pairs=int(bar.h.size), n_rows_qp=int(G.shape[0]), h_min=h_min, h_min_pair=pair,
                            exact_min=exact_min, solve_ms=solve_ms, tick_ms=(time.perf_counter() - t0) * 1e3, u_nominal=[float(v) for v in u_nom],
                            u_applied=[float(v) for v in u], correction_norm=float(np.linalg.norm(u - u_nom)), known_velocity=[float(v) for v in known_qdot],
                            slack_rows=slack_rows, max_slack=max_slack)
        return np.asarray(u, float), rec, bar

    @staticmethod
    def _solve(H, f, A, b, lb, ub) -> tuple[np.ndarray, int]:
        """DAQP with simple bounds lb <= x <= ub and rows A x <= b; returns x and DAQP's exit flag."""
        n = len(f)
        x, _, flag, _ = daqp.solve(np.ascontiguousarray(H, dtype=float), np.ascontiguousarray(f, dtype=float), np.ascontiguousarray(np.reshape(A, (-1, n)), dtype=float),
                                   np.ascontiguousarray(np.r_[ub, b], dtype=float), np.ascontiguousarray(np.r_[lb, np.full(len(b), -1e30)], dtype=float),
                                   np.zeros(n + len(b), dtype=np.intc))
        return np.asarray(x, float), int(flag)
