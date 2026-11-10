"""The safety filter's parameters and the base command model (spec v4, sections 6 and 10).

The QP of PDF eq. (9) acts on u, the joint velocities of the base (forward slide, side slide, yaw) and of the seven arm
joints. This module holds the filter constants (``FilterPolicy``, the one place for them) and the map between the
base joint velocities and RoboCasa's base command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTROL_DT_S = 0.05  # RoboCasa runs at 20 Hz
ACTION_DIM = 12
# lerobot flat action layout for RoboCasa (see lerobot.envs.robocasa.convert_action)
BASE_SLICE = slice(0, 3)  # forward, side, yaw velocity commands in [-1, 1]
TORSO_INDEX = 3
MODE_INDEX = 4  # < 0.5 arm mode, >= 0.5 base mode
EE_POS_SLICE = slice(5, 8)  # OSC_POSE delta in the base frame, output_max 0.05 m per step
EE_ROT_SLICE = slice(8, 11)
GRIPPER_INDEX = 11


@dataclass(frozen=True)
class FilterPolicy:
    """Filter constants. The first block is the PDF's (Numbers, Theorem 2); the rest describes the robot."""

    alpha: float = 1.0  # class-K gain of eq. (7)-(9)
    d_bar: float = 0.05  # bound on the tracking disturbance |d|, q_dot = u + d
    tau_m: float = 0.002  # temperature of the log-sum-exp (eq. 5)
    delta: float = 0.01  # map risk, split over the admitted objects, then over 2 |P_j| (eq. 4)
    relaxation_penalty: float = 1e3  # price per unit slack of a relaxed row (infeasible QP, section 6.3)
    goal_standoff_m: float = 0.065  # base goal clearance from the balls (the value used before; planning only)
    # RoboCasa's base is driven by MuJoCo velocity actuators (ctrlrange x kv) against a joint frictionloss: steady
    # speed = command*ctrlrange - frictionloss/kv, zero below the deadband (measured in the earlier passes).
    base_ctrlrange: tuple[float, float, float] = (1.0, 1.0, 1.5)  # m/s, m/s, rad/s per unit action
    base_deadband: tuple[float, float, float] = (0.25, 0.25, 250.0 / 1500.0)  # joint frictionloss / kv
    # the QP may only ask for base velocities reachable within one tick from the last command (as before)
    base_accel_limits: tuple[float, float, float] = (3.0, 3.0, 6.0)  # m/s^2, m/s^2, rad/s^2
    arm_velocity_limits: tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)  # rad/s, Franka datasheet

    @property
    def eps(self) -> float:
        """eps = d_bar / 2 makes the rest offset smallest (PDF, Theorem 2)."""
        return self.d_bar / 2.0

    @property
    def m_d(self) -> float:
        """m_d = d_bar^2 / (4 eps alpha), added to every ball radius (Theorem 2): 2.5 cm."""
        return self.d_bar**2 / (4.0 * self.eps * self.alpha)


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def command_to_joint_velocity(cmd_unit: np.ndarray, policy: FilterPolicy, theta: float) -> np.ndarray:
    """Steady base joint velocities (forward slide, side slide, yaw) for a unit command in [-1, 1].

    robosuite's legacy base controller turns the command (base frame) by theta = yaw - yaw at episode start onto the
    two slide joints, whose axes are fixed in the start frame; each joint then loses ``base_deadband`` to friction.
    """
    cmd = np.clip(np.asarray(cmd_unit, dtype=float), -1.0, 1.0) * np.asarray(policy.base_ctrlrange)
    target = np.r_[_rot(theta) @ cmd[:2], cmd[2]]
    return np.sign(target) * np.maximum(np.abs(target) - np.asarray(policy.base_deadband), 0.0)


def joint_velocity_to_command(u_joint: np.ndarray, policy: FilterPolicy, theta: float) -> np.ndarray:
    """Inverse of ``command_to_joint_velocity`` (deadband added back per joint); zero stays zero."""
    u = np.asarray(u_joint, dtype=float)
    target = np.where(np.abs(u) > 1e-4, u + np.sign(u) * np.asarray(policy.base_deadband), 0.0)
    cmd = np.r_[_rot(theta).T @ target[:2], target[2]]
    return np.clip(cmd / np.asarray(policy.base_ctrlrange), -1.0, 1.0)


def base_velocity_to_command(v_base: np.ndarray, policy: FilterPolicy, theta: float) -> np.ndarray:
    """Command for a base-frame velocity (vx, vy, wz) (the navigation follower's output)."""
    v = np.asarray(v_base, dtype=float)
    return joint_velocity_to_command(np.r_[_rot(theta) @ v[:2], v[2]], policy, theta)
