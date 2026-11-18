"""Xiaomi-Robotics-1 as the VLA nominal: the simulation side of scripts/xr1_server.py.

Drop-in for ``Pi05Policy``: ``reset()`` and ``__call__(obs, language) -> 12-D env-order action``. Like the official
RoboCasa365 evaluation client (third_party/Xiaomi-Robotics-1/eval_robocasa365/entry.py): the raw RoboCasa observation
(the three cameras and the 14-D end-effector-first state) is queued every tick, a query sends 4 frames 2 ticks apart,
and the 16 returned actions are executed one per tick before the next query. RoboCasa's native action order is mapped
to the env order the filter uses (as for pi0.5: scripts/pi05_nominal.py). The seed of each query comes from the scene
seed and the query count, so a run is repeatable.
"""

from __future__ import annotations

import collections
import pickle
import socket
import struct

import numpy as np

from pi05_nominal import ACTION_TO_ENV

CAMERA_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
OBS_HISTORY, OBS_INTERVAL, REPLAN_STEPS = 4, 2, 16


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:  # entry.py, unchanged
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(o: dict) -> np.ndarray:  # entry.py, unchanged
    return np.concatenate([np.asarray(o["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
                           quat_xyzw_to_axis_angle(o["state.end_effector_rotation_relative"]),
                           np.asarray(o["state.gripper_qpos"], dtype=np.float32).reshape(-1),
                           np.asarray(o["state.base_position"], dtype=np.float32).reshape(-1),
                           quat_xyzw_to_axis_angle(o["state.base_rotation"])]).astype(np.float32)


def sample_history(items: list, length: int, interval: int) -> np.ndarray:  # entry.py, unchanged
    idx = [max(0, len(items) - 1 - (length - 1 - i) * interval) for i in range(length)]
    return np.ascontiguousarray(np.stack([items[i] for i in idx], axis=0))


class XR1Policy:
    def __init__(self, env, seed: int, port: int = 10086):
        self.env, self.seed = env, int(seed)  # env: the SesameRoboCasaEnv (its last raw RoboCasa observation)
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.queries = 0
        self.reset()

    def reset(self):
        n = (OBS_HISTORY - 1) * OBS_INTERVAL + 1
        self.images = {k: collections.deque(maxlen=n) for k in CAMERA_KEYS}
        self.states = collections.deque(maxlen=n)
        self.plan = collections.deque()

    def __call__(self, obs: dict, language: str) -> np.ndarray:
        raw = self.env.last_raw_obs
        for k in CAMERA_KEYS:
            self.images[k].append(np.ascontiguousarray(raw[k], dtype=np.uint8))
        self.states.append(observation_to_state(raw))
        if not self.plan:
            req = {"state_history": sample_history(list(self.states), OBS_HISTORY, OBS_INTERVAL),
                   "images": {k: sample_history(list(q), OBS_HISTORY, OBS_INTERVAL) for k, q in self.images.items()},
                   "instruction": language, "seed": self.seed * 100003 + self.queries}
            self.queries += 1
            body = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
            self.sock.sendall(struct.pack(">I", len(body)) + body)
            n = struct.unpack(">I", self._recv(4))[0]
            self.plan.extend(pickle.loads(self._recv(n))[:REPLAN_STEPS])
        return np.asarray(self.plan.popleft(), dtype=np.float32)[ACTION_TO_ENV]

    def _recv(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("the XR1 server closed the connection")
            data += packet
        return data
