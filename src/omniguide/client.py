"""The OmniGuide row on the simulation side: the segment switch and the client of omniguide/server.py.

Kasra, 2026-11-19: the navigation segment runs with our Proposed filter (identical in every row, not scored); in the
manipulation segment OmniGuide acts alone: the policy's commands, base included, go to the robot unfiltered (the
barrier rows are still evaluated, as in the Nominal row), and the policy is Xiaomi-Robotics-1 with the collision
guidance (sampler.py). Guidance is on in the manipulation segment only (the policy is only queried there).
"""

from __future__ import annotations

import json
import pickle
import struct
import time
from pathlib import Path

import numpy as np

from omniguide.field import build_field
from pi05_nominal import ACTION_TO_ENV
from sesame_lite.env import _json_default
from sesame_lite.eval_scene import EvalFilterEnv
from sesame_lite.planner import VLA
from xr1_nominal import CAMERA_KEYS, OBS_HISTORY, OBS_INTERVAL, REPLAN_STEPS, XR1Policy, observation_to_state, sample_history

PORT = 10087  # Agent B's server (Agent A: 10086)


class OmniGuideEnv(EvalFilterEnv):
    """The Proposed filter's environment, with the filter switched off from the manipulation segment on."""

    def begin_segment(self, index: int) -> None:
        super().begin_segment(index)
        self.filter_on = self.segments[index].kind != VLA


class GuidedXR1Policy(XR1Policy):
    """XR1Policy (scripts/xr1_nominal.py) on the guided server: the scene's distance field is sent once, and every
    query carries the guidance strength and the pose at the query (RoboCasa's raw observation)."""

    def __init__(self, base, env: OmniGuideEnv, seed: int, guidance_factor: float, port: int = PORT):
        super().__init__(base, seed, port=port)
        self.filter_env, self.guidance_factor, self.log = env, float(guidance_factor), []
        field = build_field(env)
        self.field_meta = field.pop("meta")
        t0 = time.time()
        reply = self._ask({"field": field})
        self.field_meta.update(server_dims=reply["dims"], send_s=round(time.time() - t0, 2), sampler_constants=reply["constants"])

    def _ask(self, req: dict):
        body = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(body)) + body)
        n = struct.unpack(">I", self._recv(4))[0]
        return pickle.loads(self._recv(n))

    def _guide(self, raw: dict) -> dict | None:
        if self.filter_env.segment.kind != VLA:
            return None
        keys = {"ee_pos_rel": "state.end_effector_position_relative", "ee_quat_rel": "state.end_effector_rotation_relative",
                "base_pos": "state.base_position", "base_quat": "state.base_rotation"}
        return {"guidance_factor": self.guidance_factor, **{k: np.asarray(raw[v], dtype=np.float64) for k, v in keys.items()}}

    def __call__(self, obs: dict, language: str) -> np.ndarray:  # XR1Policy.__call__ (xr1_nominal.py:68-82) plus "guide"
        raw = self.env.last_raw_obs
        for k in CAMERA_KEYS:
            self.images[k].append(np.ascontiguousarray(raw[k], dtype=np.uint8))
        self.states.append(observation_to_state(raw))
        if not self.plan:
            req = {"state_history": sample_history(list(self.states), OBS_HISTORY, OBS_INTERVAL),
                   "images": {k: sample_history(list(q), OBS_HISTORY, OBS_INTERVAL) for k, q in self.images.items()},
                   "instruction": language, "seed": self.seed * 100003 + self.queries, "guide": self._guide(raw)}
            self.queries += 1
            t0 = time.time()
            reply = self._ask(req)
            self.plan.extend(reply["actions"][:REPLAN_STEPS])
            self.log.append({"tick": self.filter_env.tick, "seed": req["seed"], "round_trip_s": round(time.time() - t0, 4), **reply["info"],
                             "actions": np.round(reply["actions"], 6).tolist()})
        return np.asarray(self.plan.popleft(), dtype=np.float32)[ACTION_TO_ENV]

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({"guidance_factor": self.guidance_factor, "port": self.sock.getpeername()[1], "field": self.field_meta, "queries": self.log},
                                   default=_json_default))
