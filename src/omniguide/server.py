"""Xiaomi-Robotics-1 with OmniGuide's collision guidance: scripts/xr1_server.py with the sampler of omniguide/sampler.py.

    cd <repo> && .venv-xr1/bin/python src/omniguide/server.py --model models/Xiaomi-Robotics-1-RoboCasa365 [--port 10087]

Inputs and the language-model pass are the standard server's (scripts/xr1_server.py, Server); only the action
expert's loop is run here, outside the model's no-gradient forward (sampler.py). Two requests, length-prefixed pickle:
  {"field": {"sdf", "origin", "voxel_m"}}      the scene's distance field (omniguide/field.py), once per connection
  the standard request + "guide": None, or {"guidance_factor", "ee_pos_rel", "ee_quat_rel", "base_pos", "base_quat"}
Reply to a query: {"actions": (16, 12) float32 in RoboCasa's native order, "info": the sampler's record}.
"""

import argparse
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import sampler  # noqa: E402
from sampler import DistanceField, sample  # noqa: E402
from xr1_server import ACTION_DIM, ROBOT_TYPE, STATE_DIM, Server, messages, recv_all  # noqa: E402


CONSTANTS = ("NUM_STEPS", "PICK_STEPS", "N_RESTARTS", "SDF_DIST_M", "TOP_OFFSET_M", "COLLISION_WEIGHT", "LOG_FLOOR_M", "GAMMA_POS", "GAMMA_ROT")


class GuidedServer(Server):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model.requires_grad_(False)  # gradients go to the noisy actions only
        cfg = self.processor.action_config[ROBOT_TYPE]
        self.mean, self.std = cfg["mean"].to(self.model.device), cfg["std"].to(self.model.device)

    def decode(self, actions: torch.Tensor) -> torch.Tensor:  # processing_mibot.py:132-150 (x * std + mean), on the GPU
        return actions.float() * self.std + self.mean

    def inputs(self, req: dict) -> dict:  # Server.infer (xr1_server.py:62-71), unchanged
        hist = np.asarray(req["state_history"], dtype=np.float32)
        state = np.zeros((1, hist.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, :hist.shape[-1]] = hist
        inputs = self.processor.apply_chat_template(messages(req["images"], req["instruction"]), tokenize=True, return_dict=True, return_tensors="pt",
                                                    do_resize=False, state=state, robot_type=ROBOT_TYPE)
        data = dict(inputs)
        data["task_id"] = ROBOT_TYPE
        return {k: (v.to(device=self.model.device, dtype=self.model.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else
                    v.to(self.model.device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}

    def guided(self, req: dict, field) -> dict:
        data = self.inputs(req)
        with self.lock:
            x, info = sample(self.model, data, int(req["seed"]), self.decode, req.get("guide"), field)
        actions = self.processor.decode_action(x.cpu(), robot_type=ROBOT_TYPE)
        return {"actions": np.asarray(actions[0, :, :ACTION_DIM].float().numpy(), dtype=np.float32), "info": info}

    def handle(self, conn):
        field = None  # one scene per connection
        with conn:
            while True:
                head = recv_all(conn, 4)
                if head is None:
                    return
                body = recv_all(conn, struct.unpack(">I", head)[0])
                if body is None:
                    return
                t0 = time.time()
                req = pickle.loads(body)
                if "field" in req:
                    f = req["field"]
                    with self.lock:
                        field = DistanceField(f["sdf"], f["origin"], f["voxel_m"], device=self.model.device)
                    reply = {"dims": list(field.d.shape), "constants": {k: getattr(sampler, k) for k in CONSTANTS}}
                else:
                    reply = self.guided(req, field)
                out = pickle.dumps(reply)
                conn.sendall(struct.pack(">I", len(out)) + out)
                print(f"request {time.time() - t0:.3f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=10087)
    args = ap.parse_args()
    server = GuidedServer(args.model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", args.port))
        s.listen(8)
        print(f"OmniGuide XR1 server ready on 127.0.0.1:{args.port}", flush=True)
        while True:
            conn, _ = s.accept()
            threading.Thread(target=server.handle, args=(conn,), daemon=True).start()
