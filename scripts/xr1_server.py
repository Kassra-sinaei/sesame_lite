"""Xiaomi-Robotics-1 (RoboCasa365 checkpoint) as the VLA nominal: an inference server in its own environment.

    .venv-xr1/bin/python scripts/xr1_server.py --model models/Xiaomi-Robotics-1-RoboCasa365 [--port 10086]

The model needs transformers 4.57.1 and flash-attn (its reference environment); our main environment has newer
versions, so the model and its processor run here, and the simulation side (scripts/xr1_nominal.py) sends raw inputs.
Inputs are built exactly as the official evaluation client (third_party/Xiaomi-Robotics-1/eval_robocasa365/entry.py,
EvalClient): three cameras as 4-frame videos, centre crop 0.95, the 14-D state padded to 60, the prompt text, robot
type "robocasa365"; the forward pass is the official server's (deploy/server.py); actions are decoded with the
processor's statistics. Each request carries a seed: the flow-matching noise (torch.randn_like) is drawn after
torch.manual_seed(seed), so a run is repeatable.

Protocol: length-prefixed pickle over a local socket (one thread per connection, the model behind a lock).
Request {"state_history": (4, 14) float32, "images": {camera key: (4, H, W, 3) uint8}, "instruction": str, "seed": int};
reply (16, 12) float32 actions in RoboCasa's native order.
"""

import argparse
import pickle
import socket
import struct
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROBOT_TYPE, STATE_DIM, ACTION_DIM, CROP = "robocasa365", 60, 12, 0.95
CAMERA_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:  # entry.py, unchanged
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)


def messages(images: dict, instruction: str) -> list:  # entry.py EvalClient._build_messages, unchanged
    videos = {key: [center_crop(frame, CROP) for frame in images[key]] for key in CAMERA_KEYS}
    return [{"role": "user", "content": [
                {"type": "text", "text": "Left camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}]


class Server:
    def __init__(self, model_path: str):
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().eval()
        self.lock = threading.Lock()

    def infer(self, req: dict) -> np.ndarray:
        hist = np.asarray(req["state_history"], dtype=np.float32)
        state = np.zeros((1, hist.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, :hist.shape[-1]] = hist
        inputs = self.processor.apply_chat_template(messages(req["images"], req["instruction"]), tokenize=True, return_dict=True, return_tensors="pt",
                                                    do_resize=False, state=state, robot_type=ROBOT_TYPE)
        data = dict(inputs)
        data["task_id"] = ROBOT_TYPE
        data = {k: (v.to(device=self.model.device, dtype=self.model.dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else
                    v.to(self.model.device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        with self.lock, torch.no_grad():
            torch.manual_seed(int(req["seed"]))
            out = self.model(**data)
        actions = self.processor.decode_action(out.actions.cpu(), robot_type=ROBOT_TYPE)
        return np.asarray(actions[0, :, :ACTION_DIM].float().numpy(), dtype=np.float32)

    def handle(self, conn):
        with conn:
            while True:
                head = recv_all(conn, 4)
                if head is None:
                    return
                body = recv_all(conn, struct.unpack(">I", head)[0])
                if body is None:
                    return
                t0 = time.time()
                reply = pickle.dumps(self.infer(pickle.loads(body)))
                conn.sendall(struct.pack(">I", len(reply)) + reply)
                print(f"request {time.time() - t0:.3f} s", flush=True)


def recv_all(conn, n: int):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=10086)
    args = ap.parse_args()
    server = Server(args.model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", args.port))
        s.listen(8)
        print(f"XR1 server ready on 127.0.0.1:{args.port}", flush=True)
        while True:
            conn, _ = s.accept()
            threading.Thread(target=server.handle, args=(conn,), daemon=True).start()
