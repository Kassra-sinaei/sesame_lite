"""Clip of a map being built (EVAL_SPEC.md, section 9): the mesh of a prepared scene, shown in the order in which the
mapping pass first saw each vertex (Hydra's first-seen stamps), for the full pass and for the look-around.

    python -s scripts/eval_map_clip.py [--scene close_fridge_s3]

Default scene: the first Table II scene of test_scenes.json. Writes outputs/eval/videos/map_building/<scene>_<kind>.mp4
(1920x1080, 20 fps; colour: height) and <scene>_<kind>.npz (vertices, faces, first-seen time in seconds) for a renderer
of one's own.
"""

import argparse
import json

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sesame_lite.eval_scene import EVAL_DIR, SCENES_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=None)
ap.add_argument("--seconds", type=float, default=12.0, help="length of each clip")
args = ap.parse_args()
tests = json.loads((EVAL_DIR / "results" / "test_scenes.json").read_text())["test_scenes"]
scene = args.scene or next(t["scene"] for t in tests if t["in_table2"])
out = EVAL_DIR / "videos" / "map_building"
out.mkdir(parents=True, exist_ok=True)
for kind, title in (("full", "full pass"), ("look", "look-around")):
    mesh = json.loads((SCENES_DIR / scene / f"map_{kind}" / "dsg.json").read_text())["mesh"]
    P, t = np.array(mesh["points"]), np.array(mesh["first_seen_stamps"]) * 1e-9
    P, t = P[P[:, 2] < 2.2], t[P[:, 2] < 2.2]  # without the ceiling, which would hide the kitchen
    np.savez_compressed(out / f"{scene}_{kind}.npz", vertices=np.array(mesh["points"]), faces=np.array(mesh["faces"]), first_seen_s=np.array(mesh["first_seen_stamps"]) * 1e-9)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=55, azim=-60)
    ax.set_box_aspect(np.ptp(P, axis=0))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    lo, hi = P.min(axis=0), P.max(axis=0)
    with av.open(str(out / f"{scene}_{kind}.mp4"), "w") as video:
        stream = video.add_stream("h264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
        n = int(args.seconds * 20)
        for k in range(n + 20):  # the last second holds the finished map
            now = t.max() * min(1.0, (k + 1) / n)
            ax.cla()
            ax.set_axis_off()
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
            seen = t <= now
            ax.scatter(*P[seen].T, c=P[seen, 2], cmap="viridis", vmin=0.0, vmax=2.2, s=2.0, linewidths=0)
            fig.text(0.02, 0.95, f"{scene}  |  map: {title}  |  mapping time {now:5.1f} s  |  {int(seen.sum())} mesh vertices", fontsize=18, ha="left")
            fig.canvas.draw()
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(fig.canvas.buffer_rgba())[..., :3], format="rgb24")):
                video.mux(packet)
            fig.texts.clear()
        for packet in stream.encode():
            video.mux(packet)
    plt.close(fig)
    print(f"wrote {out / f'{scene}_{kind}.mp4'}")
