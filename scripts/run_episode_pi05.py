"""scripts/run_episode.py with lerobot/pi05_robocasa as the VLA nominal instead of SmolVLA (server; REKI is not here).

Identical CLI and behaviour; the only change is the policy class that ``sesame_lite.env.run_episode`` instantiates
for VLA segments (``--nominal smolvla`` / ``auto`` now means "the VLA", i.e. pi0.5). The original files are untouched.

    MUJOCO_GL=egl python -s scripts/run_episode_pi05.py --plan plans/open_drawer.json --seed 0 --video
"""

import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sesame_lite.env as sesame_env  # noqa: E402
from pi05_nominal import Pi05Policy  # noqa: E402

sesame_env.SmolVLAPolicy = Pi05Policy  # run_episode() looks the class up in its module at call time

exec(compile(Path(__file__).with_name("run_episode.py").read_text(), "run_episode.py", "exec"))
