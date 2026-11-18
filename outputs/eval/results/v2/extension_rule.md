# Extension of Table II from 24 to 36 scenes (Kasra, 2026-11-17, written before any scene of the extension was prepared)

- Scenes: two more per Table II instruction (12), by the selection rule of the first 24 continued
  (scripts/eval_select_more.py; its RULE string is copied into test_scenes_extension.json). The choice depends only on
  which kitchens are free and which scenes can be built, not on any result. New seeds are prepared with
  scripts/eval_prepare.py --voxel 0.05, the map voxel size the first 24 were prepared with.
- Everything else as frozen in frozen_v2.json (SHA-256 d643d969...): Xiaomi-Robotics-1, alpha 4, d_bar 0.03, arm noise
  a2, 2.5 cm maps, 1800 ticks, the obstacle rule of scripts/eval_obstacle.py (from Nominal's rollout without
  disturbance; none in the door scenes), Qwen labels, all 7 rows, deterministic runs.
- The first 24 scenes and their runs are kept unchanged; nothing is rerun.
- Reporting: Table II on the 24 scenes as before, the 12 new scenes on their own, and all 36 together; rates with 95%
  Wilson intervals, exact McNemar tests of Proposed against each row; the primary comparison stays the collision
  avoidance of Proposed against Nominal. Every run is reported.
