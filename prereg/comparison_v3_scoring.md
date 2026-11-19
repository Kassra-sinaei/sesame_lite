# Pre-registration: scoring of the comparison table v3 (Nominal, AEGIS, OmniGuide, Proposed), 2026-11-19

Written before any manipulation-only score was computed. SHA-256 of this file and of the scoring script are in
`outputs/eval/results/v3/shared/prereg_scoring.sha256`.

**Decided by Kasra (2026-11-19):** the 14 scenes with a designated obstacle (`V3_SCENES` in the script); rows Nominal,
AEGIS, OmniGuide, Proposed on the full-pass map, Xiaomi-Robotics-1, the frozen_v2 condition; the navigation segment run
by Proposed in every row and not scored; only the manipulation segment scored, with all contacts and every metric of
Table II.

**Scoring:** `scripts/score_segment.py --segment manipulation`, the same code for every row. Its docstring defines
every metric. "All contacts" is read as the counted contacts of Table II (robot and any object on a pair the plan does
not allow), as in the first campaign; the gripper on the task objects is not a contact. Contacts are also split by robot
part (base, arm, hand, fingers). Rates with 95% Wilson intervals; Proposed against each row by the exact McNemar test
on the 14 matched scenes, for the five binary columns. No threshold or pass criterion: the table is reported as it is.

**Check made before the freeze:** with `--segment all` the script reproduces every stored full-map run of Table II on
the 36 scenes (180 runs, no difference).

**Seen before the freeze:** Table II (whole episodes). While checking that the logs label every tick with its segment,
the per-segment contact tick counts of the stored Nominal, End-effector filter and Proposed runs on the 14 scenes were
printed once. Nothing was changed after that.

**Open, not decided here:** a run that never reaches the manipulation segment stops the script (none of the stored
runs does).
