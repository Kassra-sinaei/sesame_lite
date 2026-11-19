# Pre-registration: the Nominal and AEGIS rows of the comparison table v3, 2026-11-19

Written before the 14-scene runs of these rows. The SHA-256 of this file and of the code that runs them are in
`outputs/eval/results/v3/aegis/prereg_aegis.sha256`. Scoring: `prereg/comparison_v3_scoring.md` (frozen earlier).

**Common to both rows** (Kasra, 2026-11-19): the 14 scenes of `scripts/score_segment.py`, the frozen_v2 condition
(`--vla xr1 --disturbance a2 --policy '{"alpha": 4.0, "d_bar": 0.03}' --horizon 1800`, scenes_v2, full-pass map); the
navigation segment run by the Proposed filter (`filter_segments=(0,)`), the manipulation segment without it (the
Proposed filter passive: its barrier is evaluated and logged, it does not act). Methods `nominal_nav` and `aegis` of
`scripts/eval_run.py`. Proposed is the stored run of the 36-scene campaign.

**AEGIS** (arXiv 2512.11891, vlsa-aegis at ce5683c, port in `src/aegis/`):
- obstacle: the scene's designated obstacle, its map node's points (map_full); fitted once at the first tick of the
  manipulation segment, then fixed;
- clean-up: their DBSCAN (eps 0.1 mm, 50 points, largest cluster if any); their box crop and their drop of the 20%
  farthest points are left out (Kasra: "skip the box crop ... no need to throw away 20% of the points");
- shape: minimum-volume ellipsoid of the convex hull of the points (Khachiyan, optimality gap 1e-4);
- gripper: semi-axes (0.06, 0.12, 0.11) m about the hand body's axes, centred 0.08 m along its -z axis from the
  end-effector site, in every task;
- barrier and QP as their code: compute_h_coeffs_3d, u = [u_v, u_w, u_z], constraint 0.2 a_v u_v + 0.2 a_w u_w +
  a_z u_z + 10 h >= 0, weights 1/25 (u_v, u_w) and 1 (u_z), reference [5 R^T v, 5 w, 10 mu], command 0.2 R u_v and
  0.2 u_w, z updated with dt 0.05; solved with daqp; no other bound;
- the command is the policy's end-effector delta turned from the arm's base frame into the world frame (their frame) and
  back; units unchanged (the same OSC_POSE settings in both); it goes to our arm drive; the base is not filtered;
- a QP without a solution passes the nominal command for that tick; such ticks are counted;
- no uncertainty radius, no ISSf term, no pruning, no base rows.

**Runs:** one per scene and row, all reported. A run that ends without a result (crash, out of memory) is repeated once
(`scripts/eval_campaign.py`); a second failure is reported. No run is dropped for its result.

**Checks before the freeze** (no score computed):
- offline: the ellipsoid fit of all 14 obstacles (every point inside, DBSCAN keeps every point); daqp against the
  closed-form solution of the one-row QP (2000 random cases, difference 6e-16); the Khachiyan stopping gap set here;
- sink_to_counter_s7, both rows, with video: navigation and the state at the start of manipulation identical to the
  stored Proposed run (difference 0); AEGIS fitted once, acted on 235 of 1293 ticks, every QP solved; the command
  turned into the world frame points along the measured gripper velocity (median cosine 0.97); on passthrough ticks
  the command is unchanged. Seen: both runs failed the task with contacts (Nominal 946 contact ticks, AEGIS 247 over
  the whole run); AEGIS's own h fell below zero on 42 ticks. The measured gripper speed is 0.27 m/s per unit command
  where their model assumes 1.0 m/s (the OSC's lag, the same controller settings as theirs); their gains are kept.
- Nothing in the method or the scoring was changed because of these runs; both runs are kept as the campaign's runs
  of that scene.
