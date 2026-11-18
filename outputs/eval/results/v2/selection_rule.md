# Settings for the campaign with Xiaomi-Robotics-1 (written 2026-11-17 ~06:15 UTC, before the alpha = 4 ablation runs and the arm-setting checks were read)

Decisions by Kasra (2026-11-17, before going offline):
- The campaign runs only if Proposed beats Proposed without ISSf and Fixed margin on the tuning scenes. If it does not
  without a disturbance, an arm tracking setting is chosen under which Nominal keeps almost the same success (a bit
  lower is acceptable) and Proposed outperforms both ablations. One or two more tuning rounds are allowed.
- The arm setting: whichever helps Proposed.
- Scope: the 24 scenes of the first campaign, all 7 rows; the tuning scenes are reported separately.
- Nothing is committed until Kasra has checked the results.

Rule (the tuning scenes are the 10 screening scenes listed in tuning_scenes.txt; obstacles from Xiaomi-Robotics-1's
Nominal rollouts, 2.5 cm maps):
1. Measure: safe success (task done and no counted contact) per row, on the 10 tuning scenes, alpha and the arm setting
   the same for every row; Fixed margin with rho_fixed = 3.9 cm (results/rho_fixed.json, 2.5 cm design maps).
2. Arm settings considered: none, kv 40, kv 20, arm noise a1, a2. A setting is admissible if Nominal finishes at least
   7 of the 10 tuning scenes with it (8 of 10 without).
3. Gate: Proposed's safe success is strictly larger than that of Proposed without ISSf and of Fixed margin.
4. Order: first alpha = 4 without a disturbance. If the gate fails there, each admissible arm setting at alpha = 4.
   If no setting passes, at most two more rounds (alpha = 6; then epsilon x 2), each over the admissible arm settings.
5. Choice: among the (alpha, arm setting) pairs that pass the gate, the one with the largest lead of Proposed over the
   better of the two ablations; ties: no disturbance first, then the smaller tracking error, then the smaller alpha.
6. If nothing passes after the allowed rounds: no campaign; the tuning results are reported.
7. The chosen values are frozen (frozen_v2.json with SHA-256) before any campaign run.

This rule chooses the condition and the gains on the tuning scenes so that Proposed's advantage can show. The campaign
on the 24 held-out scenes then runs once per row and is reported whatever it shows; the tuning scenes are reported
separately and flagged.
