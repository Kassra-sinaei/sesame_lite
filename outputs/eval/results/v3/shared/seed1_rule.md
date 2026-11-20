# Second policy seed for 10 of the 14 comparison scenes (Kasra, 2026-11-20; written before any of these runs)

- Purpose: Table 3 reports rates over 24 runs per method instead of 14 (Kasra). The 24 runs are 14 scenes, 10 of them
  run a second time with a new policy seed; they are not 24 independent scenes.
- The 10 scenes (rule fixed before these runs, independent of results: the first scenes of each task in the scene list
  of the decision of 2026-11-19, 4/3/3 by task): sink_to_counter s7 s11 s12 s18; counter_to_cabinet s3 s10 s15;
  counter_to_stove s0 s7 s9.
- New policy seed: scene seed + 1000000 (`scripts/eval_run.py --policy-seed`). Only the seeds of the XR1 policy's
  queries change (seed x 100003 + query count); the scene, its obstacle, the arm noise a2 (seeded with the scene seed),
  the navigation (pure pursuit with the Proposed filter, no policy) and the state at the start of manipulation are the
  same as in the first run of the scene.
- Rows and code, otherwise exactly as the first 14 runs:
  - Nominal (`nominal_nav`), AEGIS (`aegis`) and Proposed (`proposed`): branch baseline-aegis at 0530d07 (= ff24dbc plus
    the seed option), standard XR1 server (scripts/xr1_server.py) on port 10086;
  - OmniGuide (`omniguide`, guidance_factor 2 as chosen by the tuning rule): branch baseline-omniguide at e9120cd (=
    eefcfaf plus the seed option), src/omniguide/server.py on port 10087.
  - Condition of Table II (frozen_v2): --map full --vla xr1 --disturbance a2 --policy '{"alpha": 4.0, "d_bar": 0.03}'
    --horizon 1800 --scenes-root outputs/eval/scenes_v2; no video.
- One run per scene and row (40 runs); a run that ends without a result is repeated once; nothing is dropped for its
  result.
- Scoring: scripts/score_segment.py (df53606, unchanged), manipulation segment, 24 runs per row (the 14 first runs and
  the 10 new ones), every Table II metric and the contacts by robot part. McNemar is not reported (Kasra).
- The first runs of sink_to_counter_s7 (recorded with video in the Nominal, AEGIS and OmniGuide rows) are kept as they
  are; Kasra's decision on them is open.
