# Table 3 with 24 runs per method (2026-11-20)

**What the 24 runs are.** The 14 comparison scenes, run once each (the first runs, as in table_v3_four_rows.md), plus 10
of them run a second time with a new policy seed. The rule for the second runs is in seed1_rule.md; its SHA-256 is in
seed1_rule.sha256, and both were written before these runs.

- They are 24 runs of 14 scenes, not 24 independent scenes.
- The header line of table_v3_24runs.md ("24 scenes; one run per scene and row") is written by score_segment.py and
  counts the second runs as scenes. The script was used unchanged (SHA-256 2eb35b41..., as frozen).

**Statistics.** The Wilson intervals and McNemar tests in table_v3_24runs.md treat the 24 runs as independent. McNemar is
not reported (Kasra).

**Second runs.**
- Seed: policy seed = scene seed + 1000000; everything else as in the first runs.
- Start state: in all 40 runs, the state at the start of manipulation equals that of the scene's first Proposed run
  (largest difference 0, measured).
- Code: Nominal, AEGIS and Proposed from branch baseline-aegis at 0530d07. OmniGuide (guidance_factor 2) from branch
  baseline-omniguide at e9120cd. No video.
- Where they are: <row>/runs/seed1/<scene>/; for Proposed, shared/proposed/seed1/<scene>/. Proposed's first runs are the
  stored Table II runs.

**Infrastructure event** (no result was produced, so none was seen or dropped).
- What happened: the first attempt at the 10 OmniGuide runs (two tries each) crashed in the policy server. It had been
  started without CUBLAS_WORKSPACE_CONFIG=:4096:8, the deterministic cuBLAS setting the guided sampler needs.
- What was done: the server was restarted with the setting and the 10 runs were made. All of them completed.

**Still open (Kasra).** The first sink_to_counter_s7 runs of Nominal, AEGIS and OmniGuide were recorded with video.
