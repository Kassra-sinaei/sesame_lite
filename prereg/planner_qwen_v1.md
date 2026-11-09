# Pre-registration: Qwen3-14B tool-calling planner on the Hydra-format 3DSG (v1) — DRAFT, not frozen

Status: **draft for Kasra's review**. Once he approves, its SHA-256 is recorded in `prereg/SHA256SUMS` before any test
scene (seeds 0–9) is planned. Until then only the design seed 100 has been used.

## Question
Given only the task instruction and five tools over the oracle scene graph in Hydra's format, how often does the
LangGraph planner (Qwen3-14B) commit a plan that names the same target and the same segments as the hand-written
reference plan?

## Frozen inputs (SHA-256 at freeze time)

| file | role |
|---|---|
| `src/sesame_lite/agents/planner_agent.py` | system prompt, tool definitions, model settings, turn budget (16) |
| `src/sesame_lite/scene_graph.py` | Hydra-format graph (one AABB per fixture/object, places, room, building, agent) |
| `config/robocasa_label_space.yaml` | label space (`b9545c35…`) |
| `scripts/plan_mission.py` | scene construction (same as `run_episode`) and output |
| `src/sesame_lite/planner.py` | how reference roles are resolved |
| `plans/*.json` | the five missions and their reference plans |
| `scripts/llm/serve_qwen.sh` | vLLM 0.33.0 server settings |

- **Model:** `Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`.
- **Mode:** thinking on.
- **Sampling:** the model card's settings for thinking mode (temperature 0.6, top_p 0.95, top_k 20), seed 0.
- **Tool calls:** `tool_choice="required"`.

## Design data (already seen, disclosed)
Seed 100 for every mission was used to debug the code.
- **The fix:** unconstrained, Qwen's `commit_plan` calls (which have nested arguments) came out as malformed JSON, so
  `tool_choice="required"` was added.
- **No prompt change.**
- **What was seen on seed 100:** all five plans were accepted; the target matched the reference in 3 of 5 missions
  (open_drawer, close_fridge, nav_mats); close_drawer and close_cabinet chose a different drawer or cabinet.

These are reported as design results, separately.

## Test set
Five missions × seeds 0–9 = 50 scenes, built exactly as `run_episode` builds them (for nav_mats, with the mats on
the floor). One planner run per scene. Every scene is reported.

## Measures (per scene)
1. **Plan accepted:** `commit_plan` accepted within 16 model turns.
2. **Target correct:** the entity in the plan's target role equals the reference role's node (the hand-written role
   resolved through the oracle lookup). Every mission has one role.
3. **Segments correct:** the list of segment kinds equals the reference's (`[VLA]`, or `[NAVIGATION]` for nav_mats).
4. **Exemptions correct:** the exempted entities in each segment equal the reference's: {target} for the four VLA
   missions, and nothing for nav_mats.
5. Also reported: model turns and wall time.

A scene without an accepted plan counts as wrong on measures 2–4.

## Comparison rule (fixed, no LLM)
"Nearest matching label":
- take the longest label name from the label space that appears in the instruction;
- pick the nearest entity with that label (nearest by distance from the robot to the box);
- add one segment (NAVIGATION if the instruction starts with "Navigate", otherwise VLA) and exempt that entity.

It is scored with the same measures. It shows how much the planner adds over a simple lookup.

## Reporting
- Per mission: counts out of 10.
- Overall: counts out of 50 with a Wilson 95% interval, for both the planner and the comparison rule.
- No pass/fail threshold: this is a first measurement of the planner, not a test of a claim.
- Failures are described from the transcripts (which tool calls were made, what was chosen).

## After the results
- No change to the prompt, tools, settings, label space or scoring on the basis of these results.
- Any later change is reported as post hoc and judged only on new seeds (10–19).
- The hand-written reference exemptions are a choice, not ground truth. A disagreement on measure 4 is reported as a
  disagreement, not automatically as a planner error.
