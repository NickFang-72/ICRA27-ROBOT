# Geometry/Affordance Probe

This folder contains the reproducible code for the X-ICM-style geometry and affordance retrieval pilot. Generated review packets, result tables, figures, and CAIR pulls live here locally, but are intentionally ignored by Git.

## Structure

- `scripts/`: CAIR/runtime scripts for sampling, Qwen geometry extraction, RoboPoint affordance extraction, normalization, review index generation, and prompt rendering/preparation.
- `cair_setup_scripts/`: setup/download runner scripts used on CAIR.
- `fixtures/`: lightweight local fixtures for prompt-format checks.
- `batch_*/`: local-only generated seen-demo review batches.
- `ablation_results/`: local-only copies of CAIR benchmark logs plus regenerated score tables.
- `review/`: local-only human-review packets and rendered debug views.
- `figures/`: local-only generated diagrams and paper/deck figures.

## Git Tracking Policy

Keep source code, launch scripts, fixtures, and documentation in Git. Keep generated artifacts out of Git:

- demo/review images
- model descriptor JSON from Qwen/RoboPoint runs
- CAIR logs and pulled benchmark folders
- ablation CSV/Markdown/JSON result tables
- generated figures and presentation exports

The ignored folders remain available on this machine. If another clone needs them, regenerate them with the scripts here or pull them again from CAIR.

## Model Roles

- Qwen2.5-VL is used only for geometry descriptions.
- RoboPoint is used only for affordance/contact keypoints.

## Review Workflow

If local batch folders are present, open each batch's `review_index.md` first. For a deeper per-demo view, open:

`human_check_bundle/<demo_id>/combined_review.json`

The combined review files contain:

- `geometry_g_i`: normalized geometry descriptor and `key_features`.
- `affordance_a_i`: RoboPoint contact points plus normalized affordance labels.
- `source_files`: raw per-model output file paths.

## Prompt Preparation

The descriptor batches cache demo-level `geometry_g_i` and `affordance_a_i` from the current/initial seen observation frame. The prompt-augmented X-ICM path is stricter: each retrieved seen demonstration must be rendered as a sequence of per-key-action observations paired with the corresponding 7D action.

- `scripts/prepare_xicm_key_action_trajectories.py`: run on CAIR inside the X-ICM environment to build JSON payloads from retrieved seen episode paths. It calls the X-ICM observation renderer at every keypoint state, not just the first state.
- `scripts/render_xicm_geometry_affordance_prompt.py`: local/CAIR renderer for the geometry/affordance prompt. It validates that seen demos have `steps` and that the unseen query has only the current observation plus descriptors.
- `scripts/score_xicm_geometry_affordance_retrieval.py`: ablation-only top-k retriever for `score = alpha*S_dyn + beta*S_geo + gamma*S_aff`. It joins original X-ICM dynamic scores with cached `g_i/a_i`.
- `scripts/tune_geometry_affordance_weights.py`: seen-task validation tuner for `alpha`, `beta`, and `gamma`. It rejects non-seen query/positive tasks.
- `scripts/cache_all_seen_geometry_affordance.py`: full-cache runner for all seen demonstrations. It builds a one-row-per-episode manifest, runs Qwen geometry, runs RoboPoint affordance, normalizes descriptors, and writes a resumable progress file.
- `fixtures/paper_faithful_prompt_fixture.json`: tiny fixture used to check the expected `observation -> 7D action` trajectory shape.

The vanilla X-ICM baseline prompt path remains separate and unchanged.

## Trial Version Map

Keep using the method names below as the stable version identifiers. They map
directly to X-ICM log folders, local table rows, and CAIR runner/watcher scripts.
The vanilla X-ICM baseline remains separate and should not be overwritten by
any geometry/affordance trial.

| Version | Ranking method | Demo K | Main change | What failed / lesson |
|---|---|---:|---|---|
| baseline | `lang_vis.out` | 18 | Original X-ICM dynamic-diffusion retrieval and one-pass action prompt. | This is the control path. Preserve it unchanged. |
| v1 | `lang_vis.out.geo_aff` | 18 | Add cached Qwen geometry `g_i/g_j` and RoboPoint affordance `a_i/a_j` into retrieval and prompt context. | Coarse descriptor words sometimes pulled in bad analogies. Average `20.87`, below the X-ICM 7B rerun. |
| v2 | `lang_vis.out.geo_aff_v2` | 6, 8, 10 | Make dynamics dominant again, add interaction signatures, transfer penalties, and prompt-visible attention bias. | Better than v1 but still below baseline. Best compact average tied at `21.74` for `k=6` and `k=8`. The first `k=18` v2 attempt overflowed context. |
| v3 | `lang_vis.out.geo_aff_v3` | 6 | Retrieve by contact-mode/mechanical compatibility with stronger conflict penalties and diversity caps. | Average `20.00`. Retrieval became more physically opinionated, but the LLM still had to convert descriptor-heavy multi-demo context directly into one 7D chain. |
| v4 | `lang_vis.out.geo_aff_v4` | 6 | Two-stage semantic bottleneck: Stage 1 turns descriptor context into a grounded intent; Stage 2 emits a relative action sketch plus final 7D actions in one call. | Improved clean rerun active on CAIR. The old partial 4/23 v4 outputs were archived before relaunch. |

Short difference between v2, v3, and v4:

- v2 is conservative. It keeps `S_dyn` dominant and uses geometry, affordance,
  profile similarity, and conflict penalties as tie-breakers.
- v3 is more mechanical. It gives contact-mode compatibility the same order of
  influence as dynamics, penalizes known bad analogies harder, and prevents the
  top-k prompt from being dominated by repeated near-duplicate demos.
- v4 changes the prompt architecture. It still retrieves once, but calls the
  same LLM twice. Stage 1 produces a clean semantic manipulation plan. Stage 2
  is one call that first emits `relative_action_sketch`, then emits
  `key_actions_7d` for the evaluator.

As of the latest completed CAIR run, v2 is complete and v3 `k=6` is complete
with an average score of `20.00`. V3 did not help because the model still had
to convert several descriptor-heavy demonstrations directly into one 7D action
chain, so bad or conflicting demo rhythms could still dominate the output.

V4 tests a cleaner split:

```text
Stage 1: descriptors + scene summaries -> semantic manipulation plan
Stage 2: semantic plan + seen observation/action trajectories + unseen current observation -> relative action sketch + 7D actions
```

The active v4 method name is:

```text
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v4_Qwen2.5.7B.instruct_icl.6_test
```

The v4 progress file is:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations/progress_v4.json
```

The current improved v4 rerun started from an empty active v4 method folder.
The earlier stopped partial run is archived under `_archived_partial_v4` so it
does not contaminate the active wide table.

The stopped-run review packet for the next meeting is:

```text
test_files/geometry_affordance_probe/review/2026-06-22_v4_research_holes
```

## V1 Geometry/Affordance Ablation

V1 was the direct version of the original idea:

```text
score = alpha*S_dyn + beta*S_geo + gamma*S_aff
```

It used the original X-ICM dynamic score plus geometry and affordance similarity
from the cached descriptor bundle:

- `S_dyn`: original X-ICM dynamic diffusion/prompt-output similarity.
- `S_geo`: similarity between cached geometry descriptors `g_i` and query geometry `g_j`.
- `S_aff`: similarity between cached affordance descriptors `a_i` and query affordance `a_j`.

V1 also rendered the retrieved demos in the paper-faithful prompt shape:

```text
Step 1 observation -> Step 1 7D action
Step 2 observation -> Step 2 7D action
...
```

What failed:

- The descriptor vocabulary was too coarse. Labels such as `slot`, `handle`,
  `pull`, `container`, and `alignment` matched tasks that looked physically
  similar in words but required different robot behavior.
- Geometry/affordance terms could overrule useful dynamic retrieval instead of
  acting as small corrections.
- The LLM received more context, but not necessarily cleaner context. Extra
  descriptors sometimes amplified bad retrieved demos.

Result: v1 completed with average `20.87`, below the local X-ICM 7B rerun
average `22.09` and below the paper X-ICM 7B average `23.54`.

## V2 Geometry/Affordance Ablation

The first geometry/affordance ablation underperformed because the descriptor
scores were too coarse and sometimes pushed retrieval toward bad analogies, such
as shape-sorter insertion demos for phone docking or charger unplugging. V2
keeps the original dynamic diffusion/X-ICM score as the main anchor, then uses
geometry and affordance as controlled tie-breakers instead of letting primitive
descriptor words dominate.

V2 adds a derived interaction signature, also called `p_i`/`p_j` or the
interaction profile. This is not a new model output. It is engineered from the
cached geometry/affordance fields plus task-specific overrides when available:

```text
interaction_family
motion_sequence
contact_strategy
target_relation
axis_constraint
articulation_model
precision_driver
transfer_caution
```

The goal was to compare "what physical manipulation is this?" rather than only
matching individual descriptor words.

V2 runs under a separate ranking method and K-sweep result folders:

```text
lang_vis.out.geo_aff_v2
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.6_test
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.8_test
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.10_test
```

The earlier `icl.18` v2 attempt is intentionally excluded from the generated
wide table because it failed on a prompt-length overflow. The compact sweep
keeps the same ranking formula and prompt ideas, but passes fewer top-ranked
demos to the LLM so the demonstrations are tighter and the prompt stays inside
the 7B model context window.

The v2 score is:

```text
score = alpha*S_dyn + beta*S_geo + gamma*S_aff + delta*S_profile - penalty_weight*S_penalty
```

Default v2 weights:

```text
alpha = 0.82
beta = 0.04
gamma = 0.04
delta = 0.22
penalty_weight = 0.30
```

Where:

- `S_dyn` is the original X-ICM dynamic diffusion/prompt-output similarity, min-max normalized per query.
- `S_geo` is the cached geometry descriptor similarity.
- `S_aff` is the cached affordance descriptor similarity.
- `S_profile` compares a derived precise interaction signature: interaction family, motion sequence, contact strategy, target relation, axis constraint, articulation model, and precision driver.
- `S_penalty` downweights bad transfer analogies, such as insertion demos for pull-out tasks, shape-sorter demos for flat phone docking, and non-button demos for button/switch tasks.

Each retrieved v2 demo also gets a prompt-visible attention score:

```text
Attention bias: 0.00 to 1.00
```

The prompt tells the LLM to treat high-bias demos as primary analogies, mid-bias demos as supporting evidence, and low-bias demos as weak fallback context. This is prompt-level attention guidance, not a transformer attention-mask modification.

What improved:

- Retrieval stopped being dominated by primitive descriptor overlap.
- Smaller `K` reduced prompt length and avoided the failed `k=18` overflow.
- Attention bias made the ranking visible to the LLM instead of presenting all
  demos as equally trustworthy.

What still failed:

- The LLM still received several full trajectories and had to decide by itself
  which action rhythm to follow.
- For tasks where all top demos were weak or misleading, attention bias could
  warn the model but could not create a correct action pattern.
- V2 best average was `21.74`, still below the original X-ICM 7B rerun average
  `22.09`.

Launch the compact v2 K sweep on CAIR:

```bash
ssh cair 'cd /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache && nohup bash scripts/run_geometry_affordance_v2_k_sweep_on_cair.sh > logs/run_geometry_affordance_v2_k_sweep.nohup.log 2>&1 & echo pid=$!'
```

Watch progress from the Mac:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v2_k_sweep_progress_from_local.sh
```

Pull logs and regenerate the table:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/pull_xicm_ablation_results_from_cair.sh
python3 test_files/geometry_affordance_probe/scripts/collect_xicm_ablation_results.py
```

Final validation, once the K sweep is complete:

```bash
python3 test_files/geometry_affordance_probe/scripts/collect_xicm_ablation_results.py --require-complete
```

## V3 Contact-Mode Retrieval

V3 targets the zero-score failure cases from the compact K sweep. It keeps the
vanilla X-ICM baseline separate, but replaces the coarse descriptor tie-breaker
with a mechanical compatibility score that compares:

- interaction family/contact mode
- target relation
- motion sequence
- required contact region
- axis and clearance constraints
- affordance motion/contact labels

V3 also adds stronger transfer penalties for bad analogies, caps duplicate
retrievals from the same seen task/family, and adds prompt guidance for
contact-sensitive tasks such as hole-over-peg placement, shelf insertion, hoop
release, and spatula scooping.

The idea was that some failures were not "more context" problems, but
"wrong physical analogy" problems. For example:

- hoop release should prefer open-goal or align-and-release demos, not stacking;
- spatula scooping should prefer tool-under-object sliding, not direct grasping;
- charger unplugging should prefer linear extraction, not insertion or twisting;
- shelf placement should prefer front-opening clearance, not simple holder
  placement.

What failed:

- The retrieval became more mechanically opinionated, but the prompt was still
  one pass from descriptor-heavy context to 7D actions.
- Strong penalties helped avoid some bad analogies but could also remove
  dynamically useful demos.
- The LLM could still blend several incompatible demo rhythms when converting
  the full prompt into actions.
- Average score was `20.00`, worse than v2 and the original rerun.

The main lesson from v3 is that improving retrieval alone is not enough if the
generation prompt still asks the 7B model to simultaneously understand the task,
resolve conflicting demos, and emit precise action coordinates.

Completed v3 run:

```text
lang_vis.out.geo_aff_v3
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v3_Qwen2.5.7B.instruct_icl.6_test
```

Default v3 weights:

```text
score = alpha*S_dyn + beta*S_geo + gamma*S_aff + delta*S_mech - penalty_weight*S_conflict

alpha = 0.45
beta = 0.10
gamma = 0.10
delta = 0.45
penalty_weight = 0.60
max_per_task = 2
max_per_family = 3
```

Historical v3 launch command:

```bash
ssh cair 'cd /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache && nohup bash scripts/run_geometry_affordance_v3_on_cair.sh > logs/run_geometry_affordance_v3.nohup.log 2>&1 & echo pid=$!'
```

Historical v3 one-shot progress check:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v3_progress_from_local.sh
```

## V4 Semantic Bottleneck

V4 tests a different hypothesis: the LLM may need to separate semantic
understanding from coordinate prediction. Instead of asking the model to inspect
all descriptors and trajectories and immediately output 7D actions, v4 splits
the generation into two calls to the same Qwen2.5 7B model.

V4 still performs only one retrieval per unseen episode:

```text
unseen query -> score all 3,600 seen demos -> keep top K=6
```

The retrieved top-k demos are reused in both stages. V4 does not retrieve a
second time.

### V4 Retrieval

V4 uses the same style of dynamics-anchored mechanical retrieval as v3, but with
less aggressive weights:

```text
score = alpha*S_dyn + beta*S_geo + gamma*S_aff + delta*S_mech - penalty_weight*S_conflict

alpha = 0.70
beta = 0.05
gamma = 0.05
delta = 0.40
penalty_weight = 0.45
max_per_task = 2
max_per_family = 3
K = 6
```

This keeps original X-ICM dynamics as the anchor while still allowing mechanical
compatibility to influence which demos make it into the prompt.

### Stage 1: Semantic Manipulation Plan

Stage 1 receives descriptor-heavy context, but no seen 7D action trajectories.
For each retrieved seen demo, Stage 1 sees:

```text
task instruction
retrieval scores
attention bias
interaction signature p_i
geometry description g_i
affordance description a_i
scene summary s_i
```

For the unseen query, Stage 1 sees:

```text
task instruction
task key
current observation
geometry description g_j
affordance description a_j
interaction signature p_j
scene summary s_j
```

Stage 1 must output only a compact semantic JSON plan. It should identify the
manipulation in simple words and bind that plan to the unseen observation's
object coordinates. In particular, it should copy the target object's current
coordinate, choose the active reference part, and copy that reference
coordinate. This prevents errors such as aiming at `stand_base` when the active
target should be `holder`.

Example Stage 1 output:

```json
{
  "target_object": "charger plug",
  "target_current_coordinate": [61, 52, 29],
  "reference_object": "computer socket",
  "active_reference_part": "socket",
  "reference_coordinate": [50, 22, 29],
  "target_location_relation": "out of the socket",
  "target_orientation": "aligned with the plug axis",
  "action_primitive": "pull",
  "motion_direction": "straight outward from the socket",
  "contact_point": "plug body",
  "gripper_plan": "pinch the plug, keep closed while pulling, open after removal",
  "success_relation": "plug fully removed from socket",
  "constraints": "avoid twisting; maintain axis alignment",
  "demo_use_hint": "prefer linear extraction demos over insertion or rotation demos"
}
```

Stage 1 is deliberately not allowed to output 7D actions. The goal is to force
the model to first answer: "what manipulation is this?"

### Stage 2: 7D Key Action Prediction

Stage 2 receives:

```text
Stage 1 semantic plan
top K seen demonstrations with observation -> 7D action pairs
unseen task instruction
unseen current observation
```

Stage 2 does not receive the full geometry/affordance blocks again. That is an
intentional bottleneck: descriptor information must pass through the Stage 1
semantic plan instead of being dumped into the final action prompt.

Stage 2 is a single LLM call, but it now has two fields in one JSON output:

```json
{
  "relative_action_sketch": [
    "approach the target object",
    "close gripper on the target",
    "move toward the reference coordinate",
    "align with the relevant axis or contact surface",
    "lower/place/pull/press/rotate as required",
    "open gripper or retreat when complete"
  ],
  "key_actions_7d": [
    [x, y, z, roll, pitch, yaw, gripper]
  ]
}
```

The evaluator still consumes only `key_actions_7d`; `relative_action_sketch`
is an internal bridge to make the final numeric action sequence less noisy.
This follows the same practical lesson as VLA and embodied-reasoning prompts:
ask the model what action should be taken, make it represent the motion in a
clean action language first, then emit the robot action representation.

The intended behavior is not to average all retrieved demos. Stage 2 should use
the semantic plan to choose one or a few compatible demo rhythms, then adapt the
coordinate/action style to the unseen observation.

### What V4 Is Testing

V4 is testing whether this decomposition fixes the v3 failure mode:

```text
v3: descriptors + many trajectories -> 7D actions
v4: descriptors -> semantic intent -> relative action sketch -> 7D actions
```

A v4 score above v2/v3 would suggest that the semantic bottleneck helps the 7B
model avoid noisy trajectory mixing. A v4 score below v2/v3 would suggest that
the extra LLM call either loses important descriptor detail, introduces semantic
errors, or still cannot recover from poor retrieved demos.

### Saved Next-Step Idea: Stage 1 Demo Role Selection

If the current v4 rerun still struggles, the next prompt-side idea is to let
Stage 1 assign roles to the retrieved demos before Stage 2 sees the action
trajectories. Instead of treating all retrieved examples equally, Stage 1 would
label each demo as something like `primary_motion_template`,
`gripper_timing_reference`, `axis_or_contact_reference`, or `ignore`.

This would allow a slightly larger retrieval pool, likely `k=8` or `k=10`,
while still asking the model to down-select the useful examples before action
prediction. The goal is not to return to noisy `k=18`; it is to retrieve a few
extra candidates and make Stage 1 explicitly say which demos should guide Stage
2 and which demos should be ignored because their mechanics conflict with the
unseen query.

Example role block:

```json
{
  "demo_roles": [
    {
      "demo_rank": 1,
      "role": "primary_motion_template",
      "reason": "same hole-over-vertical-stand mechanics"
    },
    {
      "demo_rank": 2,
      "role": "gripper_timing_reference",
      "reason": "same grasp-lift-release rhythm"
    },
    {
      "demo_rank": 3,
      "role": "ignore",
      "reason": "similar object placement but wrong contact geometry"
    }
  ]
}
```

### V4 Commands

Launch v4 on CAIR:

```bash
ssh cair 'cd /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache && nohup bash scripts/run_geometry_affordance_v4_on_cair.sh > logs/run_geometry_affordance_v4.nohup.log 2>&1 & echo pid=$!'
```

Watch progress from the Mac:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v4_progress_from_local.sh
```

Watch progress continuously and update local tables whenever new strict final
scores appear:

```bash
INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v4_progress_from_local.sh
```

Update local result tables after completed strict final scores appear:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/pull_xicm_ablation_results_from_cair.sh
python3 test_files/geometry_affordance_probe/scripts/collect_xicm_ablation_results.py
```

## Organization Rule

Do not move, rename, or consolidate trial scripts/results while any of these are
true:

- `git status --short` shows modified v1/v2 trial files.
- a local `watch_xicm_*` process is active.
- CAIR has an active `run_geometry_affordance*`, `eval_XICM.sh`, or `python main.py`
  process for a trial method.

When the repo is clean and no watcher/evaluator is active, the preferred
organization is an index-first layout: keep executable scripts in
`cair_setup_scripts/` for compatibility, and add version-specific docs or
manifest files pointing to the exact method folders instead of breaking existing
paths.

## Folder Hygiene

Generated `.DS_Store`, `__pycache__`, `.pyc`, local `.pid`, `.lock`, runtime `.log`, result CSV/Markdown/JSON, pulled CAIR logs, descriptor caches, review packets, demo images, and generated figures are ignored or treated as local workspace artifacts. Keep them locally when useful, but do not commit them.

To launch the full seen-demo descriptor cache on CAIR:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/launch_full_seen_cache_on_cair.sh
```

To watch progress:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_full_seen_cache_progress.sh
```

The full seen-demo cache completed on CAIR at:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl
```

The completion state was `geometry=3600/3600`, `affordance=3600/3600`, and `combined=3600/3600`.

## Retrieval And Tuning

Score a query against seen candidates with cached descriptors:

```bash
python test_files/geometry_affordance_probe/scripts/score_xicm_geometry_affordance_retrieval.py \
  --descriptor-cache /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl \
  --dynamic-scores path/to/query_dynamic_scores.jsonl \
  --query-geometry path/to/query_geometry_g_j.json \
  --query-affordance path/to/query_affordance_a_j.json \
  --alpha 1 --beta 0.5 --gamma 0.5 \
  --top-k 18 \
  --out path/to/ranked_topk.json
```

Then pass `ranked_topk.json` into prompt preparation:

```bash
python test_files/geometry_affordance_probe/scripts/prepare_xicm_key_action_trajectories.py \
  --retrieval-ranking path/to/ranked_topk.json \
  --descriptor-cache /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl \
  --query-task-instruction "..." \
  --query-observation "..." \
  --query-geometry path/to/query_geometry_g_j.json \
  --query-affordance path/to/query_affordance_a_j.json \
  --out path/to/prompt_payload.json
```

Tune weights only with seen-task validation rows:

```bash
python test_files/geometry_affordance_probe/scripts/tune_geometry_affordance_weights.py \
  --descriptor-cache /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl \
  --validation-queries path/to/seen_validation_queries.jsonl \
  --top-k 18 \
  --out path/to/tuned_weights.json
```

## Leakage Rule

The batches are sampled from seen-task data only. They do not include unseen AGNOSTOS demonstrations, future frames, or after-state frames. At benchmark time, unseen tasks should contribute only the current observation/instruction used to compute `g_j` and `a_j`; never unseen future observations or ground-truth actions.
