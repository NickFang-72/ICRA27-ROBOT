# Geometry/Affordance Probe

This folder contains scripts and review outputs for the X-ICM-style geometry and affordance retrieval pilot.

## Structure

- `batch_01/`: first 12 seen-task demos.
- `batch_02/`: second 12 seen-task demos, excluding batch 1 demo IDs.
- `scripts/`: CAIR/runtime scripts for sampling, Qwen geometry extraction, RoboPoint affordance extraction, normalization, review index generation, and prompt rendering/preparation.
- `cair_setup_scripts/`: setup/download runner scripts used on CAIR.
- `fixtures/`: lightweight local fixtures for prompt-format checks.
- `ablation_results/`: local copies of completed CAIR benchmark logs plus regenerated score tables.

## Model Roles

- Qwen2.5-VL is used only for geometry descriptions.
- RoboPoint is used only for affordance/contact keypoints.

## Review Workflow

Open each batch's `review_index.md` first. For a deeper per-demo view, open:

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

The trial folders are not physically reorganized yet because v1/v2-related
files are dirty in the current worktree and a local v2 watcher was active during
the 2026-06-21 documentation pass. Keep using the method names below as the
stable version identifiers until the repo is clean and no run/watch process is
active.

| Version | Ranking method | Demo K | Purpose |
|---|---|---:|---|
| v1 | `lang_vis.out.geo_aff` | 18 | First descriptor ablation: combine dynamics with geometry and affordance similarity. |
| v2 | `lang_vis.out.geo_aff_v2` | 6, 8, 10 | Compact rerun with dynamics-heavy scoring, interaction signatures, transfer penalties, and attention bias. |
| v3 | `lang_vis.out.geo_aff_v3` | 6 | Contact-mode retrieval with mechanical compatibility, stronger conflict penalties, and diversity caps. |
| v4 | `lang_vis.out.geo_aff_v4` | 6 | Two-stage semantic bottleneck: descriptor-heavy context produces a simple manipulation intent, then the intent plus seen observation/action trajectories predicts 7D actions. |

Short difference between v2 and v3:

- v2 is conservative. It keeps `S_dyn` dominant and uses geometry, affordance,
  profile similarity, and conflict penalties as tie-breakers.
- v3 is more mechanical. It gives contact-mode compatibility the same order of
  influence as dynamics, penalizes known bad analogies harder, and prevents the
  top-k prompt from being dominated by repeated near-duplicate demos.

As of the latest completed CAIR run, v2 is complete and v3 `k=6` is complete
with an average score of `20.00`. V3 did not help because the model still had
to convert several descriptor-heavy demonstrations directly into one 7D action
chain, so bad or conflicting demo rhythms could still dominate the output.

V4 tests a cleaner split:

```text
Stage 1: descriptors + scene summaries -> semantic manipulation plan
Stage 2: semantic plan + seen observation/action trajectories + unseen current observation -> 7D actions
```

The active v4 method name is:

```text
XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v4_Qwen2.5.7B.instruct_icl.6_test
```

The v4 progress file is:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations/progress_v4.json
```

## V2 Geometry/Affordance Ablation

The first geometry/affordance ablation underperformed because the descriptor scores were too coarse and sometimes pushed retrieval toward bad analogies, such as shape-sorter insertion demos for phone docking or charger unplugging. V2 keeps the original dynamic diffusion/X-ICM score as the main anchor, then uses geometry and affordance as controlled tie-breakers instead of letting primitive descriptor words dominate.

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

Current v3 run:

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

Launch v3 on CAIR:

```bash
ssh cair 'cd /data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache && nohup bash scripts/run_geometry_affordance_v3_on_cair.sh > logs/run_geometry_affordance_v3.nohup.log 2>&1 & echo pid=$!'
```

Watch progress from the Mac:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_xicm_v3_progress_from_local.sh
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

Generated `.DS_Store`, `__pycache__`, `.pyc`, local `.pid`, `.lock`, and runtime `.log` files are ignored or treated as disposable workspace clutter. Do not delete benchmark result CSV/Markdown files, pulled CAIR logs, descriptor caches, or demo data unless a user explicitly asks.

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
