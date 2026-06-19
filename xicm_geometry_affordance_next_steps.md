# X-ICM Geometry/Affordance Next Steps

Now that the data is downloaded and the original X-ICM baseline is set, the next phase is to turn the idea into a controlled ablation study.

## 1. Freeze The Baseline

Record the completed original X-ICM score, logs, seed, prompt version, command, and output artifacts. Treat this as the fixed comparison point for every later run.

Status: done. Freeze record:

- `test_files/xicm_baseline_results/original_xicm_baseline_freeze.md`

## 2. Scale Descriptor Caching

The current geometry/affordance work is a pilot. Next, compute `g_i` and `a_i` for all seen-task demonstrations that should be eligible for retrieval, ideally the full 18 seen tasks x 200 demos = 3,600 demos.

Status: done. Full-cache runner and progress watcher:

- `test_files/geometry_affordance_probe/scripts/cache_all_seen_geometry_affordance.py`
- `test_files/geometry_affordance_probe/cair_setup_scripts/launch_full_seen_cache_on_cair.sh`
- `test_files/geometry_affordance_probe/cair_setup_scripts/watch_full_seen_cache_progress.sh`

Completed CAIR cache:

- Remote cache root: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache`
- Remote log: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/full_cache_all.log`
- Completion state at 2026-06-19: `geometry=3600/3600`, `affordance=3600/3600`, `combined=3600/3600`, stage `normalize_done`
- Review bundle: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl`
- Verification command:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_full_seen_cache_progress.sh
```

## 3. Edit Retrieval

Keep X-ICM's original dynamics score as `S_dyn`, then add geometry and affordance similarity:

```text
score = alpha*S_dyn + beta*S_geo + gamma*S_aff
```

Start simple: use text/JSON embedding similarity over `key_features`, affordance labels, contact region, motion type, and object type.

Status: implemented for the geometry/affordance ablation path in:

- `test_files/geometry_affordance_probe/scripts/score_xicm_geometry_affordance_retrieval.py`

The vanilla X-ICM `lang_vis.out` baseline path is unchanged. The new scorer reads dynamic scores from the original X-ICM retrieval or a score file, joins cached `g_i/a_i`, compares them with query `g_j/a_j`, and writes a ranked top-k JSON for prompt preparation.

## 4. Tune Only On Seen-Task Validation

Choose `alpha`, `beta`, and `gamma` using seen-task validation splits only. Do not tune on the 23 held-out AGNOSTOS unseen tasks.

Status: implemented as a seen-task validation utility:

- `test_files/geometry_affordance_probe/scripts/tune_geometry_affordance_weights.py`

The tuner rejects validation rows whose query or positive task is not in the 18 seen AGNOSTOS task names.

## 5. Render The New Prompt Format

For each retrieved seen demonstration, use the paper-faithful prompt preparation path:

```text
Step 1 observation -> Step 1 7D action
Step 2 observation -> Step 2 7D action
...
```

Then add demo-level `g_i` and `a_i` blocks. The unseen task should include only the current observation, task instruction, `g_j`, and `a_j`.

Status: implemented in the separate geometry/affordance prompt path:

- `test_files/geometry_affordance_probe/scripts/prepare_xicm_key_action_trajectories.py`
- `test_files/geometry_affordance_probe/scripts/render_xicm_geometry_affordance_prompt.py`
- `test_files/geometry_affordance_probe/prompts/xicm_geometry_affordance_prompt.md`

The prompt now explicitly says "You will receive..." and "Your job is..." and uses top-k retrieved seen demos, not just one demo.

## 6. Run Ablations

Run the same 23 unseen AGNOSTOS tasks with:

- Original X-ICM baseline.
- X-ICM + geometry retrieval/prompt.
- X-ICM + affordance retrieval/prompt.
- X-ICM + geometry + affordance.

## 7. Analyze Retrieval Quality

For each unseen task, compare which seen demonstrations were retrieved by the original baseline versus the geometry/affordance variants. The goal is to see whether the new features improve physical analogy selection, not only whether the final score changes.

## Immediate Next Step

Steps 1-5 are in place. Next, run the geometry/affordance retrieval and prompt path on seen-task validation to select `alpha/beta/gamma`, then launch the unseen ablation runs without retuning on the 23 held-out tasks.
