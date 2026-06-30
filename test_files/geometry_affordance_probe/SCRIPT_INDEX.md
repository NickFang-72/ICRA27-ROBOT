# Geometry/Affordance Script Index

Use this file as the map for the cleaned probe folder. Active code stays in
`scripts/` and `cair_setup_scripts/`. Historical one-offs and superseded
experiment launchers live in `legacy/`.

## Active Python Scripts

| Script | Purpose |
|---|---|
| `scripts/apply_qwen_geometry_strict_rules.py` | Applies deterministic cleanup rules to QwenVL geometry descriptors. |
| `scripts/cache_all_seen_geometry_affordance.py` | Builds the full seen-demo manifest/cache and runs geometry normalization stages. |
| `scripts/collect_xicm_qwenvl_5ep_component_ablation_results.py` | Collects QwenVL 5-episode ablation logs into paper-style CSV/Markdown tables, including closed-loop no-plan runs. |
| `scripts/prepare_xicm_key_action_trajectories.py` | Converts retrieved seen episodes into observation/action trajectory payloads for X-ICM-style prompts. |
| `scripts/project_robopoint_contacts_to_pointcloud.py` | Utility for projecting contact points into scene geometry when contact diagnostics are needed. |
| `scripts/qwen_retrieval_geometry_rules.py` | Shared deterministic rules for geometry/target-pose retrieval descriptors. |
| `scripts/render_xicm_geometry_affordance_prompt.py` | Renders prompt payloads with descriptor and key-action trajectory context. |
| `scripts/run_qwen_dual_view_geometry_target_pose.py` | Runs QwenVL on front plus overhead views and returns geometry plus target-pose JSON. |
| `scripts/run_qwen_dual_view_retrieval_geometry.py` | Runs QwenVL geometry extraction for retrieval-oriented scene descriptors. |
| `scripts/score_xicm_geometry_affordance_retrieval.py` | Scores retrieved demonstrations using X-ICM dynamics plus geometry/contact descriptor terms. |
| `scripts/tune_geometry_affordance_weights.py` | Tunes old-style geometry/contact retrieval weights on seen validation data. |
| `scripts/tune_seen_validation_from_xicm_features.py` | Tunes current retrieval features from seen validation/X-ICM feature tables. |

## Active CAIR Scripts

| Script | Purpose |
|---|---|
| `cair_setup_scripts/cair_download_full_agnostos_and_xicm_model.sh` | Bootstrap download for AGNOSTOS data and the X-ICM model on CAIR. |
| `cair_setup_scripts/cair_download_robopoint.sh` | Bootstrap download for RoboPoint checkpoints/resources. |
| `cair_setup_scripts/cair_parallel_agnostos_seen_download.sh` | Parallel CAIR download helper for seen AGNOSTOS assets. |
| `cair_setup_scripts/launch_full_seen_geometry_target_pose_v2_cache_on_cair.sh` | Launches the active clean geometry/target-pose cache build on CAIR. |
| `cair_setup_scripts/run_xicm_qwen_vs_qwenvl_front_top_baseline_on_cair.sh` | Runs the baseline comparison: Qwen text-only versus QwenVL front+overhead. |
| `cair_setup_scripts/run_xicm_qwenvl_ablation_matrix_on_cair.sh` | Generic CAIR ablation engine used by wrapper scripts. Defaults are legacy; call wrappers for real runs. |
| `cair_setup_scripts/run_xicm_qwenvl_closed_loop_no_plan_5ep_ablation_on_cair.sh` | Main active launcher for closed-loop no-plan geometry and geometry+contact rows. |
| `cair_setup_scripts/stream_archives_to_cair_from_local.sh` | Streams local archive payloads to CAIR. |
| `cair_setup_scripts/stream_qwen25_7b_to_cair_from_local.sh` | Streams the local Qwen2.5 model payload to CAIR. |
| `cair_setup_scripts/watch_and_update_xicm_qwen_vs_qwenvl_front_top_from_local.sh` | Watches/pulls the active baseline comparison and refreshes local CSVs. |
| `cair_setup_scripts/watch_and_update_xicm_qwenvl_5ep_component_ablation_from_local.sh` | Watches/pulls QwenVL component or closed-loop no-plan ablations and runs the collector. |
| `cair_setup_scripts/watch_full_seen_geometry_target_pose_v2_cache_progress.sh` | Checks active clean geometry/target-pose cache progress on CAIR. |
| `cair_setup_scripts/xvfb-run` | Local wrapper used by headless CAIR/RLBench execution. |

## Legacy Scripts

`legacy/scripts/` contains older review builders, old single-view Qwen/RoboPoint
pilots, old v1 collectors, and normalization helpers that were superseded by the
current dual-view geometry/target-pose cache path.

`legacy/cair_setup_scripts/` contains older v1-v4 launchers and watchers,
including plan-guided and semantic-plan experiments. They are retained for
reference only. Some call paths are historical and may need adjustment before
rerunning.

