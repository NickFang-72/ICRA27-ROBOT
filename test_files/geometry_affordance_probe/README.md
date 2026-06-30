# Geometry/Affordance Probe

This folder contains the current X-ICM geometry/contact ablation code plus a
small amount of historical experiment code. Generated result files, CAIR logs,
review packets, figures, and local batch outputs are intentionally ignored by
Git.

For the script-by-script map, start with:

- `SCRIPT_INDEX.md`

## Active Pipeline

The current pipeline is the closed-loop no-plan QwenVL ablation:

1. Build a seen-demo cache from the 18 seen AGNOSTOS families.
2. For each seen demo, cache QwenVL geometry and target-pose descriptors from
   front plus overhead observations.
3. At evaluation time, observe the unseen scene, retrieve top-k seen demos with
   dynamics plus geometry/target-pose scoring, and prompt QwenVL with the
   current front plus overhead images.
4. In closed loop, execute one primitive, observe again, retrieve again from the
   new scene state, and repeat for the configured replans.
5. Compare the geometry-only row with the geometry plus contact-points row.

Plan-guided retrieval is not part of the active run. The active ranking methods
are:

- `lang_vis.out.geo.closed_loop`
- `lang_vis.out.geo.aff.closed_loop`

## Main Commands

Clean geometry/target-pose seen cache:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/launch_full_seen_geometry_target_pose_v2_cache_on_cair.sh
```

Baseline Qwen text versus QwenVL front+overhead:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/run_xicm_qwen_vs_qwenvl_front_top_baseline_on_cair.sh
```

Closed-loop no-plan ablation:

```bash
DEMO_NUM_PER_ICL=10 bash test_files/geometry_affordance_probe/cair_setup_scripts/run_xicm_qwenvl_closed_loop_no_plan_5ep_ablation_on_cair.sh
```

Closed-loop watcher and local CSV collector:

```bash
INTERVAL_SECONDS=120 \
COLLECT_METHOD_SET=closed_loop_no_plan \
RANKING_METHODS=lang_vis.out.geo.closed_loop,lang_vis.out.geo.aff.closed_loop \
bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_qwenvl_5ep_component_ablation_from_local.sh
```

## Current Results

The combined k-sweep and baseline comparison lives locally at:

```text
test_files/geometry_affordance_probe/ablation_results/closed_loop_no_plan_k4_k6_k8_k10_comparison_2026-06-30.csv
```

Latest averages:

| Row | Average |
|---|---:|
| Qwen text baseline | 26.96 |
| QwenVL front+overhead baseline | 28.70 |
| closed loop geo, k10 | 15.65 |
| closed loop geo+contact, k10 | 13.04 |
| closed loop geo, k8 | 18.26 |
| closed loop geo+contact, k8 | 15.65 |
| closed loop geo, k6 | 17.39 |
| closed loop geo+contact, k6 | 13.04 |
| closed loop geo, k4 | 16.52 |
| closed loop geo+contact, k4 | 13.04 |

## Folder Layout

- `scripts/`: active Python utilities for cache building, QwenVL descriptor
  extraction, retrieval scoring, prompt rendering, and result collection.
- `cair_setup_scripts/`: active CAIR launch, watch, download, and sync wrappers.
- `legacy/`: older v1-v4 launchers, one-off review builders, old collectors,
  and pilot scripts. Keep these for reference, but do not start new work there.
- `fixtures/`: small prompt-format fixtures that are safe to commit.
- `ablation_results/`: local-only pulled logs and CSV/Markdown outputs.
- `review/`, `figures/`, `batch_*/`: local-only generated inspection artifacts.

## Git Tracking Policy

Commit source code, launch scripts, fixtures, and documentation. Do not commit:

- demo/review images
- model descriptor JSON from generated Qwen/RoboPoint runs
- CAIR logs and pulled benchmark folders
- ablation CSV/Markdown/JSON result tables
- generated figures and presentation exports

Those outputs stay on this machine or on CAIR. Regenerate or pull them when
needed.

