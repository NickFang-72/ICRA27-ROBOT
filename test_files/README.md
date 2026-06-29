# Test Files

This folder contains experiment code, launch scripts, lightweight fixtures, and local-only generated artifacts for the AGNOSTOS geometry/affordance probe.

## Git Tracking Policy

Track code and small reproducibility fixtures. Do not track generated experiment outputs.

Ignored local artifacts include:

- `geometry_affordance_probe/batch_*/`
- `geometry_affordance_probe/ablation_results/`
- `geometry_affordance_probe/review/`
- `geometry_affordance_probe/figures/`
- `xicm_baseline_results/`

These folders can contain rendered demo images, Qwen/RoboPoint review JSON, CAIR logs, result CSVs, figures, and benchmark summaries. They stay on this machine or on CAIR and should be regenerated or pulled when needed.

## Trial Map

The active experiment line is `geometry_affordance_probe/`. Trial versions are tracked by ranking method and result folder rather than by physically moving files:

| Version | Method | Notes |
|---|---|---|
| v1 | `lang_vis.out.geo_aff` | First geometry/affordance ablation using cached Qwen geometry and RoboPoint affordance descriptors. |
| v2 | `lang_vis.out.geo_aff_v2` | Adds precise interaction signatures, transfer penalties, and attention-bias prompt guidance. Completed compact K sweep for `k=6,8,10`. |
| v3 | `lang_vis.out.geo_aff_v3` | Adds contact-mode/mechanical compatibility, diversity caps, stronger conflict penalties, and task-specific action guidance. Completed `k=6`, average `20.00`; retrieval was more physical but still required a direct descriptor-to-7D conversion. |
| v4 | `lang_vis.out.geo_aff_v4` | Uses a two-stage semantic bottleneck. Stage 1 writes a grounded semantic manipulation plan; Stage 2 returns a relative action sketch plus final `key_actions_7d` in one call. Improved `k=6` rerun is active on CAIR. |

Do not reorganize these folders while watcher/evaluator processes are active. On 2026-06-21, v1/v2-related files were dirty and a v2 watcher was active, so the safe organization path is documentation only until the tree is clean.

Baseline and ablation comparisons should now use seeds `0,50,99`, with `25` evaluation episodes per task per seed. The current vanilla X-ICM baseline rerun follows that protocol and writes its comparison files under `xicm_baseline_results/vanilla_rerun_3seed_2026-06-22/`.

Current vanilla 3-seed baseline watcher:

```bash
INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_baseline_rerun_from_local.sh
```

Current improved v4 watcher:

```bash
INTERVAL_SECONDS=120 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_and_update_xicm_v4_progress_from_local.sh
```

## Main Folder

- `geometry_affordance_probe/`

## Batches

- `geometry_affordance_probe/batch_01/` local-only, ignored
  - First 12 seen-task demos.
  - One initial-state image per demo from the original pilot snapshot.
  - Qwen2.5-VL geometry outputs and RoboPoint affordance outputs.

- `geometry_affordance_probe/batch_02/` local-only, ignored
  - Second 12 seen-task demos, excluding all batch 1 demo IDs.
  - Two uniquely named initial-state views per demo: `view0_*.png` and `view1_*.png`.
  - Qwen2.5-VL geometry outputs and RoboPoint affordance outputs.

## Files To Open First

If the local batch folders are present:

- `geometry_affordance_probe/batch_01/review_index.md`
- `geometry_affordance_probe/batch_02/review_index.md`

Each batch also includes:

- `manifest.json`: selected demos and image paths.
- `review_index.json`: structured index for the batch.
- `review_bundle.jsonl`: one combined review record per demo.
- `human_check_bundle/*/combined_review.json`: per-demo geometry plus affordance record.
- `demos/*/geometry_qwen2_5_vl.json`: Qwen geometry descriptor.
- `demos/*/affordance_robopoint.json`: RoboPoint affordance/contact descriptor.

The CAIR live experiment root is:

`/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe`
