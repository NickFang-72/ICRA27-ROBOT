# Test Files

This folder contains local review artifacts for the AGNOSTOS geometry/affordance probe.

## Main Folder

- `geometry_affordance_probe/`

## Batches

- `geometry_affordance_probe/batch_01/`
  - First 12 seen-task demos.
  - One initial-state image per demo from the original pilot snapshot.
  - Qwen2.5-VL geometry outputs and RoboPoint affordance outputs.

- `geometry_affordance_probe/batch_02/`
  - Second 12 seen-task demos, excluding all batch 1 demo IDs.
  - Two uniquely named initial-state views per demo: `view0_*.png` and `view1_*.png`.
  - Qwen2.5-VL geometry outputs and RoboPoint affordance outputs.

## Files To Open First

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
