# Geometry/Affordance Probe

This folder contains scripts and review outputs for the X-ICM-style geometry and affordance retrieval pilot.

## Structure

- `batch_01/`: first 12 seen-task demos.
- `batch_02/`: second 12 seen-task demos, excluding batch 1 demo IDs.
- `scripts/`: CAIR runtime scripts for sampling, Qwen geometry extraction, RoboPoint affordance extraction, normalization, and review index generation.
- `cair_setup_scripts/`: setup/download runner scripts used on CAIR.

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

## Leakage Rule

The batches are sampled from seen-task data only. They do not include unseen AGNOSTOS demonstrations, future frames, or after-state frames.
