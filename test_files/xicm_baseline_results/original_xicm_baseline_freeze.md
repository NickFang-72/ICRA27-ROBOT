# Original X-ICM Baseline Freeze

Frozen on: 2026-06-18 19:56:46 EDT

Freeze ID: `original-xicm-seed0-20260618`

## Status

This is the fixed comparison point for geometry/affordance ablations. Do not change the baseline prompt, retrieval mode, seed, task set, or scoring artifacts when comparing later runs.

## Result Summary

- Condition: original X-ICM baseline.
- Prompt: original X-ICM paper-style prompt, with no geometry or affordance descriptors.
- Model: `Qwen2.5.7B.instruct`.
- Seed: `0`.
- Episodes per task: `25`.
- In-context demos: `18`.
- Retrieval mode: `lang_vis.out`.
- Seen task folders linked: `18`.
- Unseen task folders linked: `23`.
- Completed unseen task scores: `23/23`.
- Mean success score: `22.087`.
- Min score: `0`.
- Max score: `96`.

## Re-Run Recipe

The baseline launch was driven by the local CAIR launcher scripts with these effective defaults:

```bash
cd /data/yf23/projects/ICRA27-ROBOT/X-ICM
source /data/yf23/miniconda3/etc/profile.d/conda.sh
conda activate /data/yf23/conda/envs/zero-shot
SEEDS=0 \
EPISODES=25 \
GPU_IDS=0 \
NUM_ICLS=18 \
RANKING_METHOD=lang_vis.out \
MODELNAME=Qwen2.5.7B.instruct \
./run_baseline_xicm_after_data_ready.sh
```

Local provenance for the launch command:

- `test_files/geometry_affordance_probe/cair_setup_scripts/run_xicm_goal_until_ready.sh`
- `test_files/geometry_affordance_probe/cair_setup_scripts/run_agnostos_xicm_download_loop.sh`

Note: `run_baseline_xicm_after_data_ready.sh` is a CAIR-side helper and is not present in this local workspace at freeze time.

## Prompt Fingerprint

Prompt source:

- `X-ICM/crosstask_icl_agent.py`

System prompt used by the original baseline:

```text
You are a Franka Panda robot with a parallel gripper. We provide you with some demos from some seen tasks, in the format of [task_instruction, observation]>[ 7-dim action_1, 7-dim action_2, ..., 7-dim action_N ]. Then you will receive an unseen task instruction with a new observation, and you need to output a list of 7-dim actions that match the trends in the demos. Do not output anything else.
```

Baseline prompt rule: no `geometry_g_i`, `affordance_a_i`, `geometry_g_j`, `affordance_a_j`, or geometry/affordance wording is part of this condition.

## Code Provenance

- Local branch at freeze: `main`.
- Local repository HEAD at freeze: `14feb4b49ca0973fee76b242853aad23dcaaa4f2`.
- `X-ICM/` baseline source has no local diff relative to that HEAD at freeze time.
- Worktree note: the wider workspace has unrelated uncommitted geometry/affordance prep docs/scripts and baseline result artifacts. Those are not part of the vanilla baseline condition.

## Artifact Paths

Local copied artifacts:

- `test_files/xicm_baseline_results/seed0_original_prompt_summary.md`
- `test_files/xicm_baseline_results/seed0_original_prompt_scores.csv`

CAIR artifacts:

- Main log: `/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/baseline_xicm_original_prompt/run_20260618_210948.log`
- Result directory: `/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test`
- Score CSV: `/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/baseline_xicm_original_prompt/seed0_original_prompt_scores.csv`

## Artifact Checksums

```text
61a51d8c6f53d8bf4c6313cd8983a7f569df652563f39738247494f891b1b91f  test_files/xicm_baseline_results/seed0_original_prompt_summary.md
b6dde63c840978bf88e6938381dae58aad46fbba155fa85bb5bc3f47cdc22d72  test_files/xicm_baseline_results/seed0_original_prompt_scores.csv
f114e0468366d1f0ef2d9e40f1a28d34cfede94325b338acb1439a01a81595aa  X-ICM/crosstask_icl_agent.py
```

## Scores

| Task | Score |
| --- | ---: |
| put_toilet_roll_on_stand | 0 |
| put_knife_on_chopping_board | 20 |
| close_fridge | 20 |
| close_microwave | 48 |
| close_laptop_lid | 40 |
| phone_on_base | 56 |
| toilet_seat_down | 60 |
| lamp_off | 60 |
| lamp_on | 40 |
| put_books_on_bookshelf | 0 |
| put_umbrella_in_umbrella_stand | 0 |
| open_grill | 4 |
| put_rubbish_in_bin | 16 |
| take_usb_out_of_computer | 96 |
| take_lid_off_saucepan | 16 |
| take_plate_off_colored_dish_rack | 0 |
| basketball_in_hoop | 4 |
| scoop_with_spatula | 0 |
| straighten_rope | 8 |
| turn_oven_on | 16 |
| beat_the_buzz | 0 |
| water_plants | 0 |
| unplug_charger | 4 |

## Next Comparison Rule

Every later ablation should report:

- Delta against mean score `22.087`.
- Same 23 unseen tasks.
- Same seed/episode setup unless explicitly labeled as a different run.
- Whether retrieval changed, prompt changed, or both changed.
