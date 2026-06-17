# Current Test Status

Updated: 2026-06-17

## Completed Setup

- Created CAIR experiment under `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe`.
- Downloaded and extracted AGNOSTOS `seen_tasks` dataset on CAIR.
- Downloaded Qwen2.5-VL-7B-Instruct to `/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct`.
- Downloaded RoboPoint `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` to `/data/yf23/checkpoints/ICRA27-ROBOT/robopoint-v1-vicuna-v1.5-13b`.
- Used conda env `/data/yf23/conda/envs/icra27-robot`.

## Results

- `geometry_affordance_probe/batch_01`: 12 seen demos, 12 Qwen geometry files, 12 RoboPoint affordance files, 12 combined review files.
- `geometry_affordance_probe/batch_02`: 12 new seen demos excluding batch 1 IDs, 12 Qwen geometry files, 12 RoboPoint affordance files, 12 combined review files.
- Both batches use initial/current observation frames only. No future frames, after-states, or unseen-task demonstrations are used.

## Output Locations

- Local batch 1: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/batch_01`
- Local batch 2: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/batch_02`
- CAIR batch 1: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results`
- CAIR batch 2: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02`

## Important Caveat

RoboPoint used a local CLIP ViT-L/14 vision tower fallback on CAIR because the original `openai/clip-vit-large-patch14-336` PyTorch weights could not be fetched through Hugging Face SSL during the pilot. Treat RoboPoint affordance outputs as human-check candidates.
