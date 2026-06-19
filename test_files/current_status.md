# Current Test Status

Updated: 2026-06-19

## Completed Setup

- Created CAIR experiment under `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe`.
- Downloaded and extracted AGNOSTOS `seen_tasks` dataset on CAIR.
- Downloaded Qwen2.5-VL-7B-Instruct to `/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct`.
- Downloaded RoboPoint `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` to `/data/yf23/checkpoints/ICRA27-ROBOT/robopoint-v1-vicuna-v1.5-13b`.
- Used conda env `/data/yf23/conda/envs/icra27-robot`.

## Results

- `geometry_affordance_probe/batch_01`: 12 seen demos, 12 Qwen geometry files, 12 RoboPoint affordance files, 12 combined review files.
- `geometry_affordance_probe/batch_02`: 12 new seen demos excluding batch 1 IDs, 12 Qwen geometry files, 12 RoboPoint affordance files, 12 combined review files.
- Both descriptor batches use initial/current observation frames only. No future frames, after-states, or unseen-task demonstrations are used.
- The prompt-augmented X-ICM preparation now has a separate paper-faithful renderer path: retrieved seen demos are rendered as `Step k observation -> Step k 7D action` trajectories, while the unseen query still includes only the current observation and descriptors.
- Full seen-demo descriptor cache is complete on CAIR: `geometry=3600/3600`, `affordance=3600/3600`, `combined=3600/3600`.
- Added the ablation-only retrieval scorer for `alpha*S_dyn + beta*S_geo + gamma*S_aff`.
- Added the seen-task-only `alpha/beta/gamma` tuning utility.
- Refined the geometry/affordance prompt to use top-k retrieved seen demos and explicit "You will receive..." / "Your job is..." wording.

## Output Locations

- Local batch 1: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/batch_01`
- Local batch 2: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/batch_02`
- CAIR batch 1: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results`
- CAIR batch 2: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02`
- CAIR full cache: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl`
- Retrieval scorer: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/scripts/score_xicm_geometry_affordance_retrieval.py`
- Weight tuner: `/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/scripts/tune_geometry_affordance_weights.py`

## Important Caveat

RoboPoint used a local CLIP ViT-L/14 vision tower fallback on CAIR because the original `openai/clip-vit-large-patch14-336` PyTorch weights could not be fetched through Hugging Face SSL during the pilot. Treat RoboPoint affordance outputs as human-check candidates.
