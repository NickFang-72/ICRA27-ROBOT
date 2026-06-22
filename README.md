# Geometry- and Affordance-Augmented X-ICM

This repository is Nicholas's working project for an ICRA 2027 robot manipulation idea built on top of **AGNOSTOS** and **X-ICM**.

The current direction is simple and modular: follow the X-ICM pipeline, but add two separate retrieval features before the in-context LLM prediction step:

1. `g_i`: object-centric **geometric features** from Qwen2.5-VL.
2. `a_i`: manipulation **affordance/contact features** from RoboPoint.

The goal is to test whether X-ICM retrieves better seen demonstrations when the original dynamics similarity is augmented with geometry and affordance similarity.

## Trial Versions

This project now has four geometry/affordance trial lines. They should be kept conceptually separate even when the implementation shares `X-ICM/form_icl_demonstrations_crosstask_ranking.py`.

| Version | Ranking method | Main idea | Current status |
|---|---|---|---|
| v1 | `lang_vis.out.geo_aff` | Add Qwen geometry `g_i/g_j` and RoboPoint affordance `a_i/a_j` to retrieval and prompt context. | Completed ablation, average `20.87`; underperformed the original rerun. |
| v2 | `lang_vis.out.geo_aff_v2` | Keep dynamics similarity as the anchor, add a precise interaction signature, transfer penalties, and prompt-visible attention bias. | Compact K sweep complete for `k=6,8,10`; best average tied at `21.74` for `k=6` and `k=8`. |
| v3 | `lang_vis.out.geo_aff_v3` | Use contact-mode/mechanical compatibility, stronger conflict penalties, retrieval diversity caps, and explicit contact-mode prompt guidance. | Completed `k=6`, average `20.00`; failed to improve because the prompt still asked the model to turn many descriptor-heavy demos directly into one 7D action chain. |
| v4 | `lang_vis.out.geo_aff_v4` | Test a semantic bottleneck: Stage 1 converts descriptor-heavy context into a simple manipulation intent, then Stage 2 uses that intent plus seen observation/action trajectories to predict 7D actions. | Prepared for `k=6` CAIR benchmark. |

The important difference between v2, v3, and v4 is where the extra structure enters. V2 uses descriptors as conservative retrieval tie-breakers. V3 tries to retrieve physically compatible demonstrations more aggressively, but still gives the LLM one large descriptor-and-trajectory prompt. V4 keeps the retrieval idea but splits prompting into two calls so the model first states the manipulation intent in simple words, then predicts the 7D action sequence from that intent and the retrieved action examples.

Directory organization is intentionally postponed as of 2026-06-21. Several v1/v2-related files are already edited in the working tree, and a local v2 watcher was active during this documentation pass. To avoid disturbing running or dirty trial state, do not move, rename, or reorganize trial scripts/logs until the worktree is clean and no watcher/evaluator is active.

## References

- **AGNOSTOS / X-ICM paper**: [Exploring the Limits of Vision-Language-Action Manipulation in Cross-task Generalization](https://arxiv.org/pdf/2505.15660)
- **AGNOSTOS project page**: [jiaming-zhou.github.io/AGNOSTOS](https://jiaming-zhou.github.io/AGNOSTOS/)
- **X-ICM GitHub repository**: [jiaming-zhou/X-ICM](https://github.com/jiaming-zhou/X-ICM)
- **AGNOSTOS dataset**: [Hugging Face dataset](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)
- **X-ICM model**: [Hugging Face model](https://huggingface.co/Jiaming2472/X-ICM)
- **Local X-ICM checkout**: `X-ICM/`

## Background

AGNOSTOS is a benchmark for cross-task zero-shot manipulation generalization. It separates RLBench tasks into:

- 18 seen training tasks.
- 23 held-out unseen test tasks.

X-ICM improves cross-task manipulation by retrieving relevant seen demonstrations and placing them into an in-context prompt for an LLM. The original X-ICM pipeline retrieves demonstrations primarily through dynamics/action-effect similarity, then asks the LLM to infer key actions for a new task.

Our hypothesis is that dynamics similarity alone misses useful physical analogies. For example:

- `close jar`, `turn tap`, and `screw in light bulb` share rotational/axis-sensitive structure.
- `open drawer`, `put item in drawer`, and `put groceries in cupboard` share containment/opening structure.
- `insert onto square peg`, `place shape in sorter`, and `put money in safe` share alignment/slot/hole structure.

The proposed addition is to make those physical analogies explicit through geometry and affordance descriptors.

## Core Idea

For each seen demonstration `D_i`, compute:

```python
g_i = GeometryVLM(O_i, L_i)
a_i = AffordanceVLM(O_i, L_i)
```

For each benchmark-time unseen query `Q_j`, compute:

```python
g_j = GeometryVLM(O_j, L_j)
a_j = AffordanceVLM(O_j, L_j)
```

Where:

- `O_i` and `O_j` are current observation frames used to compute descriptor caches.
- `L_i` and `L_j` are language instructions.
- No future frames, after-states, unseen demonstrations, or unseen ground-truth actions are used.

For prompt-augmented X-ICM, the retrieved seen demonstrations are then rendered in the original paper style as per-key-action trajectories:

```text
Step 1 observation -> Step 1 7D action
Step 2 observation -> Step 2 7D action
...
```

The unseen query remains only the current observation/instruction plus `g_j` and `a_j`; it never includes unseen future frames or actions.

The retrieval score can then be extended from X-ICM's dynamics retrieval:

```python
score(D_i, Q_j) =
    alpha * S_dyn(D_i, Q_j)
  + beta  * S_geo(g_i, g_j)
  + gamma * S_aff(a_i, a_j)
```

Weights should be tuned only on seen-task validation splits, not on the 23 AGNOSTOS unseen tasks.

The ablation scorer is separate from the vanilla X-ICM baseline:

- `test_files/geometry_affordance_probe/scripts/score_xicm_geometry_affordance_retrieval.py`
- `test_files/geometry_affordance_probe/scripts/tune_geometry_affordance_weights.py`

## Geometry Descriptor

Qwen2.5-VL is used only for geometry. The descriptor is intended to capture object shape, parts, orientation, openings, axes, and compact retrieval features.

Example fields:

```json
{
  "primary_shape": "jar",
  "part_geometry": ["lid", "body"],
  "size": "small",
  "aspect_ratio": "wide",
  "orientation": "upright",
  "opening_geometry": "top",
  "axis_geometry": "vertical",
  "symmetry": "bilateral",
  "key_features": ["jar", "round", "cylindrical", "hollow", "lid", "rim", "top_opening"]
}
```

Useful `key_features` include:

- Shape: `round`, `rectangular`, `cylindrical`, `flat`, `elongated`, `solid`, `hollow`.
- Parts: `handle`, `knob`, `lid`, `rim`, `slot`, `hole`, `hinge`, `thin_edge`.
- Structure: `open_container`, `front_opening`, `top_opening`, `sliding_axis`, `rotational_axis`.
- Task sensitivity: `alignment_sensitive`, `matching_geometry`, `target_region`.

Prompt implementation:

- `test_files/geometry_affordance_probe/scripts/run_qwen_geometry.py`
- Prompt constant: `GEOMETRY_PROMPT`

The current prompt asks Qwen to return only valid JSON and includes this schema:

```json
{
  "primary_shape": "string",
  "part_geometry": ["string"],
  "size": "small|medium|large|unknown",
  "aspect_ratio": "string",
  "orientation": "string",
  "front_face_direction": "string",
  "pose_relation": "string",
  "opening_geometry": "string",
  "axis_geometry": "string",
  "symmetry": "string",
  "clearance_geometry": "string",
  "task_relevant_geometric_cues": ["string"],
  "uncertain_fields": ["string"]
}
```

After Qwen runs, `normalize_geometry_affordance_outputs.py` adds `manipulated_object` and normalized `key_features`.

## Affordance Descriptor

RoboPoint is used only for affordance/contact prediction. Its native output is spatial: normalized image keypoints for where the robot should interact.

Example fields:

```json
{
  "grasp_affordance": "rim_grasp",
  "contact_affordance": "rotate_part",
  "motion_affordance": "twist",
  "required_contact_region": "lid_or_rim",
  "preferred_contact_points": [[0.498, 0.6], [0.534, 0.596]],
  "precision_requirement": "medium",
  "failure_sensitive_property": "wrong_axis"
}
```

Prompt implementation:

- `test_files/geometry_affordance_probe/scripts/run_robopoint_affordance.py`
- Prompt constant: `QUESTION_TEMPLATE`

The current RoboPoint prompt is:

```text
<image>
Task instruction: {task}
For robot manipulation, identify the best visible contact or grasp affordance points for accomplishing this task. Return several normalized image keypoints as a list of tuples [(x1, y1), ...] where x and y are between 0 and 1. After the list, add a short phrase naming the contact region and motion affordance.
```

After RoboPoint runs, `normalize_geometry_affordance_outputs.py` parses the contact points and normalizes symbolic affordance labels such as:

- `handle_grasp`
- `rim_grasp`
- `body_grasp`
- `push_surface`
- `pull_handle`
- `rotate_part`
- `insert_object`
- `drawer_slide`
- `screw_twist`

## Current Experiment State

The current pilot has two seen-task review batches:

- `test_files/geometry_affordance_probe/batch_01`
- `test_files/geometry_affordance_probe/batch_02`

Each batch contains:

- 12 seen-task demos.
- Qwen geometry output per demo.
- RoboPoint affordance/contact output per demo.
- A per-demo `combined_review.json`.
- A batch-level `review_index.md`.
- A batch-level `review_bundle.jsonl`.

Open these first:

- `test_files/geometry_affordance_probe/batch_01/review_index.md`
- `test_files/geometry_affordance_probe/batch_02/review_index.md`

Batch 2 excludes all batch 1 demo IDs. Batch 2 also preserves two uniquely named initial-state views per demo.

Prompt preparation scripts live in:

- `test_files/geometry_affordance_probe/scripts/prepare_xicm_key_action_trajectories.py`
- `test_files/geometry_affordance_probe/scripts/render_xicm_geometry_affordance_prompt.py`

These are separate from the vanilla X-ICM baseline path.

The full seen-demo cache is complete on CAIR:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl
```

It contains normalized geometry/affordance rows for all 3,600 seen demonstrations.

The compact v2 K sweep is complete on CAIR under:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations
```

The current v3 status is an active CAIR run. The read-only check on 2026-06-21 found `progress_v3.json` with status `running`, `0/23` completed task CSVs, and log path:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations/logs/geometry_affordance_v3_k6_20260621_232651.log
```

## CAIR Setup

The CAIR experiment root is:

```text
/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe
```

The conda environment used for the pilot is:

```text
/data/yf23/conda/envs/icra27-robot
```

Important model/data locations:

```text
/data/yf23/datasets/ICRA27-ROBOT/seen_tasks
/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct
/data/yf23/checkpoints/ICRA27-ROBOT/robopoint-v1-vicuna-v1.5-13b
```

RoboPoint caveat: the pilot used a local CLIP ViT-L/14 vision tower fallback because the original `openai/clip-vit-large-patch14-336` PyTorch weights could not be fetched through Hugging Face SSL on CAIR. Treat RoboPoint outputs as human-check candidates.

### X-ICM Build Status - 2026-06-17

X-ICM is cloned on CAIR at:

```text
/data/yf23/projects/ICRA27-ROBOT/X-ICM
```

The active X-ICM conda environment is:

```text
/data/yf23/conda/envs/zero-shot
```

This environment currently verifies:

```text
torch 2.6.0+cu124, CUDA 12.4, torch.cuda.is_available() == True
PyRep 4.1.0.3
RLBench 1.2.0
YARR 0.1
```

CoppeliaSim 4.1 is linked from an existing CAIR install:

```text
/data/yf23/projects/ICRA27-ROBOT/X-ICM/CoppeliaSim -> /data/yf23/instant_policy_cair/sim/CoppeliaSim
```

The X-ICM data directory currently has the 18 seen tasks linked:

```text
/data/yf23/projects/ICRA27-ROBOT/X-ICM/data/seen_tasks -> /data/yf23/datasets/ICRA27-ROBOT/seen_tasks
```

The full unseen-task archive and dynamics-diffusion checkpoint are not complete yet because CAIR resets connections to Hugging Face CAS/Xet hosts. The fallback is the resumable local relay:

```text
test_files/geometry_affordance_probe/cair_setup_scripts/stream_archives_to_cair_from_local.sh
```

The relay streams byte ranges from the Mac to CAIR over SSH, then concatenates/extracts:

```text
/data/yf23/datasets/ICRA27-ROBOT/unseen_tasks.tar
/data/yf23/checkpoints/ICRA27-ROBOT/dynamics_diffusion.tar
```

It is slow on the current network path, so the preferred next step is either to run the relay overnight or obtain a CAIR-accessible mirror/direct transfer for the two archives.

## How This Extends X-ICM

The intended X-ICM prompt should keep dynamics, geometry, and affordance separate:

```text
Seen demonstration:
Task: close jar

Geometric features g_i:
- jar
- round
- cylindrical
- hollow
- lid
- rim
- top_opening

Affordance features a_i:
- grasp: rim_grasp
- contact: rotate_part
- motion: twist
- contact region: lid_or_rim

Key actions:
...

Unseen task:
Task: turn tap

Geometric features g_j:
- tap
- handle
- knob
- rotational_axis
- cylindrical

Affordance features a_j:
- grasp: knob_grasp
- contact: rotate_part
- motion: rotate
- contact region: tap_handle_or_knob

Predict the intended next state or key 7D actions.
```

The research question is whether adding `g_i/a_i` and `g_j/a_j` improves retrieval and in-context prediction on cross-task manipulation, especially for tasks whose language differs but whose physical structure is similar.

The prepared combined prompt template lives at:

```text
test_files/geometry_affordance_probe/prompts/xicm_geometry_affordance_prompt.md
```

## Next Steps

1. Preserve the completed v1/v2 artifacts before reorganizing anything.
2. Launch v3 only after confirming no evaluator/watch process is using the same files.
3. Pull v3 results and regenerate the paper-style tables once all 23 strict final scores exist.
4. Compare v3 against the v2 `k=6` and `k=8` tie, not only against the original v1 descriptor ablation.
5. Inspect whether v3 improves the four all-zero v2 tasks: `put_toilet_roll_on_stand`, `put_books_on_bookshelf`, `basketball_in_hoop`, and `scoop_with_spatula`.
6. Only after the working tree is clean, reorganize trial artifacts into stable v1/v2/v3 folders or add symlink-style indexes that do not break existing scripts.

## Living Notes

This README is the main project idea document. Update it as the experiment changes:

- Prompt changes.
- Descriptor schema changes.
- Retrieval score changes.
- Human-checking findings.
- Benchmark results.
