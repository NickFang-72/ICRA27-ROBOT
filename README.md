# Geometry- and Affordance-Augmented X-ICM

This repository is Nicholas's working project for an ICRA 2027 robot manipulation idea built on top of **AGNOSTOS** and **X-ICM**.

The current direction is simple and modular: follow the X-ICM pipeline, but add two separate retrieval features before the in-context LLM prediction step:

1. `g_i`: object-centric **geometric features** from Qwen2.5-VL.
2. `a_i`: manipulation **affordance/contact features** from RoboPoint.

The goal is to test whether X-ICM retrieves better seen demonstrations when the original dynamics similarity is augmented with geometry and affordance similarity.

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

- `O_i` and `O_j` are current observation frames.
- `L_i` and `L_j` are language instructions.
- No future frames, after-states, unseen demonstrations, or unseen ground-truth actions are used.

The retrieval score can then be extended from X-ICM's dynamics retrieval:

```python
score(D_i, Q_j) =
    alpha * S_dyn(D_i, Q_j)
  + beta  * S_geo(g_i, g_j)
  + gamma * S_aff(a_i, a_j)
```

Weights should be tuned only on seen-task validation splits, not on the 23 AGNOSTOS unseen tasks.

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

1. Human-check the two pilot batches.
2. Fix incorrect geometry or affordance labels and update the normalizer rules.
3. Scale preprocessing to all 3,600 seen-task demonstrations.
4. Build separate retrieval ablations:
   - X-ICM baseline: dynamics only.
   - X-ICM + geometry.
   - X-ICM + affordance.
   - X-ICM + geometry + affordance.
5. Define similarity functions for `S_geo` and `S_aff`.
6. Tune `alpha`, `beta`, and `gamma` on seen-task validation splits only.
7. Add `g_i`, `a_i`, `g_j`, and `a_j` to the X-ICM in-context prompt.
8. Evaluate once on the held-out AGNOSTOS unseen tasks.
9. Record failure cases: wrong object, wrong contact point, wrong axis, collision, misalignment, or impossible affordance.

## Living Notes

This README is the main project idea document. Update it as the experiment changes:

- Prompt changes.
- Descriptor schema changes.
- Retrieval score changes.
- Human-checking findings.
- Benchmark results.
