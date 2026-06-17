# Geometry- and Affordance-Augmented X-ICM Plan

## Core Idea

Follow the original X-ICM pipeline, but add two separate retrieval signals before the in-context LLM prediction step:

1. **Geometric similarity** from object-centric geometric descriptors `g_i`, `g_j`.
2. **Affordance similarity** from manipulation affordance descriptors `a_i`, `a_j`.

The goal is to improve cross-task retrieval on AGNOSTOS by finding seen demonstrations that are not only dynamically similar, but also geometrically and functionally analogous to the unseen task.

This is intentionally lighter than building a new world model or effect encoder. The contribution is a modular retrieval and prompting augmentation to X-ICM.

## Model Roles

### Geometry-VLM

Recommended model: **Qwen2.5-VL**

Purpose:

- Extract shape, pose, orientation, object parts, openings, axes, and spatial relations.
- Produce structured geometric descriptors for seen demonstrations and unseen query observations.
- Support retrieval by physical scene similarity.

Why Qwen2.5-VL:

- Open-source and citable.
- Strong visual recognition and object localization.
- Can output structured descriptions, boxes, points, and spatial information.

Alternative geometry models:

- GPT-4.1 / GPT-5.5 vision as a high-quality teacher or annotation checker.
- Gemini 2.5 Pro or newer Gemini vision models for cross-model validation.
- InternVL3 / InternVL3.5 as an open-source ablation.
- Molmo when pointing or part localization is more important than long-form description.

### Affordance-VLM

Recommended model: **RoboPoint**

Purpose:

- Predict where the robot should act in the image.
- Extract contact points, grasp points, and action-relevant regions.
- Convert visual affordances into structured descriptors for retrieval and prompting.

Why RoboPoint:

- Explicitly designed for spatial affordance prediction for robotics.
- Takes image and language instruction as input.
- Outputs action-relevant keypoints, which can be converted into contact/grasp affordance labels.

Alternative affordance models:

- AffordanceLLM for affordance region or mask prediction.
- ManipLLM for robot-specific contact and end-effector direction reasoning.
- General VLMs such as Qwen2.5-VL, GPT, or Gemini for symbolic affordance labels, but these should be treated as weaker for precise robot contact grounding.

## Feature Definitions

For a seen demonstration `D_i`:

```python
g_i = GeometryVLM(O_i, L_i)
a_i = AffordanceVLM(O_i, L_i)
```

For an unseen AGNOSTOS query `Q_j`:

```python
g_j = GeometryVLM(O_j, L_j)
a_j = AffordanceVLM(O_j, L_j)
```

Where:

- `O_i` is the seen observation.
- `L_i` is the seen language instruction.
- `O_j` is the current unseen query observation.
- `L_j` is the unseen language instruction.

No unseen demonstrations, future observations, or ground-truth unseen trajectories are used.

## Geometric Descriptor `g`

```python
g = {
    "primary_shape": "box | cylinder | sphere | flat_panel | ring | rope | irregular | articulated_composite",
    "part_geometry": ["handle", "knob", "lid", "rim", "hinge", "opening", "slot", "thin_edge"],
    "size": "small | medium | large",
    "aspect_ratio": "compact | elongated | flat | tall | wide",
    "orientation": "horizontal | vertical | tilted | upright | upside_down",
    "front_face_direction": "+x | -x | +y | -y | unknown",
    "pose_relation": "on_top_of | inside | next_to | attached_to | inserted_into | stacked",
    "opening_geometry": "top_opening | front_opening | side_opening | slot | hole | none",
    "axis_geometry": "vertical_axis | horizontal_axis | sliding_axis | free_motion | none",
    "symmetry": "radial | bilateral | asymmetric | unknown",
    "clearance_geometry": "open_path | narrow_path | blocked_path | requires_lift"
}
```

Example:

```python
g_i = {
    "primary_shape": "flat_panel",
    "part_geometry": ["hinge", "handle"],
    "size": "medium",
    "aspect_ratio": "wide_flat",
    "orientation": "vertical",
    "front_face_direction": "toward_robot",
    "pose_relation": "attached_to_base",
    "opening_geometry": "front_opening",
    "axis_geometry": "vertical_axis",
    "symmetry": "bilateral",
    "clearance_geometry": "open_path"
}
```

## Affordance Descriptor `a`

```python
a = {
    "grasp_affordance": "handle_grasp | knob_grasp | rim_grasp | body_grasp | edge_grasp | pinch | none",
    "contact_affordance": "push_surface | pull_handle | lift_top | rotate_part | slide_part | insert_object",
    "motion_affordance": "push | pull | lift | rotate | slide | twist | insert | scoop | pour",
    "support_affordance": "can_support | can_contain | can_hang | can_stack | none",
    "containment_affordance": "open_container | closed_container | receptacle | slot | hole | none",
    "articulation_affordance": "hinge_open_close | drawer_slide | lid_remove | screw_twist | flexible_deform | none",
    "required_contact_region": "handle | knob | rim | top | side | front_face | free_end",
    "preferred_contact_point": [0.0, 0.0],
    "precision_requirement": "low | medium | high",
    "force_requirement": "low | medium | high",
    "failure_sensitive_property": "misalignment | collision | wrong_grasp | insufficient_lift | wrong_axis"
}
```

Example:

```python
a_i = {
    "grasp_affordance": "handle_grasp",
    "contact_affordance": "pull_handle",
    "motion_affordance": "rotate",
    "support_affordance": "none",
    "containment_affordance": "closed_container",
    "articulation_affordance": "hinge_open_close",
    "required_contact_region": "handle",
    "preferred_contact_point": [412, 238],
    "precision_requirement": "medium",
    "force_requirement": "medium",
    "failure_sensitive_property": "wrong_axis"
}
```

## Retrieval

X-ICM retrieves seen demonstrations using action-effect dynamics. This project adds two independent retrieval scores:

```python
score(D_i, Q_j) =
    alpha * S_dyn(D_i, Q_j)
  + beta  * S_geo(g_i, g_j)
  + gamma * S_aff(a_i, a_j)
```

Where:

- `S_dyn` is the original X-ICM dynamics similarity.
- `S_geo` compares geometric descriptors.
- `S_aff` compares affordance descriptors.

The weights `alpha`, `beta`, and `gamma` are tuned only on seen-task validation splits, not on the held-out AGNOSTOS unseen tasks.

## Prompt Format

The in-context prompt should keep geometry and affordance separate:

```text
Seen demonstration:
Task: close microwave

Geometric features g_i:
- rectangular box body
- hinged front door
- vertical hinge axis on left side
- handle on front face

Affordance features a_i:
- handle affords pull/push contact
- door affords rotation around hinge
- contact should occur away from hinge
- goal requires door alignment with frame

Key actions:
...

Unseen task:
Task: close laptop

Geometric features g_j:
- flat screen panel attached to base
- horizontal hinge axis along rear edge
- open angle between screen and base

Affordance features a_j:
- screen affords rotation around hinge
- contact should occur on upper panel
- goal requires panel aligned against base

Predict the intended next state or key 7D actions.
```

## Training and Preprocessing

Use only the AGNOSTOS/RLBench seen-task training set:

- 18 seen RLBench tasks.
- 200 demonstrations per seen task.
- 3,600 seen demonstrations total.

Preprocessing steps:

1. Run the Geometry-VLM on seen observations to cache `g_i`.
2. Run the Affordance-VLM on seen observations to cache `a_i`.
3. Optionally run GPT/Gemini/Claude as teacher checkers on a subset.
4. Human-verify a subset of annotations for schema consistency.
5. Tune retrieval weights using seen-task splits only.

Important leakage rule:

- Do not use unseen-task demonstrations.
- Do not use unseen-task future frames.
- Do not tune retrieval weights on the 23 held-out AGNOSTOS tasks.
- At benchmark time, only the current unseen observation and language instruction are used to compute `g_j` and `a_j`.

## Benchmark-Time Inference

For each unseen AGNOSTOS task:

1. Receive current observation `O_j` and language instruction `L_j`.
2. Extract `g_j` with the Geometry-VLM.
3. Extract `a_j` with the Affordance-VLM.
4. Retrieve top-k seen demonstrations using combined dynamics, geometry, and affordance similarity.
5. Build the X-ICM in-context prompt with retrieved key actions plus `g_i`, `a_i`, `g_j`, and `a_j`.
6. Ask the frozen LLM to infer the intended next state, subgoal, or 7D key action sequence.
7. Use the existing controller/action model to execute or evaluate the predicted actions.

## Experimental Setup

### Main Baselines

1. **X-ICM baseline**
   - Dynamics retrieval only.
   - Original in-context prompt.

2. **X-ICM + Geometry**
   - Dynamics retrieval plus `S_geo(g_i, g_j)`.
   - Prompt includes `g_i` and `g_j`.

3. **X-ICM + Affordance**
   - Dynamics retrieval plus `S_aff(a_i, a_j)`.
   - Prompt includes `a_i` and `a_j`.

4. **Full model**
   - Dynamics, geometry, and affordance retrieval.
   - Prompt includes both `g_i`, `a_i`, `g_j`, and `a_j`.

### Model Ablations

Geometry extractor ablations:

- Qwen2.5-VL.
- InternVL3 / InternVL3.5.
- GPT/Gemini teacher labels.

Affordance extractor ablations:

- RoboPoint keypoints.
- AffordanceLLM masks.
- General VLM symbolic affordance labels.

Retrieval ablations:

- Similarity using only categorical fields.
- Similarity using text embeddings of descriptors.
- Similarity using weighted field-level matching.
- Top-1, top-3, and top-5 retrieved demonstrations.

Prompt ablations:

- Retrieval uses geometry/affordance, but prompt does not include them.
- Prompt includes geometry only.
- Prompt includes affordance only.
- Prompt includes both.

This separates whether the improvement comes from better retrieval, better LLM reasoning, or both.

## Evaluation

Primary metric:

- AGNOSTOS unseen-task success rate.

Secondary metrics:

- Level-1 vs Level-2 unseen success rate.
- Retrieval quality: whether retrieved seen demos share shape, axis, contact, or motion affordances with the unseen query.
- Key action prediction error.
- Contact-region accuracy when ground truth or human annotation is available.
- Failure category: wrong object, wrong contact point, wrong axis, collision, insufficient lift, misalignment.

Expected outcome:

- Geometry should help when unseen tasks share shape or kinematic structure with seen tasks.
- Affordances should help when the important analogy is action-centric rather than object-name-centric.
- The full model should be strongest on Level-1 AGNOSTOS tasks and may provide partial gains on some Level-2 tasks with reusable affordances.

## Main Risks

1. Geometry descriptors may be too vague unless the schema is strict.
2. Affordance predictions may be image-space accurate but not physically executable.
3. General VLMs may hallucinate hidden articulation or contact properties.
4. Improvements may come from better prompting rather than better retrieval, so retrieval-only and prompt-only ablations are necessary.
5. Level-2 AGNOSTOS tasks may still fail when no seen task shares useful geometry or affordances.

## Citeable Starting Points

- X-ICM / AGNOSTOS: cross-task in-context manipulation benchmark and pipeline.
- Qwen2.5-VL: geometry and structured visual extraction.
- RoboPoint: spatial affordance prediction for robotics.
- AffordanceLLM: grounding affordances from vision-language models.
- ManipLLM: embodied multimodal model for object-centric manipulation.
