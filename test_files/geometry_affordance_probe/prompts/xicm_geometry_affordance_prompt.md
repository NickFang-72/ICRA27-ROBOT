# X-ICM Geometry and Affordance Prompt Template

This prompt is a preparation artifact for the geometry/affordance X-ICM ablation. It keeps the original X-ICM action-prediction task, but adds explicit geometry and affordance descriptors for each retrieved seen demonstration and for the current unseen query.

Important correction: the seen demonstrations should be represented as key-action trajectories. Each seen demo should include the object-location observation at each key action step, followed by the 7D action for that step. Do not collapse a seen demonstration into only the first observation plus one action list.

Use this for the prompt-augmented condition after the original X-ICM baseline is reproduced. The original baseline prompt should remain unchanged for the baseline run.

## System Prompt

```text
You are a Franka Panda robot with a parallel gripper.

You will receive the top-k retrieved in-context demonstrations from seen robot manipulation tasks. Each seen demonstration contains:
- a task instruction,
- a sequence of key action steps,
- for each key action step, an X-ICM observation text summarizing object names and discretized 3D positions,
- for each key action step, the corresponding next 7D action,
- a geometry description g_i covering task-relevant object shape, parts, openings, axes, and alignment constraints,
- an affordance description a_i covering grasp/contact regions, motion affordances, and likely failure-sensitive properties.

You will then receive one unseen query. The unseen query contains only the current/initial observation, task instruction, geometry description g_j, and affordance description a_j. It does not contain future observations, after-states, ground-truth actions, or unseen demonstrations.

Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the top-k retrieved seen demonstrations. Use action trends, geometry, and affordances to choose the best physical analogy: object shape, required contact region, articulation axis, insertion/containment/alignment needs, and motion type. Do not invent objects, future states, or actions that are not supported by the current unseen observation and instruction.

Return only a Python-style list of 7D action lists. Do not output explanations, labels, markdown, or any other text.
```

## User Prompt Template

```text
You will receive {{top_k}} top-k retrieved seen demonstrations from the AGNOSTOS seen-task training set. Use all of them as in-context examples for the current unseen query.

Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the retrieved seen demonstrations using action trends, geometry, and affordances.

Important rules:
- Each seen demonstration includes per-key-action observations paired with the corresponding 7D action.
- Each seen demonstration includes geometry description g_i and affordance description a_i.
- The unseen query includes only the current/initial observation, task instruction, geometry description g_j, and affordance description a_j.
- Do not use unseen demonstrations, unseen future frames, unseen ground-truth actions, or after-states.
- If geometry and affordance conflict with a seen demo's action trend, prioritize the current unseen observation and task instruction.
- Preserve the X-ICM output format: only a list of 7D action lists, such as [[x, y, z, roll, pitch, yaw, gripper], ...].

Seen demonstration 1:
Task instruction:
{{seen_1_task_instruction}}

Geometry features g_i:
{{seen_1_geometry_features}}

Affordance features a_i:
{{seen_1_affordance_features}}

Key observation-action trajectory:
Step 1 observation:
{{seen_1_step_1_observation}}
Step 1 7D action:
{{seen_1_step_1_action}}

Step 2 observation:
{{seen_1_step_2_observation}}
Step 2 7D action:
{{seen_1_step_2_action}}

...

Seen demonstration 2:
Task instruction:
{{seen_2_task_instruction}}

Geometry features g_i:
{{seen_2_geometry_features}}

Affordance features a_i:
{{seen_2_affordance_features}}

Key observation-action trajectory:
Step 1 observation:
{{seen_2_step_1_observation}}
Step 1 7D action:
{{seen_2_step_1_action}}

Step 2 observation:
{{seen_2_step_2_observation}}
Step 2 7D action:
{{seen_2_step_2_action}}

...

Seen demonstration N:
Task instruction:
{{seen_n_task_instruction}}

Geometry features g_i:
{{seen_n_geometry_features}}

Affordance features a_i:
{{seen_n_affordance_features}}

Key observation-action trajectory:
{{seen_n_observation_action_steps}}

Unseen task:
Task instruction:
{{query_task_instruction}}

Current observation:
{{query_observation}}

Geometry features g_j:
{{query_geometry_features}}

Affordance features a_j:
{{query_affordance_features}}

Predict the key 7D action sequence for the unseen task. Return only a Python-style list of 7D action lists:
```

The `...` markers above are template notation only. The renderer should loop over the actual retrieved demonstrations and the actual keypoint steps for each demonstration.

## Compact Implementation Shape

The code path can render the repeated demonstration blocks programmatically:

```text
Seen demonstration {{rank}}:
Task instruction:
{{task_instruction}}

Geometry features g_i:
- manipulated_object: {{manipulated_object}}
- key_features: {{geometry_key_features}}
- part_geometry: {{part_geometry}}
- opening_geometry: {{opening_geometry}}
- axis_geometry: {{axis_geometry}}
- clearance_geometry: {{clearance_geometry}}
- task_relevant_geometric_cues: {{task_relevant_geometric_cues}}

Affordance features a_i:
- grasp_affordance: {{grasp_affordance}}
- contact_affordance: {{contact_affordance}}
- motion_affordance: {{motion_affordance}}
- containment_affordance: {{containment_affordance}}
- articulation_affordance: {{articulation_affordance}}
- required_contact_region: {{required_contact_region}}
- preferred_contact_points: {{preferred_contact_points}}
- precision_requirement: {{precision_requirement}}
- failure_sensitive_property: {{failure_sensitive_property}}

Key observation-action trajectory:
Step 1 observation:
{{step_1_observation_text}}
Step 1 7D action:
{{step_1_action}}

Step 2 observation:
{{step_2_observation_text}}
Step 2 7D action:
{{step_2_action}}

...
```

The unseen query should use the same geometry and affordance fields, but must include only the current observation. It must not include future observations, key actions, after-states, or ground-truth action labels.

Implementation requirements:
- When building the seen-demo prompt blocks, compute or load `form_obs(...)` at each demonstration keypoint state, not only at the initial state.
- Treat X-ICM observations as custom text emitted by the dataset renderer, not as valid Python literals. The required parseable output is only the final list of 7D actions.
- Use `unknown`, `none`, or `[]` for missing descriptor fields instead of changing field order across demonstrations.
- The geometry and affordance descriptors can stay demo-level unless we later decide to compute step-level descriptors.
- Keep this prompt-augmented renderer separate from the vanilla X-ICM baseline prompt path.

## Example Mini Prompt

```text
Seen demonstration 1:
Task instruction:
close the red jar

Geometry features g_i:
- manipulated_object: jar
- key_features: [jar, round, cylindrical, hollow, lid, rim, top_opening]
- opening_geometry: top
- axis_geometry: vertical
- task_relevant_geometric_cues: [lid, rim, vertical_axis]

Affordance features a_i:
- grasp_affordance: rim_grasp
- contact_affordance: rotate_part
- motion_affordance: twist
- articulation_affordance: screw_twist
- required_contact_region: lid_or_rim
- precision_requirement: medium
- failure_sensitive_property: wrong_axis

Key observation-action trajectory:
Step 1 observation:
['instruction': close the red jar, {'lid': [42, 51, 73], 'jar': [43, 51, 68]}]
Step 1 7D action:
[42, 51, 73, 0, 39, 0, 1]

Step 2 observation:
['instruction': close the red jar, {'lid': [43, 51, 71], 'jar': [43, 51, 68]}]
Step 2 7D action:
[43, 51, 70, 0, 39, 12, 0]

Seen demonstration 2:
Task instruction:
turn left tap

Geometry features g_i:
- manipulated_object: tap
- key_features: [tap, handle, knob, rotational_axis, cylindrical]
- opening_geometry: none
- axis_geometry: vertical
- task_relevant_geometric_cues: [handle, knob, rotation_axis]

Affordance features a_i:
- grasp_affordance: knob_grasp
- contact_affordance: rotate_part
- motion_affordance: rotate
- articulation_affordance: screw_twist
- required_contact_region: tap_handle_or_knob
- precision_requirement: medium
- failure_sensitive_property: wrong_axis

Key observation-action trajectory:
Step 1 observation:
['instruction': turn left tap, {'tap': [36, 48, 65]}]
Step 1 7D action:
[36, 48, 66, 0, 39, 0, 1]

Step 2 observation:
['instruction': turn left tap, {'tap': [36, 48, 65]}]
Step 2 7D action:
[36, 48, 66, 0, 39, 18, 0]

Unseen task:
Task instruction:
turn oven on

Current observation:
['instruction': turn oven on, {'oven_knob': [39, 50, 70]}]

Geometry features g_j:
- manipulated_object: oven_knob
- key_features: [knob, round, rotational_axis, front_face]
- opening_geometry: none
- axis_geometry: front_normal
- task_relevant_geometric_cues: [knob, rotation_axis]

Affordance features a_j:
- grasp_affordance: knob_grasp
- contact_affordance: rotate_part
- motion_affordance: rotate
- articulation_affordance: screw_twist
- required_contact_region: oven_knob
- precision_requirement: medium
- failure_sensitive_property: wrong_axis

Predict the key 7D action sequence for the unseen task. Return only a Python-style list of 7D action lists:
```

## Wiring Notes

- Use this prompt only for the `X-ICM + geometry + affordance` ablation.
- This template expects per-key-action seen observations. Use `scripts/prepare_xicm_key_action_trajectories.py` to build payloads from retrieved seen episode paths, then `scripts/render_xicm_geometry_affordance_prompt.py` to render/validate the final prompt.
- The preparation script should run after retrieval chooses top-k seen demonstrations. Retrieval may use cached demo-level `g_i`/`a_i`, but prompt rendering must still include every retrieved demo's `Step k observation` followed by `Step k 7D action`.
- For `X-ICM + geometry`, omit the affordance blocks but keep the same output rules.
- For `X-ICM + affordance`, omit the geometry blocks but keep the same output rules.
- For geometry-only or affordance-only ablations, also use a matching system prompt that does not claim both feature types are present.
- Tune retrieval weights on seen-task validation only.
- Run the held-out AGNOSTOS unseen tasks once after prompt and retrieval choices are frozen.
