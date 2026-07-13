# Phase-Anchor Pipeline

This project copy separates the new open-loop phase-anchor pipeline from the
current double-retrieval checkpoint in `../X-ICM`.

Design rule:

```text
Query decides WHAT. Anchors provide WHERE. Retrieval helps HOW.
The compiler produces open-loop 7D actions.
```

## Pipeline

1. Query scene extraction
   - instruction
   - front/overhead RGB
   - masks/object names
   - point cloud
   - geometry descriptor `g_j`
   - goal descriptor `h_j`
   - role-labeled scene anchors `c_j`

2. Query-only phase interpreter
   - file: `phase_interpreter.py`
   - input: only the current query scene and descriptors
   - output: task family, success condition, and phase plan
   - retrieved scenes are not allowed in this module

3. Phase normalization and anchor binding
   - file: `phase_anchor_pipeline/phase_normalizer.py`
   - canonicalizes generic families such as `insertion`
   - binds phases to scene anchors such as `manipulated_object_contact` and
     `goal_region`
   - inserts compiler-ready phases such as `pregrasp`, `grasp`, `lift`,
     `move_above_goal`, `insert/lower`, and `release`

4. Double retrieval as execution prior
   - still comes from the existing X-ICM retrieval path
   - retrieval does not enter the phase interpreter
   - retrieval may later populate `execution_prior`, for example:
     `default_rotation`, `phase_rotations`, `safe_lift_voxels`,
     `hinge_push_offset`, or `phase_motion_offsets`

5. Primitive template compiler
   - file: `phase_anchor_pipeline/primitive_compiler.py`
   - converts anchor-bound phases into open-loop 7D keyframes
   - output format: `[x, y, z, roll, pitch, yaw, gripper]`

6. Action verifier
   - file: `phase_anchor_pipeline/action_verifier.py`
   - checks gripper rhythm, lift clearance, release-at-goal, insertion/hinge
     constraints, and missing anchors

7. Orchestrator
   - file: `phase_anchor_pipeline/pipeline.py`
   - runs normalize/bind -> compile -> verify

No closed loop is used in this design.

## Module Review Command

Run this on a folder created by `scripts/generate_phase_interpreter_review.py`:

```bash
python3 scripts/generate_phase_anchor_pipeline_review.py \
  --phase-review-root /path/to/phase_interpreter_query_only_review \
  --output-root /path/to/results \
  --name phase_anchor_pipeline_compile
```

Each output task folder contains:

```text
05_normalized_anchored_phase_plan.json
05_normalized_anchored_phase_plan.md
06_compiled_open_loop_actions.json
06_compiled_open_loop_actions.txt
07_action_verifier.json
07_action_verifier.md
08_phase_anchor_pipeline_combined.json
```

## First Local Compile Check

Using the five query-only phase packets from `../results/phase_interpreter_query_only_20260708_160500`,
the compiler produced:

```text
phone_on_base: flat_object_docking_place, 6 phases, 6 actions, verifier passed
put_rubbish_in_bin: object_into_open_receptacle, 6 phases, 6 actions, verifier passed
put_toilet_roll_on_stand: hole_over_vertical_stand, 6 phases, 6 actions, verifier passed
lamp_on: button_or_switch_press, 3 phases, 3 actions, verifier passed
close_microwave: hinged_door_close, 3 phases, 3 actions, verifier passed
```

The current compiler uses conservative default rotations. The next useful test
is module-by-module inspection, then adding retrieval-derived execution priors
for wrist rotation, approach direction, and task-specific lift heights.
