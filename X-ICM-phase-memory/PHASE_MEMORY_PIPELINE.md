# Phase Memory VLM Controller

This checkpoint is copied from `X-ICM-phase-anchor` and changes the execution
stage from static phase-action replay to a no-repair-loop phase-step controller.

## Design

1. Query-only phase interpreter produces phase packets.
2. Phase-anchor normalizer binds each phase to scene anchors.
3. Retrieved seen demos are compressed into long-term memory.
4. At runtime, each RLBench step feeds the current phase, compiler candidate
   actions, long-term memory, and short-term memory to QwenVL. Images are off
   by default in the phase-step prompt so the selector focuses on the bounded
   action candidates.
5. QwenVL returns one JSON decision: `continue`, `done`, or `failed`, and when
   continuing it selects a `candidate_action_id` rather than inventing a fresh
   coordinate.
6. The controller enforces a phase budget. There is no endless repair loop.

## Failure Policy

- Required phase fails or repeats: abort/hold for the rest of the episode.
- Optional phase fails: skip to the next phase.
- Same 7D action is blocked.
- Same target voxel is blocked after the configured repeated-voxel budget.

## Main Files

- `phase_memory_pipeline/long_term_memory.py`
- `phase_memory_pipeline/short_term_memory.py`
- `phase_memory_pipeline/phase_step_prompt.py`
- `phase_memory_pipeline/action_guard.py`
- `phase_memory_pipeline/phase_memory_controller.py`
- `crosstask_icl_agent.py`

## Runtime Environment

```bash
export XICM_PHASE_MEMORY_REVIEW_ROOT=/path/to/phase_packets
export XICM_PHASE_MEMORY_RETRIEVAL_METHOD=lang_vis.out.geo.aff_v3.rerank_top50
export XICM_PHASE_MEMORY_RETRIEVAL_K=4
export XICM_PHASE_MEMORY_MAX_ACTIONS_PER_PHASE=2
export XICM_PHASE_MEMORY_USE_IMAGES=0
```

Run with:

```bash
framework.ranking_method=phase_memory
method.name=phase_memory_Qwen2.5.VL.7B.instruct.debug
```

Runtime review packets are written under:

```text
logs/.../phase_memory_runtime/<task>/episode<N>/step_XXX_phase_YY/
```

Each step folder contains the prompt packet, raw VLM output, controller decision,
and short-term memory after action execution.
