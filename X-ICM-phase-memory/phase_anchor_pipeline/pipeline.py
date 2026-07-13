"""Orchestrator for the query-first phase-anchor open-loop pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .action_verifier import verify_pipeline_outputs
from .phase_normalizer import normalize_and_bind_phase_plan
from .primitive_compiler import compile_open_loop_actions


def run_phase_anchor_pipeline(
    phase_output: Dict[str, Any],
    query_context: Dict[str, Any],
    *,
    execution_prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run normalize/bind -> compile -> verify.

    Retrieval is intentionally represented as an optional execution_prior. The
    phase plan remains authoritative; retrieval can only fill HOW details such
    as rotation, approach offset, or lift height.
    """

    normalized = normalize_and_bind_phase_plan(
        phase_output,
        query_context,
        execution_prior=execution_prior,
    )
    compilation = compile_open_loop_actions(
        normalized,
        execution_prior=execution_prior,
    )
    verification = verify_pipeline_outputs(normalized, compilation)
    return {
        "schema_version": "phase_anchor_pipeline_v1",
        "task": query_context.get("task") or query_context.get("task_key"),
        "instruction": query_context.get("instruction"),
        "design_rule": "Query decides WHAT. Anchors provide WHERE. Retrieval helps HOW. Compiler produces 7D actions.",
        "retrieval_policy": "No retrieved scenes enter the phase interpreter. Retrieval may only populate execution_prior for the compiler.",
        "normalized_phase_plan": normalized,
        "compiled_actions": compilation,
        "verification": verification,
    }
