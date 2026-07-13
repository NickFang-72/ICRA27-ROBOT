"""Phase-anchor open-loop manipulation pipeline.

The package is intentionally split into small pure-Python modules so each stage
can be inspected before connecting it to RLBench execution.
"""

from .phase_normalizer import normalize_and_bind_phase_plan
from .primitive_compiler import compile_open_loop_actions
from .action_verifier import verify_pipeline_outputs
from .pipeline import run_phase_anchor_pipeline

__all__ = [
    "normalize_and_bind_phase_plan",
    "compile_open_loop_actions",
    "verify_pipeline_outputs",
    "run_phase_anchor_pipeline",
]
