"""Controller-side guardrails for phase memory execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from phase_anchor_pipeline.anchor_utils import clip_voxel, normalized_text, voxel_distance


DEFAULT_NOOP_ACTION = [50, 50, 80, 0, 36, 0, 1]

REQUIRED_PHASES = {
    "grasp",
    "contact",
    "press",
    "push",
    "pull",
    "rotate",
    "slide",
    "sweep",
    "insert",
    "lower",
    "place",
    "drop",
    "lift",
}

OPTIONAL_PHASES = {
    "pregrasp",
    "approach",
    "approach_press",
    "align",
    "align_above_goal",
    "move",
    "move_above_goal",
    "retract",
    "release",
    "stop",
}


def is_required_phase(phase: Dict[str, Any]) -> bool:
    name = normalized_text(phase.get("phase"))
    if phase.get("required") is True:
        return True
    if phase.get("required") is False:
        return False
    if name in REQUIRED_PHASES:
        return True
    if name in OPTIONAL_PHASES:
        return False
    role = normalized_text(phase.get("anchor_role"))
    return role in {"manipulated_object_contact", "tool_working_edge"}


def sanitize_action(action: Any, fallback: Optional[List[int]] = None) -> List[int]:
    if not isinstance(action, list) or len(action) != 7:
        return list(fallback or DEFAULT_NOOP_ACTION)
    try:
        parsed = [int(round(float(value))) for value in action]
    except (TypeError, ValueError):
        return list(fallback or DEFAULT_NOOP_ACTION)
    parsed[:3] = clip_voxel(parsed[:3])
    parsed[3:6] = [max(0, min(71, value)) for value in parsed[3:6]]
    parsed[6] = 1 if parsed[6] >= 1 else 0
    return parsed


def guard_action(
    *,
    phase: Dict[str, Any],
    action_7d: List[int],
    attempts_so_far: int,
    repeated_action: bool,
    repeated_voxel: bool,
    max_actions_per_phase: int,
    max_repeated_voxel: int,
    anchor_distance_limit: float,
) -> Dict[str, Any]:
    """Return whether a candidate action can execute for this phase."""

    sanitized = sanitize_action(action_7d)
    issues = []
    decision = "execute"

    anchor = phase.get("resolved_voxel_xyz") or phase.get("anchor_voxel_xyz")
    if anchor and normalized_text(phase.get("anchor_role")) not in {"none", ""}:
        distance = voxel_distance(sanitized[:3], anchor)
        if distance > anchor_distance_limit:
            decision = "far_from_phase_anchor_blocked"
            issues.append(
                {
                    "code": "far_from_phase_anchor",
                    "message": f"action voxel is {distance:.1f} voxels from phase anchor",
                }
            )
    if attempts_so_far >= max_actions_per_phase:
        decision = "phase_budget_exhausted"
        issues.append({"code": "phase_budget_exhausted", "message": "max actions per phase reached"})
    if repeated_action:
        decision = "repeated_action_blocked"
        issues.append({"code": "repeated_action", "message": "same 7D action was already executed"})
    elif repeated_voxel and attempts_so_far >= max_repeated_voxel:
        decision = "repeated_voxel_blocked"
        issues.append({"code": "repeated_voxel", "message": "same target voxel would be repeated"})

    return {
        "decision": decision,
        "allowed": decision == "execute",
        "action_7d": sanitized,
        "issues": issues,
    }


def phase_failure_transition(phase: Dict[str, Any]) -> str:
    return "failed_required" if is_required_phase(phase) else "skipped_optional"
