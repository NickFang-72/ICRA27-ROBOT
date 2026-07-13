"""Verifier for phase-anchor plans and compiled 7D actions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .anchor_utils import normalized_text, voxel_distance
from .phase_normalizer import HINGE_FAMILIES, PRESS_FAMILIES, TRANSPORT_FAMILIES


def _issue(severity: str, code: str, message: str, phase_index: Optional[int] = None) -> Dict[str, Any]:
    result = {"severity": severity, "code": code, "message": message}
    if phase_index is not None:
        result["phase_index"] = phase_index
    return result


def _is_goal_phase(step: Dict[str, Any]) -> bool:
    return normalized_text(step.get("anchor_role")) == "goal_region"


def _is_contact_phase(step: Dict[str, Any]) -> bool:
    return normalized_text(step.get("anchor_role")) == "manipulated_object_contact"


def verify_normalized_phase_plan(normalized_plan: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    family = normalized_text(normalized_plan.get("task_family"))
    phases = normalized_plan.get("anchored_phase_plan") or []

    if not phases:
        issues.append(_issue("error", "empty_phase_plan", "No anchored phases were produced."))

    for phase in phases:
        if normalized_text(phase.get("anchor_role")) not in {"none", ""} and not phase.get("anchor_voxel_xyz"):
            issues.append(
                _issue(
                    "error",
                    "missing_phase_anchor",
                    f"Phase '{phase.get('phase')}' requested anchor role '{phase.get('anchor_role')}' but no anchor was bound.",
                    phase.get("phase_index"),
                )
            )

    if family in TRANSPORT_FAMILIES:
        has_contact = any(_is_contact_phase(phase) for phase in phases)
        has_goal = any(_is_goal_phase(phase) for phase in phases)
        has_release_at_goal = any(
            normalized_text(phase.get("phase")) == "release" and _is_goal_phase(phase)
            for phase in phases
        )
        if not has_contact:
            issues.append(_issue("error", "transport_missing_contact_anchor", "Transport task has no manipulated_object_contact phase."))
        if not has_goal:
            issues.append(_issue("error", "transport_missing_goal_anchor", "Transport task has no goal_region phase."))
        if not has_release_at_goal:
            issues.append(_issue("error", "transport_release_not_goal", "Transport task must release at a goal_region anchor."))

    if family in PRESS_FAMILIES:
        if not any(normalized_text(phase.get("phase")) == "press" for phase in phases):
            issues.append(_issue("error", "press_missing_press_phase", "Press task has no press phase."))

    if family in HINGE_FAMILIES:
        if not any(normalized_text(phase.get("motion_constraint")) == "hinge_arc" for phase in phases):
            issues.append(_issue("warning", "hinge_without_hinge_arc", "Hinge task should use hinge_arc motion constraints."))

    return {
        "passed": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
    }


def verify_compiled_actions(
    compilation: Dict[str, Any],
    normalized_plan: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    family = normalized_text(normalized_plan.get("task_family"))
    steps = compilation.get("compiled_steps") or []
    actions = compilation.get("actions_7d") or []

    if not actions:
        issues.append(_issue("error", "no_compiled_actions", "Compiler produced no 7D actions."))

    for idx, action in enumerate(actions, start=1):
        if not isinstance(action, list) or len(action) != 7:
            issues.append(_issue("error", "bad_action_shape", f"Action {idx} is not a 7D list."))
            continue
        for value in action:
            if not isinstance(value, int):
                issues.append(_issue("error", "bad_action_type", f"Action {idx} contains a non-integer value."))
                break
        if action[-1] not in {0, 1}:
            issues.append(_issue("error", "bad_gripper_state", f"Action {idx} has non-binary gripper state {action[-1]}."))

    if family in TRANSPORT_FAMILIES and steps:
        first_close_index = next((i for i, step in enumerate(steps) if step["action_7d"][-1] == 0), None)
        first_goal_index = next((i for i, step in enumerate(steps) if _is_goal_phase(step)), None)
        release_steps = [step for step in steps if step["action_7d"][-1] == 1]
        if first_close_index is None:
            issues.append(_issue("error", "transport_never_closes", "Transport action never closes the gripper."))
        if first_goal_index is not None and first_close_index is not None and first_goal_index < first_close_index:
            issues.append(_issue("error", "goal_before_grasp", "Transport reaches goal before closing the gripper."))
        if release_steps and not _is_goal_phase(release_steps[-1]):
            issues.append(_issue("error", "last_release_not_goal", "Final open-gripper action is not anchored to the goal."))
        if not release_steps:
            issues.append(_issue("error", "missing_release", "Transport task has no release/open action."))

        contact_steps = [step for step in steps if _is_contact_phase(step)]
        goal_steps = [step for step in steps if _is_goal_phase(step)]
        if contact_steps and goal_steps:
            contact_voxel = contact_steps[0].get("anchor_voxel_xyz")
            goal_voxel = goal_steps[0].get("anchor_voxel_xyz")
            if contact_voxel and goal_voxel and voxel_distance(contact_voxel, goal_voxel) < 5:
                issues.append(
                    _issue(
                        "warning",
                        "contact_goal_very_close",
                        "Contact and goal anchors are very close; verify contact extraction did not duplicate roles.",
                    )
                )
            contact_z = contact_steps[0]["action_7d"][2]
            max_closed_z = max((step["action_7d"][2] for step in steps if step["action_7d"][-1] == 0), default=contact_z)
            if max_closed_z < contact_z + 6:
                issues.append(_issue("warning", "low_lift_clearance", "Closed-gripper transport has little z lift clearance."))

    if family in PRESS_FAMILIES:
        press_steps = [step for step in steps if normalized_text(step.get("phase")) == "press"]
        if not press_steps:
            issues.append(_issue("error", "missing_press_action", "Press task compiled without a press action."))

    if family in HINGE_FAMILIES:
        push_steps = [step for step in steps if normalized_text(step.get("phase")) == "push"]
        if not push_steps:
            issues.append(_issue("error", "missing_hinge_push", "Hinge task compiled without a push action."))

    return {
        "passed": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
    }


def verify_pipeline_outputs(
    normalized_plan: Dict[str, Any],
    compilation: Dict[str, Any],
) -> Dict[str, Any]:
    phase_check = verify_normalized_phase_plan(normalized_plan)
    action_check = verify_compiled_actions(compilation, normalized_plan)
    issues = [*phase_check["issues"], *action_check["issues"]]
    return {
        "schema_version": "phase_anchor_verifier_v1",
        "task": normalized_plan.get("task"),
        "task_family": normalized_plan.get("task_family"),
        "passed": not any(issue["severity"] == "error" for issue in issues),
        "phase_plan_passed": phase_check["passed"],
        "compiled_actions_passed": action_check["passed"],
        "issues": issues,
        "repair_recommendations": repair_recommendations(normalized_plan, compilation, issues),
    }


def repair_recommendations(
    normalized_plan: Dict[str, Any],
    compilation: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> List[str]:
    recommendations: List[str] = []
    issue_codes = {issue["code"] for issue in issues}
    family = normalized_text(normalized_plan.get("task_family"))
    if "transport_missing_goal_anchor" in issue_codes:
        recommendations.append("Regenerate or repair c_j goal_region anchors before compiling this transport task.")
    if "transport_release_not_goal" in issue_codes or "last_release_not_goal" in issue_codes:
        recommendations.append("Force the final release phase to bind to the selected goal_region anchor.")
    if "low_lift_clearance" in issue_codes and family in TRANSPORT_FAMILIES:
        recommendations.append("Increase safe_lift_voxels or retrieval-provided lift height for this task family.")
    if "hinge_without_hinge_arc" in issue_codes:
        recommendations.append("Normalize hinge phases to approach_contact -> push along hinge_arc -> retract.")
    if not recommendations:
        recommendations.append("No automatic repair needed before module-level inspection.")
    return recommendations
