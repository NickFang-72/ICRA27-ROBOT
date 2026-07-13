"""Compile anchored phases into open-loop 7D discrete keyframes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .anchor_utils import add_voxel_offset, clip_voxel, normalized_text


DEFAULT_ROTATION = [0, 36, 0]
FAMILY_ROTATION_DEFAULTS = {
    "button_or_switch_press": [0, 36, 0],
    "button_press": [0, 36, 0],
    "hinged_door_close": [0, 36, 0],
    "hinged_panel_close": [0, 36, 0],
    "hinged_lid_open": [0, 36, 0],
    "flat_object_docking_place": [0, 36, 0],
    "object_into_open_receptacle": [0, 36, 0],
    "hole_over_vertical_stand": [0, 36, 0],
}


def _int_list(values: Iterable[Any], default: Optional[List[int]] = None) -> List[int]:
    try:
        parsed = [int(round(float(value))) for value in values]
    except (TypeError, ValueError):
        return list(default or [])
    return parsed


def _rotation_for_phase(
    phase: Dict[str, Any],
    normalized_plan: Dict[str, Any],
    execution_prior: Dict[str, Any],
) -> List[int]:
    phase_name = normalized_text(phase.get("phase"))
    phase_rotations = execution_prior.get("phase_rotations") or {}
    if phase_name in phase_rotations:
        rotation = _int_list(phase_rotations[phase_name], DEFAULT_ROTATION)
        if len(rotation) == 3:
            return rotation

    default_rotation = execution_prior.get("default_rotation")
    if default_rotation is not None:
        rotation = _int_list(default_rotation, DEFAULT_ROTATION)
        if len(rotation) == 3:
            return rotation

    family = normalized_text(normalized_plan.get("task_family"))
    return list(FAMILY_ROTATION_DEFAULTS.get(family, DEFAULT_ROTATION))


def _gripper_to_binary(gripper: Any) -> int:
    text = normalized_text(gripper)
    if text in {"open", "opened", "release", "1"}:
        return 1
    return 0


def _phase_extra_motion_offset(
    phase: Dict[str, Any],
    normalized_plan: Dict[str, Any],
    execution_prior: Dict[str, Any],
) -> List[int]:
    """Small optional offsets for non-transport actions.

    These are intentionally conservative. Later, retrieved demos can fill this
    through execution_prior["phase_motion_offsets"].
    """

    phase_name = normalized_text(phase.get("phase"))
    phase_offsets = execution_prior.get("phase_motion_offsets") or {}
    if phase_name in phase_offsets:
        offset = _int_list(phase_offsets[phase_name], [0, 0, 0])
        if len(offset) == 3:
            return offset

    family = normalized_text(normalized_plan.get("task_family"))
    if family in {"button_press", "button_or_switch_press"} and phase_name == "retract":
        return [0, 0, 5]
    if family in {"hinged_door_close", "hinged_panel_close"} and phase_name == "push":
        return _int_list(execution_prior.get("hinge_push_offset", [0, -8, 0]), [0, -8, 0])
    if family == "hinged_lid_open" and phase_name in {"pull", "lift"}:
        return [0, 0, 10]
    return [0, 0, 0]


def _action_from_phase(
    phase: Dict[str, Any],
    normalized_plan: Dict[str, Any],
    execution_prior: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    voxel = phase.get("resolved_voxel_xyz") or phase.get("anchor_voxel_xyz")
    if voxel is None:
        return None
    extra_offset = _phase_extra_motion_offset(phase, normalized_plan, execution_prior)
    translation = add_voxel_offset(voxel, extra_offset)
    rotation = _rotation_for_phase(phase, normalized_plan, execution_prior)
    gripper = _gripper_to_binary(phase.get("gripper"))
    action = [*clip_voxel(translation), *rotation, gripper]
    return {
        "phase_index": phase.get("phase_index"),
        "phase": phase.get("phase"),
        "anchor_role": phase.get("anchor_role"),
        "anchor_index": phase.get("anchor_index"),
        "anchor_voxel_xyz": phase.get("anchor_voxel_xyz"),
        "resolved_voxel_xyz": phase.get("resolved_voxel_xyz"),
        "extra_motion_offset_voxel_xyz": extra_offset,
        "rotation_discrete_euler": rotation,
        "gripper": phase.get("gripper"),
        "action_7d": action,
    }


def compile_open_loop_actions(
    normalized_plan: Dict[str, Any],
    *,
    execution_prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile anchored phases into a candidate open-loop 7D keyframe list."""

    prior = dict(normalized_plan.get("execution_prior") or {})
    if execution_prior:
        prior.update(execution_prior)

    compiled_steps: List[Dict[str, Any]] = []
    skipped_steps: List[Dict[str, Any]] = []
    for phase in normalized_plan.get("anchored_phase_plan") or []:
        compiled = _action_from_phase(phase, normalized_plan, prior)
        if compiled is None:
            skipped_steps.append(
                {
                    "phase_index": phase.get("phase_index"),
                    "phase": phase.get("phase"),
                    "reason": "no resolved or anchor voxel",
                    "anchor_role": phase.get("anchor_role"),
                }
            )
            continue
        compiled_steps.append(compiled)

    return {
        "schema_version": "phase_anchor_actions_v1",
        "task": normalized_plan.get("task"),
        "task_family": normalized_plan.get("task_family"),
        "success_condition": normalized_plan.get("success_condition"),
        "execution_prior_used": bool(prior),
        "execution_prior": prior,
        "compiled_steps": compiled_steps,
        "skipped_steps": skipped_steps,
        "actions_7d": [step["action_7d"] for step in compiled_steps],
        "compiler_notes": [
            "Open-loop candidate generated from anchored phases.",
            "Rotations are defaults unless execution_prior supplies default_rotation or phase_rotations.",
            "Retrieval may supply approach angle, lift height, wrist rotation, or phase_motion_offsets later.",
        ],
    }
