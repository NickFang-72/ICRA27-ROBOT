"""Normalize query-only phase plans and bind each phase to scene anchors."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .anchor_utils import add_voxel_offset, choose_anchor, extract_scene_anchors, normalized_text


TRANSPORT_FAMILIES = {
    "flat_object_docking_place",
    "object_into_open_receptacle",
    "hole_over_vertical_stand",
    "object_into_shelf",
    "elongated_object_into_stand",
    "round_object_into_open_goal",
}

PRESS_FAMILIES = {"button_press", "button_or_switch_press"}
HINGE_FAMILIES = {"hinged_door_close", "hinged_panel_close", "hinged_lid_open"}
ROTATION_FAMILIES = {"knob_or_handle_rotation", "screw_closure"}
PULL_FAMILIES = {"linear_pull_from_slot", "linear_handle_pull"}


def canonical_task_family(
    raw_family: Any,
    query_context: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Prefer descriptor-derived families over generic VLM family labels."""

    notes: List[str] = []
    raw = normalized_text(raw_family)
    profile = query_context.get("interaction_profile_p_j") or {}
    descriptor_family = normalized_text(profile.get("interaction_family"))
    task_key = normalized_text(query_context.get("task") or query_context.get("task_key"))

    if descriptor_family and descriptor_family != raw:
        if raw in {"", "unknown", "insertion", "placement", "pick_and_place", "push"}:
            notes.append(f"canonicalized generic task_family '{raw or 'unknown'}' to descriptor family '{descriptor_family}'")
            return descriptor_family, notes
        if task_key in {
            "put_toilet_roll_on_stand",
            "phone_on_base",
            "put_rubbish_in_bin",
            "close_microwave",
            "lamp_on",
        }:
            notes.append(f"used descriptor family '{descriptor_family}' for known task '{task_key}'")
            return descriptor_family, notes

    if task_key == "put_toilet_roll_on_stand":
        return "hole_over_vertical_stand", ["canonicalized toilet-roll task to hole_over_vertical_stand"]
    if task_key == "phone_on_base":
        return "flat_object_docking_place", ["canonicalized phone task to flat_object_docking_place"]
    if task_key == "put_rubbish_in_bin":
        return "object_into_open_receptacle", ["canonicalized bin task to object_into_open_receptacle"]
    if task_key == "close_microwave":
        return "hinged_door_close", ["canonicalized microwave task to hinged_door_close"]
    if task_key in {"lamp_on", "lamp_off"}:
        return "button_or_switch_press", ["canonicalized lamp task to button_or_switch_press"]
    return raw or descriptor_family or "unknown", notes


def _safe_lift_voxels(task_family: str, query_context: Dict[str, Any]) -> int:
    if task_family in {"object_into_open_receptacle", "hole_over_vertical_stand", "elongated_object_into_stand"}:
        return 18
    if task_family in {"flat_object_docking_place", "object_into_shelf", "round_object_into_open_goal"}:
        return 14
    return int(query_context.get("safe_lift_voxels", 12))


def _hover_voxels(task_family: str) -> int:
    if task_family in PRESS_FAMILIES or task_family in HINGE_FAMILIES:
        return 4
    return 8


def _phase_value(phase: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = phase.get(key, default)
    if value is None or value == "":
        return default
    return value


def _requested_anchor_role(phase: Dict[str, Any], task_family: str, previous_anchor_role: Optional[str]) -> str:
    raw_phase = normalized_text(phase.get("phase"))
    raw_role = normalized_text(phase.get("anchor_role") or phase.get("target_role") or phase.get("uses_point_role"))
    if raw_role not in {"", "none", "null", "unknown"}:
        return raw_role

    if raw_phase in {"pregrasp", "grasp", "press", "push", "pull", "rotate", "slide", "sweep"}:
        return "manipulated_object_contact"
    if raw_phase in {"lift", "retract"}:
        return previous_anchor_role or "manipulated_object_contact"
    if raw_phase in {"align", "move", "move_above_goal", "lower", "place", "insert", "release", "drop"}:
        if task_family in HINGE_FAMILIES and raw_phase in {"push", "release"}:
            return previous_anchor_role or "manipulated_object_contact"
        return "goal_region"
    return "none"


def _phase_offset(
    phase_name: str,
    task_family: str,
    anchor_role: str,
    gripper: str,
    query_context: Dict[str, Any],
) -> List[int]:
    safe_lift = _safe_lift_voxels(task_family, query_context)
    hover = _hover_voxels(task_family)
    phase = normalized_text(phase_name)

    if phase in {"pregrasp", "approach", "approach_press"}:
        return [0, 0, hover]
    if phase in {"lift", "move_above_goal", "align_above_goal"}:
        return [0, 0, safe_lift]
    if phase in {"align", "move"} and anchor_role == "goal_region" and gripper in {"closed", "close"}:
        return [0, 0, safe_lift]
    if phase == "retract":
        return [0, 0, hover]
    if phase == "press":
        return [0, 0, 0]
    return [0, 0, 0]


def _gripper_state(raw: Any, phase_name: str) -> str:
    text = normalized_text(raw)
    phase = normalized_text(phase_name)
    if text in {"open", "opened", "1"}:
        return "open"
    if text in {"close", "closed", "0"}:
        return "close" if phase == "grasp" else "closed"
    if phase in {"pregrasp", "approach", "approach_press", "release"}:
        return "open"
    if phase in {"grasp"}:
        return "close"
    return "closed"


def _transport_template(task_family: str, raw_phases: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if task_family not in TRANSPORT_FAMILIES:
        return None
    final_phase = "insert" if task_family in {"object_into_open_receptacle", "hole_over_vertical_stand", "object_into_shelf", "elongated_object_into_stand"} else "lower"
    if task_family == "round_object_into_open_goal":
        final_phase = "drop"
    return [
        {
            "phase": "pregrasp",
            "purpose": "approach manipulated object contact anchor with open gripper",
            "target_role": "manipulated_object_contact",
            "gripper": "open",
            "motion_constraint": "free_motion",
            "stop_condition": "ready_to_grasp",
        },
        {
            "phase": "grasp",
            "purpose": "close gripper at manipulated object contact anchor",
            "target_role": "manipulated_object_contact",
            "gripper": "close",
            "motion_constraint": "vertical_axis",
            "stop_condition": "object_grasped",
        },
        {
            "phase": "lift",
            "purpose": "lift the grasped object clear of the table or source fixture",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "vertical_axis",
            "stop_condition": "safe_lift_clearance",
        },
        {
            "phase": "move_above_goal",
            "purpose": "transport the grasped object above the target anchor",
            "target_role": "goal_region",
            "gripper": "closed",
            "motion_constraint": "free_motion",
            "stop_condition": "object_center_above_goal",
        },
        {
            "phase": final_phase,
            "purpose": "lower/place/insert the object at the target anchor",
            "target_role": "goal_region",
            "gripper": "closed",
            "motion_constraint": "vertical_axis",
            "stop_condition": "target_relation_satisfied",
        },
        {
            "phase": "release",
            "purpose": "open gripper only after the target relation is satisfied",
            "target_role": "goal_region",
            "gripper": "open",
            "motion_constraint": "none",
            "stop_condition": "released_at_goal",
        },
    ]


def _press_template(task_family: str) -> Optional[List[Dict[str, Any]]]:
    if task_family not in PRESS_FAMILIES:
        return None
    return [
        {
            "phase": "approach_press",
            "purpose": "approach the button/switch contact anchor",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "surface_normal_press_axis",
            "stop_condition": "ready_to_press",
        },
        {
            "phase": "press",
            "purpose": "press through the button/switch contact direction",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "surface_normal_press_axis",
            "stop_condition": "control_state_changed",
        },
        {
            "phase": "retract",
            "purpose": "retract from the button/switch",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "surface_normal_press_axis",
            "stop_condition": "clear_of_control",
        },
    ]


def _hinge_template(task_family: str) -> Optional[List[Dict[str, Any]]]:
    if task_family not in HINGE_FAMILIES:
        return None
    return [
        {
            "phase": "approach_contact",
            "purpose": "approach the door/panel/lid contact anchor away from hinge",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "hinge_arc",
            "stop_condition": "contact_ready",
        },
        {
            "phase": "push",
            "purpose": "push along hinge arc until requested state is reached",
            "target_role": "manipulated_object_contact",
            "gripper": "closed",
            "motion_constraint": "hinge_arc",
            "stop_condition": "hinged_state_reached",
        },
        {
            "phase": "retract",
            "purpose": "retract after panel state is reached",
            "target_role": "manipulated_object_contact",
            "gripper": "open",
            "motion_constraint": "none",
            "stop_condition": "clear_of_panel",
        },
    ]


def _select_normalized_phase_source(
    task_family: str,
    raw_phases: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    template = _transport_template(task_family, raw_phases)
    if template:
        notes.append(f"replaced raw phases with canonical transport template for {task_family}")
        return template, notes
    template = _press_template(task_family)
    if template:
        notes.append(f"replaced raw phases with canonical press template for {task_family}")
        return template, notes
    template = _hinge_template(task_family)
    if template:
        notes.append(f"replaced raw phases with canonical hinge template for {task_family}")
        return template, notes
    return copy.deepcopy(raw_phases), notes


def normalize_and_bind_phase_plan(
    phase_output: Dict[str, Any],
    query_context: Dict[str, Any],
    *,
    execution_prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonicalize phase family, choose anchors, and resolve phase coordinates."""

    phase_context = dict(query_context)
    if execution_prior and execution_prior.get("safe_lift_voxels") is not None:
        phase_context["safe_lift_voxels"] = execution_prior.get("safe_lift_voxels")

    task_family, notes = canonical_task_family(phase_output.get("task_family"), query_context)
    anchors = extract_scene_anchors(query_context.get("contact_hints_c_j") or {}, phase_output)
    raw_phases = phase_output.get("phase_plan") or []
    phase_source, source_notes = _select_normalized_phase_source(task_family, raw_phases)
    notes.extend(source_notes)

    anchored_phases: List[Dict[str, Any]] = []
    previous_anchor_role = None
    for index, raw_phase in enumerate(phase_source, start=1):
        phase_name = _phase_value(raw_phase, "phase", "unknown")
        gripper = _gripper_state(raw_phase.get("gripper"), phase_name)
        anchor_role = _requested_anchor_role(raw_phase, task_family, previous_anchor_role)
        preferred_index = raw_phase.get("anchor_index") or raw_phase.get("point_id") or raw_phase.get("anchor_id")
        anchor, trace = choose_anchor(
            anchors,
            anchor_role,
            target_object=raw_phase.get("target_object"),
            target_part=raw_phase.get("target_part"),
            preferred_index=preferred_index,
        )
        anchor_voxel = anchor.get("voxel_xyz") if anchor else None
        offset = _phase_offset(phase_name, task_family, anchor_role, gripper, phase_context)
        resolved_voxel = add_voxel_offset(anchor_voxel, offset) if anchor_voxel else None
        if anchor is not None:
            previous_anchor_role = anchor.get("role")
        anchored_phases.append(
            {
                "phase_index": index,
                "phase": normalized_text(phase_name) or "unknown",
                "purpose": raw_phase.get("purpose") or "",
                "anchor_role": anchor_role,
                "anchor_index": anchor.get("index") if anchor else None,
                "anchor_object": anchor.get("object") if anchor else None,
                "anchor_part": anchor.get("part") if anchor else None,
                "anchor_voxel_xyz": anchor_voxel,
                "anchor_world_xyz": anchor.get("world_xyz") if anchor else None,
                "anchor_selection_trace": trace,
                "offset_voxel_xyz": offset,
                "resolved_voxel_xyz": resolved_voxel,
                "gripper": gripper,
                "motion_constraint": raw_phase.get("motion_constraint") or "none",
                "stop_condition": raw_phase.get("stop_condition") or "",
                "source_phase": raw_phase,
            }
        )

    return {
        "schema_version": "phase_anchor_v1",
        "task": query_context.get("task") or query_context.get("task_key"),
        "instruction": query_context.get("instruction"),
        "task_family_raw": phase_output.get("task_family"),
        "task_family": task_family,
        "manipulated_object": phase_output.get("manipulated_object"),
        "target_object_or_region": phase_output.get("target_object_or_region"),
        "success_condition": phase_output.get("success_condition"),
        "execution_constraints": phase_output.get("execution_constraints") or {},
        "failure_modes_to_avoid": phase_output.get("failure_modes_to_avoid") or [],
        "scene_anchors": anchors,
        "anchored_phase_plan": anchored_phases,
        "execution_prior": execution_prior or {},
        "normalization_notes": notes,
    }
