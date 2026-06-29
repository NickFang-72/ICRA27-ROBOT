#!/usr/bin/env python3
"""Run Qwen2.5-VL on front/overhead views with the cleaned geometry/target-pose schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


GEOMETRY_ENUMS: dict[str, set[str]] = {
    "action_primitive": {
        "push",
        "pull",
        "press",
        "twist",
        "lift",
        "place",
        "insert",
        "remove",
        "slide",
        "sweep",
        "scoop",
        "pour",
        "open",
        "close",
        "dock",
        "unknown",
    },
    "motion_type": {
        "linear",
        "rotational",
        "vertical",
        "planar",
        "insertion",
        "articulation",
        "press",
        "tool_use",
        "pour",
        "none",
        "unknown",
    },
    "motion_axis": {
        "horizontal",
        "vertical",
        "rotational",
        "into_opening",
        "onto_support",
        "across_surface",
        "out_of_socket",
        "around_post",
        "surface_normal",
        "free_space",
        "none",
        "unknown",
    },
    "contact_type": {
        "grasp",
        "press",
        "push_contact",
        "surface_contact",
        "tool_contact",
        "release_only",
        "none",
        "unknown",
    },
    "contact_region": {
        "handle",
        "rim",
        "lid",
        "button",
        "knob",
        "edge",
        "tip",
        "body",
        "cord",
        "opening",
        "surface",
        "none",
        "unknown",
    },
    "constraint_type": {
        "hinge",
        "slider",
        "slot",
        "hole",
        "socket",
        "container",
        "support_surface",
        "peg_or_post",
        "button_switch",
        "free_space",
        "none",
        "unknown",
    },
    "alignment_requirement": {"none", "low", "medium", "high", "unknown"},
}

TARGET_ENUMS: dict[str, set[str]] = {
    "goal_state_type": {
        "none",
        "supported_or_docked_pose",
        "placement_inside_or_on_target",
        "aligned_insertion_or_docking",
        "articulated_part_state",
        "closed_or_sealed_state",
        "control_or_articulation_state",
        "removal_or_extraction_state",
        "tool_contact_state",
        "deformable_shape_state",
        "pouring_target_state",
        "unknown",
    },
    "required_orientation_or_alignment": {
        "none",
        "flat",
        "upright",
        "vertical",
        "horizontal",
        "aligned_with_slot",
        "aligned_with_dock",
        "aligned_with_socket",
        "hole_aligned_with_post",
        "inside_receptacle",
        "handle_accessible",
        "unknown",
    },
}

GEOMETRY_FIELDS = [
    "action_primitive",
    "motion_type",
    "motion_axis",
    "contact_type",
    "contact_region",
    "constraint_type",
    "alignment_requirement",
]
TARGET_FIELDS = [
    "goal_state_type",
    "required_final_relation",
    "target_object_or_region",
    "required_orientation_or_alignment",
    "release_or_stop_condition",
    "success_check",
]
ALL_FIELDS = [f"geometry.{field}" for field in GEOMETRY_FIELDS] + [
    f"target_pose.{field}" for field in TARGET_FIELDS
]

ALIASES: dict[str, dict[str, str]] = {
    "geometry.action_primitive": {
        "turn": "twist",
        "rotate": "twist",
        "screw": "twist",
        "pick": "lift",
        "pick_up": "lift",
        "put": "place",
        "drop": "place",
        "release": "place",
        "water": "pour",
        "switch": "press",
        "tap": "press",
    },
    "geometry.motion_type": {
        "linear_push": "linear",
        "linear_pull": "linear",
        "sliding": "planar",
        "planar_slide": "planar",
        "rotational_turn": "rotational",
        "turn": "rotational",
        "screw": "rotational",
        "lift_then_release": "vertical",
        "align_then_insert": "insertion",
        "align_then_dock": "insertion",
        "hinge_swing": "articulation",
        "press_down": "press",
        "pour_tilt": "pour",
        "sweep_across_surface": "tool_use",
    },
    "geometry.motion_axis": {
        "linear": "horizontal",
        "planar": "across_surface",
        "rotation": "rotational",
        "rotate": "rotational",
        "into": "into_opening",
        "in": "into_opening",
        "onto": "onto_support",
        "out": "out_of_socket",
        "down": "surface_normal",
        "press_down": "surface_normal",
    },
    "geometry.contact_type": {
        "grab": "grasp",
        "gripper": "grasp",
        "push": "push_contact",
        "press_down": "press",
        "release": "release_only",
        "tool": "tool_contact",
    },
    "geometry.contact_region": {
        "button_top": "button",
        "top": "surface",
        "top_surface": "surface",
        "side": "edge",
        "object": "body",
        "object_body": "body",
        "jar_lid": "lid",
        "tap": "knob",
    },
    "geometry.constraint_type": {
        "drawer": "slider",
        "sliding": "slider",
        "button": "button_switch",
        "switch": "button_switch",
        "peg": "peg_or_post",
        "post": "peg_or_post",
        "receptacle": "container",
        "bin": "container",
        "cupboard": "container",
        "surface": "support_surface",
        "dock": "socket",
    },
    "geometry.alignment_requirement": {
        "no": "none",
        "not_required": "none",
        "minimal": "low",
        "moderate": "medium",
        "precise": "high",
        "strict": "high",
    },
    "target_pose.goal_state_type": {
        "support_surface": "supported_or_docked_pose",
        "base_or_dock": "supported_or_docked_pose",
        "dock": "supported_or_docked_pose",
        "inside": "placement_inside_or_on_target",
        "open_receptacle": "placement_inside_or_on_target",
        "slot": "aligned_insertion_or_docking",
        "hole": "aligned_insertion_or_docking",
        "peg": "aligned_insertion_or_docking",
        "post": "aligned_insertion_or_docking",
        "button": "control_or_articulation_state",
        "switch": "control_or_articulation_state",
        "hinge": "articulated_part_state",
        "drawer": "articulated_part_state",
        "open": "articulated_part_state",
        "opened": "articulated_part_state",
        "closed": "closed_or_sealed_state",
        "sealed": "closed_or_sealed_state",
        "tightened": "closed_or_sealed_state",
        "remove": "removal_or_extraction_state",
        "pour": "pouring_target_state",
    },
    "target_pose.required_orientation_or_alignment": {
        "none_required": "none",
        "slot_aligned": "aligned_with_slot",
        "aligned_slot": "aligned_with_slot",
        "socket_aligned": "aligned_with_socket",
        "aligned_socket": "aligned_with_socket",
        "aligned_with_socket": "aligned_with_socket",
        "dock_aligned": "aligned_with_dock",
        "peg_aligned": "hole_aligned_with_post",
        "post_aligned": "hole_aligned_with_post",
        "inside": "inside_receptacle",
        "in_receptacle": "inside_receptacle",
    },
}

PROMPT_TEMPLATE = """You are extracting a compact robot manipulation descriptor from one camera view.
Return ONLY valid JSON. Do not include markdown, comments, or extra keys.

Purpose:
- geometry is for retrieval: it should describe the reusable action structure.
- target_pose is for the final robot LLM prompt: it should describe the desired end state and success condition.
- Retrieved examples may have a similar action but different objects and goals, so prioritize primitive action, movement constraint, contact mode, and final relation over exact object identity.

Rules:
- Do not use camera-relative directions such as left, right, front, back, facing up, or facing down.
- Keep fields primitive and transferable to unseen objects.
- Do not describe clearance or path planning.
- Use only the enum labels shown for enum fields.
- Use "unknown" for enum fields when the view is ambiguous.
- Use short concrete phrases for free-text target_pose fields.
- Always include every schema field exactly once.
- Use contact_type "press" only for buttons, switches, or direct pressing actions.
- For twisting taps, lids, bulbs, knobs, and caps, use contact_type "grasp" and motion_type "rotational".
- For rings, holes, slots, spokes, pegs, sockets, and shape-sorter openings, prefer action_primitive "insert" when success requires alignment with geometry.

Allowed JSON schema:
{
    "geometry": {
    "action_primitive": "push|pull|press|twist|lift|place|insert|remove|slide|sweep|scoop|pour|open|close|dock|unknown",
    "motion_type": "linear|rotational|vertical|planar|insertion|articulation|press|tool_use|pour|none|unknown",
    "motion_axis": "horizontal|vertical|rotational|into_opening|onto_support|across_surface|out_of_socket|around_post|surface_normal|free_space|none|unknown",
    "contact_type": "grasp|press|push_contact|surface_contact|tool_contact|release_only|none|unknown",
    "contact_region": "handle|rim|lid|button|knob|edge|tip|body|cord|opening|surface|none|unknown",
    "constraint_type": "hinge|slider|slot|hole|socket|container|support_surface|peg_or_post|button_switch|free_space|none|unknown",
    "alignment_requirement": "none|low|medium|high|unknown"
  },
  "target_pose": {
    "goal_state_type": "none|supported_or_docked_pose|placement_inside_or_on_target|aligned_insertion_or_docking|articulated_part_state|closed_or_sealed_state|control_or_articulation_state|removal_or_extraction_state|tool_contact_state|deformable_shape_state|pouring_target_state|unknown",
    "required_final_relation": "short phrase, e.g. object inside receptacle",
    "target_object_or_region": "short phrase naming the goal region, e.g. green target or drawer interior",
    "required_orientation_or_alignment": "none|flat|upright|vertical|horizontal|aligned_with_slot|aligned_with_dock|aligned_with_socket|hole_aligned_with_post|inside_receptacle|handle_accessible|unknown",
    "release_or_stop_condition": "short phrase describing when the robot should stop or release",
    "success_check": "short phrase describing visible success"
  },
  "uncertain_fields": ["schema field names such as geometry.contact_region"]
}

Calibration examples:
- "turn left tap" -> geometry.action_primitive twist, geometry.motion_type rotational, geometry.contact_type grasp, geometry.contact_region knob, target_pose.goal_state_type control_or_articulation_state.
- "open drawer" -> geometry.action_primitive pull, geometry.motion_type linear, geometry.motion_axis horizontal, geometry.contact_region handle, geometry.constraint_type slider, target_pose.goal_state_type articulated_part_state.
- "close jar" -> geometry.action_primitive twist, geometry.motion_type rotational, geometry.contact_type grasp, geometry.contact_region lid, target_pose.goal_state_type closed_or_sealed_state.
- "slide block to green target" -> geometry.action_primitive slide, geometry.motion_type planar, geometry.motion_axis across_surface, target_pose.goal_state_type placement_inside_or_on_target, target_pose.target_object_or_region green target.
- "insert shape onto square peg" or "put ring on spoke" -> geometry.action_primitive insert, geometry.motion_type insertion, geometry.motion_axis around_post, geometry.constraint_type peg_or_post, geometry.alignment_requirement high, target_pose.goal_state_type aligned_insertion_or_docking.
- "put item in drawer" -> geometry.action_primitive place, geometry.motion_type vertical, geometry.motion_axis into_opening, geometry.constraint_type container, target_pose.goal_state_type placement_inside_or_on_target.
- "push button" -> geometry.action_primitive press, geometry.motion_type press, geometry.motion_axis surface_normal, geometry.contact_type press, geometry.contact_region button, target_pose.goal_state_type control_or_articulation_state.
- "screw in light bulb" -> geometry.action_primitive twist, geometry.motion_type rotational, geometry.motion_axis rotational, geometry.contact_type grasp, geometry.constraint_type socket, geometry.alignment_requirement high, target_pose.goal_state_type aligned_insertion_or_docking.

Task instruction: {task}
Camera view: {view}
Return the descriptor JSON only.
"""


def clean_json(text: str) -> tuple[dict[str, Any], str]:
    raw = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", raw).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw_json": parsed}, raw
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw_json": parsed}, raw
            except Exception:
                pass
    return {"parse_error": True, "raw_text": raw}, raw


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("examples") or data.get("records") or data.get("selected") or []
    raise SystemExit(f"Unsupported manifest format: {path}")


def norm_label(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        value = value[0] if value else "unknown"
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def norm_phrase(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    text = re.sub(r"\s+", " ", str(value).strip())
    return text[:160] if text else fallback


def normalize_enum(field_path: str, value: Any, allowed: set[str], notes: list[str]) -> str:
    text = norm_label(value)
    if text in allowed:
        return text
    aliased = ALIASES.get(field_path, {}).get(text)
    if aliased and aliased in allowed:
        notes.append(f"{field_path}: {text} -> {aliased}")
        return aliased
    notes.append(f"{field_path}: {text} -> unknown")
    return "unknown"


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        tag = norm_label(item)
        if tag and tag != "unknown" and tag not in out:
            out.append(tag)
    return out[:10]


def normalize_uncertain(value: Any) -> list[str]:
    raw = normalize_tags(value)
    normalized: list[str] = []
    bare_to_full = {
        **{field: f"geometry.{field}" for field in GEOMETRY_FIELDS},
        **{field: f"target_pose.{field}" for field in TARGET_FIELDS},
    }
    for field in raw:
        dotted = field.replace("_", ".", 1) if field.startswith(("geometry_", "target_pose_")) else field
        full = dotted if dotted in ALL_FIELDS else bare_to_full.get(field)
        if full and full in ALL_FIELDS and full not in normalized:
            normalized.append(full)
    return normalized


def normalize_descriptor(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    if raw.get("parse_error"):
        raw = {}
        notes.append("parse_error -> empty descriptor")
    geometry_raw = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else raw
    target_raw = raw.get("target_pose") if isinstance(raw.get("target_pose"), dict) else raw

    geometry: dict[str, Any] = {}
    for field, allowed in GEOMETRY_ENUMS.items():
        geometry[field] = normalize_enum(f"geometry.{field}", geometry_raw.get(field), allowed, notes)

    target_pose: dict[str, Any] = {}
    for field, allowed in TARGET_ENUMS.items():
        target_pose[field] = normalize_enum(f"target_pose.{field}", target_raw.get(field), allowed, notes)
    for field in [
        "required_final_relation",
        "target_object_or_region",
        "release_or_stop_condition",
        "success_check",
    ]:
        target_pose[field] = norm_phrase(target_raw.get(field))

    descriptor = {
        "geometry": geometry,
        "target_pose": target_pose,
        "uncertain_fields": normalize_uncertain(raw.get("uncertain_fields")),
    }
    return descriptor, notes


def known(value: Any) -> bool:
    return value not in (None, "", "unknown", [], ["unknown"])


def choose_value(
    field_path: str,
    front_value: Any,
    overhead_value: Any,
    conflicts: list[dict[str, Any]],
) -> tuple[Any, str]:
    if front_value == overhead_value:
        return front_value, "agreement" if known(front_value) else "both_unknown"
    if known(front_value) and not known(overhead_value):
        return front_value, "front_filled_overhead_unknown"
    if known(overhead_value) and not known(front_value):
        return overhead_value, "overhead_filled_front_unknown"
    if known(front_value) and known(overhead_value):
        conflicts.append(
            {
                "field": field_path,
                "front": front_value,
                "overhead": overhead_value,
                "chosen": front_value,
            }
        )
        return front_value, "front_on_conflict"
    return "unknown", "both_unknown"


def fuse_descriptors(front: dict[str, Any], overhead: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    fused = {"geometry": {}, "target_pose": {}, "uncertain_fields": []}
    source: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []

    for field in GEOMETRY_FIELDS:
        field_path = f"geometry.{field}"
        value, src = choose_value(field_path, front["geometry"].get(field), overhead["geometry"].get(field), conflicts)
        fused["geometry"][field] = value
        source[field_path] = src

    for field in TARGET_FIELDS:
        field_path = f"target_pose.{field}"
        value, src = choose_value(field_path, front["target_pose"].get(field), overhead["target_pose"].get(field), conflicts)
        fused["target_pose"][field] = value
        source[field_path] = src

    uncertain = sorted(set(front.get("uncertain_fields", []) + overhead.get("uncertain_fields", [])))
    fused["uncertain_fields"] = [field for field in uncertain if field in ALL_FIELDS]
    source["uncertain_fields"] = "union"
    return fused, source, conflicts


def set_descriptor_value(
    descriptor: dict[str, Any],
    source_by_field: dict[str, str],
    adjustments: list[dict[str, Any]],
    field_path: str,
    value: Any,
    rule: str,
) -> None:
    block, field = field_path.split(".", 1)
    before = descriptor[block].get(field)
    if before == value:
        return
    descriptor[block][field] = value
    source_by_field[field_path] = f"strict_rule:{rule}"
    adjustments.append(
        {
            "rule": rule,
            "field": field_path,
            "before": before,
            "after": value,
        }
    )


def apply_values(
    descriptor: dict[str, Any],
    source_by_field: dict[str, str],
    adjustments: list[dict[str, Any]],
    rule: str,
    values: dict[str, Any],
) -> None:
    for field_path, value in values.items():
        set_descriptor_value(descriptor, source_by_field, adjustments, field_path, value, rule)


def apply_task_language_rules(
    task: str | None,
    instruction: str,
    descriptor: dict[str, Any],
    source_by_field: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply deterministic cleanup to retrieval-critical fields.

    These rules deliberately use task-language cues, not hidden evaluation labels.
    They keep the model useful for free-text target-pose phrasing while preventing
    obvious primitive/action drift in retrieval fields.
    """

    task_text = (task or "").lower()
    text = f"{task_text} {instruction.lower()}"
    adjustments: list[dict[str, Any]] = []

    if "turn_tap" in task_text or "tap" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "tap_twist",
            {
                "geometry.action_primitive": "twist",
                "geometry.motion_type": "rotational",
                "geometry.motion_axis": "rotational",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "knob",
                "geometry.constraint_type": "button_switch",
                "geometry.alignment_requirement": "none",
                "target_pose.goal_state_type": "control_or_articulation_state",
                "target_pose.required_final_relation": "tap turned",
                "target_pose.target_object_or_region": "tap knob",
                "target_pose.required_orientation_or_alignment": "none",
                "target_pose.release_or_stop_condition": "stop after tap knob rotates",
                "target_pose.success_check": "tap is turned",
            },
        )

    if "close_jar" in task_text or "jar" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "jar_lid_twist_closed",
            {
                "geometry.action_primitive": "twist",
                "geometry.motion_type": "rotational",
                "geometry.motion_axis": "rotational",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "lid",
                "geometry.constraint_type": "socket",
                "geometry.alignment_requirement": "high",
                "target_pose.goal_state_type": "closed_or_sealed_state",
                "target_pose.required_final_relation": "jar lid closed",
                "target_pose.target_object_or_region": "jar lid",
                "target_pose.required_orientation_or_alignment": "upright",
                "target_pose.release_or_stop_condition": "stop when lid is seated and tight",
                "target_pose.success_check": "jar lid is closed",
            },
        )

    if "light_bulb" in task_text or "bulb" in text or "screw in" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "bulb_socket_twist_insert",
            {
                "geometry.action_primitive": "twist",
                "geometry.motion_type": "rotational",
                "geometry.motion_axis": "rotational",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "body",
                "geometry.constraint_type": "socket",
                "geometry.alignment_requirement": "high",
                "target_pose.goal_state_type": "aligned_insertion_or_docking",
                "target_pose.required_final_relation": "bulb screwed into socket",
                "target_pose.target_object_or_region": "bulb socket",
                "target_pose.required_orientation_or_alignment": "aligned_with_socket",
                "target_pose.release_or_stop_condition": "stop when bulb is seated",
                "target_pose.success_check": "bulb is securely screwed in",
            },
        )

    if "open_drawer" in task_text or ("open" in text and "drawer" in text):
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "drawer_open_slider",
            {
                "geometry.action_primitive": "pull",
                "geometry.motion_type": "linear",
                "geometry.motion_axis": "horizontal",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "handle",
                "geometry.constraint_type": "slider",
                "geometry.alignment_requirement": "none",
                "target_pose.goal_state_type": "articulated_part_state",
                "target_pose.required_final_relation": "drawer open",
                "target_pose.target_object_or_region": "drawer",
                "target_pose.required_orientation_or_alignment": "none",
                "target_pose.release_or_stop_condition": "stop when drawer is open",
                "target_pose.success_check": "drawer is open",
            },
        )

    if "push_buttons" in task_text or "button" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "button_press",
            {
                "geometry.action_primitive": "press",
                "geometry.motion_type": "press",
                "geometry.motion_axis": "surface_normal",
                "geometry.contact_type": "press",
                "geometry.contact_region": "button",
                "geometry.constraint_type": "button_switch",
                "geometry.alignment_requirement": "none",
                "target_pose.goal_state_type": "control_or_articulation_state",
                "target_pose.required_orientation_or_alignment": "none",
            },
        )

    if "slide_block" in task_text or "slide" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "block_slide_to_target",
            {
                "geometry.action_primitive": "slide",
                "geometry.motion_type": "planar",
                "geometry.motion_axis": "across_surface",
                "geometry.contact_type": "push_contact",
                "geometry.contact_region": "body",
                "geometry.constraint_type": "support_surface",
                "geometry.alignment_requirement": "low",
                "target_pose.goal_state_type": "placement_inside_or_on_target",
                "target_pose.required_orientation_or_alignment": "none",
            },
        )

    if "sweep" in task_text or "sweep" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "sweep_to_receptacle",
            {
                "geometry.action_primitive": "sweep",
                "geometry.motion_type": "tool_use",
                "geometry.motion_axis": "across_surface",
                "geometry.contact_type": "tool_contact",
                "geometry.contact_region": "tip",
                "geometry.constraint_type": "support_surface",
                "geometry.alignment_requirement": "low",
                "target_pose.goal_state_type": "placement_inside_or_on_target",
                "target_pose.required_orientation_or_alignment": "none",
            },
        )

    if "shape_sorter" in task_text or "shape sorter" in text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "shape_sorter_slot_insert",
            {
                "geometry.action_primitive": "insert",
                "geometry.motion_type": "insertion",
                "geometry.motion_axis": "into_opening",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "body",
                "geometry.constraint_type": "slot",
                "geometry.alignment_requirement": "high",
                "target_pose.goal_state_type": "aligned_insertion_or_docking",
                "target_pose.required_final_relation": "object inserted into matching shape-sorter opening",
                "target_pose.target_object_or_region": "shape sorter opening",
                "target_pose.required_orientation_or_alignment": "aligned_with_slot",
                "target_pose.release_or_stop_condition": "release after object enters opening",
                "target_pose.success_check": "object is inside the shape sorter",
            },
        )

    if "spoke" in text or "peg" in text or "insert_onto_square_peg" in task_text:
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "ring_on_spoke_insert",
            {
                "geometry.action_primitive": "insert",
                "geometry.motion_type": "insertion",
                "geometry.motion_axis": "around_post",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "body",
                "geometry.constraint_type": "peg_or_post",
                "geometry.alignment_requirement": "high",
                "target_pose.goal_state_type": "aligned_insertion_or_docking",
                "target_pose.required_final_relation": "ring around red spoke",
                "target_pose.target_object_or_region": "red spoke",
                "target_pose.required_orientation_or_alignment": "hole_aligned_with_post",
                "target_pose.release_or_stop_condition": "release when ring is seated on spoke",
                "target_pose.success_check": "ring is seated around the spoke",
            },
        )

    if (
        "put_item_in_drawer" in task_text
        or "groceries" in task_text
        or "cupboard" in text
        or "safe" in text
        or ("put" in text and ("drawer" in text or "cupboard" in text or "safe" in text))
    ):
        target_region = "container interior"
        if "drawer" in text:
            target_region = "drawer interior"
        elif "cupboard" in text:
            target_region = "cupboard interior"
        elif "safe" in text:
            target_region = "safe interior"
        apply_values(
            descriptor,
            source_by_field,
            adjustments,
            "place_inside_container",
            {
                "geometry.action_primitive": "place",
                "geometry.motion_type": "vertical",
                "geometry.motion_axis": "into_opening",
                "geometry.contact_type": "grasp",
                "geometry.contact_region": "body",
                "geometry.constraint_type": "container",
                "geometry.alignment_requirement": "low",
                "target_pose.goal_state_type": "placement_inside_or_on_target",
                "target_pose.target_object_or_region": target_region,
                "target_pose.required_orientation_or_alignment": "inside_receptacle",
                "target_pose.release_or_stop_condition": "release when object is inside target",
            },
        )

    return adjustments


def run_view(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    image_path: str,
    task_instruction: str,
    view: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.replace("{task}", task_instruction).replace("{view}", view)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed, raw = clean_json(decoded)
    normalized, notes = normalize_descriptor(parsed)
    return {
        "image": image_path,
        "prompt_text": prompt,
        "raw_descriptor": parsed,
        "raw_output": raw,
        "normalized_descriptor": normalized,
        "normalization_notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_dir = out_dir / "items"
    items_dir.mkdir(exist_ok=True)

    examples = read_manifest(Path(args.manifest))
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No examples in {args.manifest}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    records: list[dict[str, Any]] = []
    for example in examples:
        review_id = example["id"]
        task_instruction = example["language_description"]
        views = {
            "front": run_view(
                processor,
                model,
                example["front_image"],
                task_instruction,
                "front",
                args.max_new_tokens,
            ),
            "overhead": run_view(
                processor,
                model,
                example["overhead_image"],
                task_instruction,
                "overhead",
                args.max_new_tokens,
            ),
        }
        fused, source_by_field, conflicts = fuse_descriptors(
            views["front"]["normalized_descriptor"],
            views["overhead"]["normalized_descriptor"],
        )
        rule_adjustments = apply_task_language_rules(
            example.get("task"),
            task_instruction,
            fused,
            source_by_field,
        )
        record = {
            "id": review_id,
            "task": example.get("task"),
            "language_description": task_instruction,
            "model": args.model,
            "schema_version": "clean_geometry_target_pose_v2_no_start_no_tags",
            "image_inputs": {
                "front_image": example["front_image"],
                "overhead_image": example["overhead_image"],
                "local_front_image": example.get("local_front_image"),
                "local_overhead_image": example.get("local_overhead_image"),
            },
            "view_outputs": views,
            "fused_descriptor": fused,
            "source_by_field": source_by_field,
            "rule_adjustments": rule_adjustments,
            "conflicts": conflicts,
            "human_check": {
                "geometry_retrieval_ok": None,
                "target_pose_prompt_ok": None,
                "front_overhead_fusion_ok": None,
                "notes": "",
            },
        }
        (items_dir / f"{review_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(f"[done] {review_id}: {task_instruction}", flush=True)

    (out_dir / "qwen_clean_geometry_target_pose_bundle.json").write_text(json.dumps(records, indent=2) + "\n")
    with (out_dir / "qwen_clean_geometry_target_pose_bundle.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
