#!/usr/bin/env python3
"""Strict postprocessing rules for Qwen retrieval-geometry descriptors."""

from __future__ import annotations

import re
from typing import Any


SCHEMA: dict[str, set[str]] = {
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
        "linear_push",
        "linear_pull",
        "vertical_lift",
        "lift_then_release",
        "planar_slide",
        "rotational_turn",
        "hinge_swing",
        "press_down",
        "align_then_insert",
        "align_then_dock",
        "sweep_across_surface",
        "scoop_under_then_lift",
        "pour_tilt",
        "none",
        "unknown",
    },
    "target_pose_type": {
        "none",
        "support_surface",
        "open_receptacle",
        "slot",
        "peg_or_post",
        "hole_or_socket",
        "base_or_dock",
        "shelf",
        "rack",
        "button_or_switch",
        "hinge_or_joint",
        "screw_or_twist_socket",
        "pour_target",
        "free_space_target",
        "unknown",
    },
    "manipulated_object_family": {
        "rigid_compact_object",
        "thin_rigid_object",
        "elongated_tool",
        "container_or_lid",
        "round_or_cylindrical_object",
        "ring_or_hollow_object",
        "flat_deformable_object",
        "deformable_rope",
        "button_or_switch",
        "door_drawer_or_lid",
        "tool_or_utensil",
        "unknown",
    },
    "target_object_family": {
        "flat_support_surface",
        "open_receptacle",
        "slot_or_hole",
        "peg_or_post",
        "dock_or_base",
        "shelf_or_rack",
        "button_panel",
        "hinged_articulation",
        "drawer_or_container",
        "plant_or_pour_target",
        "free_space",
        "none",
        "unknown",
    },
    "manipulated_part": {
        "body",
        "handle",
        "rim",
        "lid",
        "button_top",
        "knob",
        "edge",
        "tip",
        "opening",
        "hole",
        "cord",
        "surface",
        "none",
        "unknown",
    },
    "target_part": {
        "top_surface",
        "inside_volume",
        "opening",
        "slot",
        "hole",
        "peg_post",
        "dock_cradle",
        "shelf_surface",
        "button_top",
        "handle",
        "hinge_lid",
        "socket",
        "plant_soil",
        "none",
        "unknown",
    },
    "articulation_type": {
        "none",
        "hinge",
        "slider",
        "button",
        "switch",
        "knob",
        "lid",
        "drawer",
        "door",
        "socket",
        "unknown",
    },
    "required_alignment": {
        "none",
        "low",
        "medium",
        "high",
        "unknown",
    },
}

SCALAR_FIELDS = list(SCHEMA)
ALL_FIELDS = SCALAR_FIELDS + ["geometry_tags", "uncertain_fields"]


def _norm_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _has(text: str, *terms: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _normalize_tags(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        value = [value]
    tags: list[str] = []
    for item in value:
        tag = re.sub(r"[^a-z0-9]+", "_", str(item).strip().lower()).strip("_")
        if tag and tag != "unknown" and tag not in tags:
            tags.append(tag)
    return tags[:10]


def _base_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    out = {field: descriptor.get(field, "unknown") for field in SCALAR_FIELDS}
    out["geometry_tags"] = _normalize_tags(descriptor.get("geometry_tags"))
    out["uncertain_fields"] = [
        field for field in _normalize_tags(descriptor.get("uncertain_fields")) if field in ALL_FIELDS
    ]
    for field, allowed in SCHEMA.items():
        if out[field] not in allowed:
            out[field] = "unknown"
    return out


def _merge_tags(existing: list[str], added: list[str]) -> list[str]:
    tags = list(existing)
    for tag in added:
        if tag not in tags:
            tags.append(tag)
    return tags[:10]


def _apply_rule(
    descriptor: dict[str, Any],
    adjustments: list[dict[str, Any]],
    rule: str,
    reason: str,
    overrides: dict[str, Any],
    tags: list[str] | None = None,
) -> None:
    for field, value in overrides.items():
        if field not in SCALAR_FIELDS:
            continue
        if value not in SCHEMA[field]:
            raise ValueError(f"Rule {rule} tried to set invalid {field}={value}")
        before = descriptor.get(field, "unknown")
        if before != value:
            descriptor[field] = value
            adjustments.append(
                {
                    "rule": rule,
                    "field": field,
                    "before": before,
                    "after": value,
                    "reason": reason,
                }
            )
    if tags:
        before_tags = descriptor.get("geometry_tags", [])
        after_tags = _merge_tags(before_tags, tags)
        if after_tags != before_tags:
            descriptor["geometry_tags"] = after_tags
            adjustments.append(
                {
                    "rule": rule,
                    "field": "geometry_tags",
                    "before": before_tags,
                    "after": after_tags,
                    "reason": reason,
                }
            )


def apply_strict_rules(
    task: str | None,
    language_description: str | None,
    descriptor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply deterministic language/task rules after Qwen normalization.

    The rules intentionally override Qwen only for fields that are structurally
    determined by the task language, such as button press, drawer slide, peg
    insertion, or shelf placement.
    """

    result = _base_descriptor(descriptor)
    adjustments: list[dict[str, Any]] = []
    text = _norm_text(task, language_description)

    if _has(text, "button", "buzzer") or "lamp_on" in text or "lamp_off" in text:
        _apply_rule(
            result,
            adjustments,
            "button_or_switch_press",
            "button/switch language fixes primitive and contact target",
            {
                "action_primitive": "press",
                "motion_type": "press_down",
                "target_pose_type": "button_or_switch",
                "manipulated_object_family": "button_or_switch",
                "target_object_family": "button_panel",
                "manipulated_part": "button_top",
                "target_part": "button_top",
                "articulation_type": "button",
                "required_alignment": "high",
            },
            ["button"],
        )

    if _has(text, "turn", "twist", "rotate") and _has(text, "tap", "knob", "dial"):
        _apply_rule(
            result,
            adjustments,
            "knob_or_tap_twist",
            "tap/knob turn language should retrieve rotational manipulation demos",
            {
                "action_primitive": "twist",
                "motion_type": "rotational_turn",
                "target_pose_type": "screw_or_twist_socket",
                "manipulated_object_family": "rigid_compact_object",
                "target_object_family": "hinged_articulation",
                "manipulated_part": "knob",
                "target_part": "handle",
                "articulation_type": "knob",
                "required_alignment": "medium",
            },
            ["tap", "knob", "twist"],
        )

    if _has(text, "screw") and _has(text, "light", "bulb"):
        _apply_rule(
            result,
            adjustments,
            "screw_light_bulb",
            "screw-in bulb language is a rotational socket insertion",
            {
                "action_primitive": "twist",
                "motion_type": "rotational_turn",
                "target_pose_type": "screw_or_twist_socket",
                "manipulated_object_family": "round_or_cylindrical_object",
                "target_object_family": "slot_or_hole",
                "manipulated_part": "body",
                "target_part": "socket",
                "articulation_type": "socket",
                "required_alignment": "high",
            },
            ["light_bulb", "socket", "twist"],
        )

    if _has(text, "close") and _has(text, "jar"):
        _apply_rule(
            result,
            adjustments,
            "close_jar_lid",
            "closing a jar should retrieve lid/container closing, not slot placement",
            {
                "action_primitive": "close",
                "motion_type": "rotational_turn",
                "target_pose_type": "screw_or_twist_socket",
                "manipulated_object_family": "container_or_lid",
                "target_object_family": "drawer_or_container",
                "manipulated_part": "lid",
                "target_part": "opening",
                "articulation_type": "lid",
                "required_alignment": "high",
            },
            ["jar", "lid", "rim"],
        )

    if _has(text, "open") and _has(text, "drawer"):
        _apply_rule(
            result,
            adjustments,
            "open_drawer_slider",
            "drawer opening is a linear slider pull",
            {
                "action_primitive": "open",
                "motion_type": "linear_pull",
                "target_pose_type": "none",
                "manipulated_object_family": "door_drawer_or_lid",
                "target_object_family": "drawer_or_container",
                "manipulated_part": "handle",
                "target_part": "handle",
                "articulation_type": "drawer",
                "required_alignment": "medium",
            },
            ["drawer", "handle", "slider"],
        )

    if _has(text, "close") and _has(text, "drawer"):
        _apply_rule(
            result,
            adjustments,
            "close_drawer_slider",
            "drawer closing is a linear slider push",
            {
                "action_primitive": "close",
                "motion_type": "linear_push",
                "target_pose_type": "none",
                "manipulated_object_family": "door_drawer_or_lid",
                "target_object_family": "drawer_or_container",
                "manipulated_part": "handle",
                "target_part": "handle",
                "articulation_type": "drawer",
                "required_alignment": "medium",
            },
            ["drawer", "handle", "slider"],
        )

    if _has(text, "open") and not _has(text, "drawer"):
        _apply_rule(
            result,
            adjustments,
            "generic_hinged_open",
            "generic open task usually retrieves hinge/door/lid opening",
            {
                "action_primitive": "open",
                "motion_type": "hinge_swing",
                "target_pose_type": "hinge_or_joint",
                "manipulated_object_family": "door_drawer_or_lid",
                "target_object_family": "hinged_articulation",
                "articulation_type": "hinge",
                "required_alignment": "medium",
            },
            ["hinge", "open"],
        )

    if _has(text, "close") and not _has(text, "drawer", "jar"):
        _apply_rule(
            result,
            adjustments,
            "generic_hinged_close",
            "generic close task usually retrieves hinge/door/lid closing",
            {
                "action_primitive": "close",
                "motion_type": "hinge_swing",
                "target_pose_type": "hinge_or_joint",
                "manipulated_object_family": "door_drawer_or_lid",
                "target_object_family": "hinged_articulation",
                "articulation_type": "hinge",
                "required_alignment": "medium",
            },
            ["hinge", "close"],
        )

    if _has(text, "slide") and _has(text, "target", "green", "red", "blue", "color"):
        _apply_rule(
            result,
            adjustments,
            "slide_to_spatial_target",
            "sliding to a colored target is planar sliding to a target region",
            {
                "action_primitive": "slide",
                "motion_type": "planar_slide",
                "target_pose_type": "free_space_target",
                "manipulated_object_family": "rigid_compact_object",
                "target_object_family": "free_space",
                "manipulated_part": "body",
                "target_part": "top_surface",
                "articulation_type": "none",
                "required_alignment": "medium",
            },
            ["block", "target", "slide"],
        )

    if _has(text, "sweep") and _has(text, "dustpan"):
        _apply_rule(
            result,
            adjustments,
            "sweep_to_dustpan",
            "sweeping to a dustpan is tool sweeping toward an open receptacle",
            {
                "action_primitive": "sweep",
                "motion_type": "sweep_across_surface",
                "target_pose_type": "open_receptacle",
                "manipulated_object_family": "elongated_tool",
                "target_object_family": "open_receptacle",
                "manipulated_part": "handle",
                "target_part": "inside_volume",
                "articulation_type": "none",
                "required_alignment": "medium",
            },
            ["sweep", "dirt", "dustpan"],
        )

    if _has(text, "water") and _has(text, "plant"):
        _apply_rule(
            result,
            adjustments,
            "water_plant_pour",
            "watering plants is a pour primitive toward plant soil",
            {
                "action_primitive": "pour",
                "motion_type": "pour_tilt",
                "target_pose_type": "pour_target",
                "manipulated_object_family": "container_or_lid",
                "target_object_family": "plant_or_pour_target",
                "manipulated_part": "handle",
                "target_part": "plant_soil",
                "articulation_type": "none",
                "required_alignment": "medium",
            },
            ["watering_can", "plant", "pour"],
        )

    if _has(text, "put") and _has(text, "in") and _has(text, "drawer"):
        _apply_rule(
            result,
            adjustments,
            "place_in_drawer",
            "putting an item in an open drawer is placement into a receptacle, not peg insertion",
            {
                "action_primitive": "place",
                "motion_type": "lift_then_release",
                "target_pose_type": "open_receptacle",
                "manipulated_object_family": "rigid_compact_object",
                "target_object_family": "drawer_or_container",
                "manipulated_part": "body",
                "target_part": "inside_volume",
                "articulation_type": "drawer",
                "required_alignment": "medium",
            },
            ["drawer", "inside_volume", "place"],
        )

    if _has(text, "put") and _has(text, "in") and _has(text, "bin"):
        _apply_rule(
            result,
            adjustments,
            "place_in_bin",
            "putting rubbish in a bin is placement into an open receptacle",
            {
                "action_primitive": "place",
                "motion_type": "lift_then_release",
                "target_pose_type": "open_receptacle",
                "target_object_family": "open_receptacle",
                "target_part": "inside_volume",
                "articulation_type": "none",
                "required_alignment": "medium",
            },
            ["bin", "inside_volume"],
        )

    if _has(text, "put") and _has(text, "in") and _has(text, "cupboard", "safe"):
        _apply_rule(
            result,
            adjustments,
            "place_on_internal_shelf",
            "cupboard/safe placement targets an internal shelf/support surface",
            {
                "action_primitive": "place",
                "motion_type": "lift_then_release",
                "target_pose_type": "shelf",
                "target_object_family": "shelf_or_rack",
                "manipulated_part": "body",
                "target_part": "shelf_surface",
                "articulation_type": "none",
                "required_alignment": "high",
            },
            ["shelf"],
        )

    if _has(text, "put") and _has(text, "on") and _has(text, "peg", "spoke", "stand"):
        _apply_rule(
            result,
            adjustments,
            "insert_onto_peg_or_post",
            "putting a ring/roll onto a peg/post/spoke requires aligned insertion",
            {
                "action_primitive": "insert",
                "motion_type": "align_then_insert",
                "target_pose_type": "peg_or_post",
                "manipulated_object_family": "ring_or_hollow_object",
                "target_object_family": "peg_or_post",
                "manipulated_part": "opening",
                "target_part": "peg_post",
                "articulation_type": "none",
                "required_alignment": "high",
            },
            ["peg", "post", "ring"],
        )

    if _has(text, "put") and _has(text, "in") and _has(text, "shape", "sorter"):
        _apply_rule(
            result,
            adjustments,
            "insert_shape_into_sorter",
            "shape-sorter task requires aligned insertion into a slot/opening",
            {
                "action_primitive": "insert",
                "motion_type": "align_then_insert",
                "target_pose_type": "slot",
                "manipulated_object_family": "rigid_compact_object",
                "target_object_family": "slot_or_hole",
                "manipulated_part": "body",
                "target_part": "slot",
                "articulation_type": "none",
                "required_alignment": "high",
            },
            ["shape_sorter", "slot"],
        )

    if _has(text, "put") and _has(text, "on") and _has(text, "board", "shelf", "rack"):
        target_pose = "rack" if _has(text, "rack") else "shelf" if _has(text, "shelf") else "support_surface"
        target_family = "shelf_or_rack" if target_pose in {"rack", "shelf"} else "flat_support_surface"
        target_part = "shelf_surface" if target_pose in {"rack", "shelf"} else "top_surface"
        _apply_rule(
            result,
            adjustments,
            "place_on_support",
            "putting an object on a support uses placement on a support surface",
            {
                "action_primitive": "place",
                "motion_type": "lift_then_release",
                "target_pose_type": target_pose,
                "target_object_family": target_family,
                "manipulated_part": "body",
                "target_part": target_part,
                "articulation_type": "none",
                "required_alignment": "low",
            },
            [target_pose],
        )

    if _has(text, "take", "remove", "unplug") and _has(text, "out"):
        _apply_rule(
            result,
            adjustments,
            "remove_from_socket_or_container",
            "take/remove/unplug out language should retrieve linear removal",
            {
                "action_primitive": "remove",
                "motion_type": "linear_pull",
                "target_pose_type": "none",
                "required_alignment": "medium",
            },
            ["remove"],
        )

    if _has(text, "usb", "charger", "plug"):
        _apply_rule(
            result,
            adjustments,
            "plug_or_usb_socket",
            "USB/charger/plug language fixes socket target geometry",
            {
                "target_object_family": "slot_or_hole",
                "manipulated_part": "body",
                "target_part": "socket",
                "articulation_type": "socket",
                "required_alignment": "high",
            },
            ["plug", "socket"],
        )

    result["uncertain_fields"] = [
        field for field in result.get("uncertain_fields", []) if field in ALL_FIELDS
    ]
    return result, adjustments
