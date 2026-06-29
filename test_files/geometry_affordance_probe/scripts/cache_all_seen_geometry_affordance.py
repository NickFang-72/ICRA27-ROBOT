#!/usr/bin/env python3
"""Cache geometry and affordance descriptors for all AGNOSTOS seen demos.

This is the full-scale version of the geometry/contact-hint pilot. It builds a
manifest from the seen-task train JSON, runs Qwen2.5-VL for primitive geometry,
runs RoboPoint for optional contact points, normalizes outputs into review
fields, and writes a progress file that can be watched during long CAIR runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


GEOMETRY_PROMPT = """You are extracting a primitive manipulation-geometry descriptor for robot demo retrieval.
Return ONLY valid JSON. Do not include markdown.

Goal:
Describe the object, part, contact region, action primitive, and mechanical constraint needed for the task.
Use only compact primitive labels. Do not describe camera-relative directions such as left, right, front, back, or facing upward.
Only use coarse motion geometry: horizontal, vertical, rotational, into_opening, across_surface, surface_normal, none, unknown.
If unsure, use "unknown" rather than guessing.

Schema:
{
  "manipulated_object": string,
  "object_category": "rigid_object|articulated_or_control|alignment_target|receptacle_or_support|elongated_or_tool|deformable|unknown",
  "primary_shape": "round_object|box_like|thin_flat_object|elongated_tool|button|peg|irregular|unknown",
  "target_part": "handle|lid|rim|knob|button_top|slot|hole|opening|body|edge|spout|socket|surface|unknown",
  "secondary_parts": [string],
  "action_primitive": "push|pull|press|twist|lift|place|insert|slide|sweep|stack|drag|scoop|pour|none|unknown",
  "motion_type": "linear|rotational|vertical|planar|insertion|none|unknown",
  "motion_axis": "horizontal|vertical|rotational|into_opening|across_surface|surface_normal|none|unknown",
  "contact_type": "grasp|pinch|press|surface_contact|tool_contact|none|unknown",
  "contact_region": string,
  "constraint_type": "slot|hole|container|joint|support_surface|surface_target|free_space|none|unknown",
  "alignment_requirement": "none|low|medium|high|unknown",
  "state": "open|closed|attached|detached|inside|on_surface|free|unknown",
  "geometry_tags": [string],
  "execution_clearance_hint": "none|open_path|narrow_path|swing_path|requires_lift|requires_slide_under|unknown"
}

Task instruction: {task}
Use the current observation images and task instruction.
Return the descriptor JSON only.
"""


QUESTION_TEMPLATE = """<image>
Task instruction: {task}
Primitive action: {action_primitive}
Contact mode: {contact_mode}
Target object: {target_object}
Target part: {target_part}

For robot manipulation, identify the best visible contact points for accomplishing this task.
If contact_mode is single_contact, return one normalized image point.
If contact_mode is grasp_pair, return two normalized image points for the two parallel-gripper finger contacts when visible.
If the contact is not visible or unreliable, return no points and say none.
Use normalized tuples [(x1, y1), ...] where x and y are between 0 and 1. After the list, add a short phrase naming the contact region only."""


CONTACT_HINT_SCHEMA = {
    "contact_mode": "single_contact | grasp_pair | none",
    "source_view": "front_rgb_initial",
    "target_object": "Object named by the primitive geometry descriptor",
    "target_part": "Part/contact region named by the primitive geometry descriptor",
    "points_2d_normalized": "RoboPoint normalized image points, if produced",
    "contact_region_text": "Short RoboPoint phrase naming the contact region",
    "raw_robopoint_text": "Original RoboPoint answer",
    "source_role": "keypoint_contact_hint_only",
}


TASK_RULES: dict[str, dict[str, Any]] = {
    "turn_tap": {
        "object": "tap",
        "features": ["tap", "handle", "knob", "rotational_axis", "cylindrical"],
        "affordance": {
            "grasp_affordance": "knob_grasp",
            "contact_affordance": "rotate_part",
            "motion_affordance": "rotate",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "tap_handle_or_knob",
            "failure_sensitive_property": "wrong_axis",
        },
    },
    "close_jar": {
        "object": "jar",
        "features": ["jar", "round", "cylindrical", "hollow", "lid", "rim", "top_opening"],
        "affordance": {
            "grasp_affordance": "rim_grasp",
            "contact_affordance": "rotate_part",
            "motion_affordance": "twist",
            "containment_affordance": "closed_container",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "lid_or_rim",
            "failure_sensitive_property": "wrong_axis",
        },
    },
    "light_bulb_in": {
        "object": "light_bulb",
        "features": ["light_bulb", "round", "bulb", "threaded_base", "socket", "rotational_axis", "alignment_sensitive"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert_then_twist",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "bulb_body_or_base",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        },
    },
    "slide_block_to_color_target": {
        "object": "block",
        "features": ["block", "rectangular", "flat_faces", "solid", "target_region"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "push_surface",
            "motion_affordance": "slide",
            "required_contact_region": "block_side_or_top",
            "failure_sensitive_property": "overshoot_or_wrong_direction",
        },
    },
    "sweep_to_dustpan_of_size": {
        "object": "dustpan",
        "features": ["dustpan", "open_container", "thin_edge", "flat_floor", "target_region"],
        "affordance": {
            "grasp_affordance": "handle_grasp",
            "contact_affordance": "push_surface",
            "motion_affordance": "push",
            "containment_affordance": "receptacle",
            "required_contact_region": "sweeping_tool_or_object_side",
            "failure_sensitive_property": "missed_receptacle",
        },
    },
    "push_buttons": {
        "object": "button",
        "features": ["button", "round", "small", "raised_surface", "flat_top"],
        "affordance": {
            "grasp_affordance": "none",
            "contact_affordance": "push_surface",
            "motion_affordance": "push",
            "required_contact_region": "button_top",
            "precision_requirement": "high",
            "force_requirement": "low",
            "failure_sensitive_property": "wrong_button",
        },
    },
    "put_groceries_in_cupboard": {
        "object": "groceries_and_cupboard",
        "features": ["cupboard", "box_like", "hollow", "shelf", "open_container"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "lift_then_insert",
            "support_affordance": "can_support",
            "containment_affordance": "receptacle",
            "required_contact_region": "object_body",
            "failure_sensitive_property": "collision_with_shelf",
        },
    },
    "put_money_in_safe": {
        "object": "money_and_safe",
        "features": ["safe", "rectangular", "hollow", "slot", "front_opening"],
        "affordance": {
            "grasp_affordance": "pinch",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "slot",
            "required_contact_region": "thin_object_edge",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        },
    },
    "place_shape_in_shape_sorter": {
        "object": "shape_and_shape_sorter",
        "features": ["shape_sorter", "shape_profile", "slot", "hole", "matching_geometry", "alignment_sensitive"],
        "affordance": {
            "grasp_affordance": "edge_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "slot",
            "required_contact_region": "shape_body",
            "precision_requirement": "high",
            "failure_sensitive_property": "wrong_shape_orientation",
        },
    },
    "put_item_in_drawer": {
        "object": "item_and_drawer",
        "features": ["drawer", "rectangular", "hollow", "open_container", "sliding_axis", "handle"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "lift_then_place",
            "containment_affordance": "open_container",
            "articulation_affordance": "drawer_slide",
            "required_contact_region": "object_body",
            "failure_sensitive_property": "collision_with_drawer",
        },
    },
    "insert_onto_square_peg": {
        "object": "shape_and_square_peg",
        "features": ["peg", "square_peg", "hole", "alignment_sensitive", "insertable_part"],
        "affordance": {
            "grasp_affordance": "edge_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "hole",
            "required_contact_region": "object_body",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        },
    },
    "open_drawer": {
        "object": "drawer",
        "features": ["drawer", "rectangular", "hollow", "handle", "sliding_axis", "front_face"],
        "affordance": {
            "grasp_affordance": "handle_grasp",
            "contact_affordance": "pull_handle",
            "motion_affordance": "pull",
            "containment_affordance": "open_container",
            "articulation_affordance": "drawer_slide",
            "required_contact_region": "handle",
            "failure_sensitive_property": "wrong_grasp",
        },
    },
    "reach_and_drag": {
        "object": "stick_and_cube",
        "features": ["stick", "elongated_tool", "cube", "dragging", "target_region", "contact_edge"],
        "affordance": {
            "grasp_affordance": "handle_grasp",
            "contact_affordance": "drag_object",
            "motion_affordance": "drag",
            "required_contact_region": "stick_or_cube_contact_edge",
            "failure_sensitive_property": "lost_contact",
        },
    },
    "stack_blocks": {
        "object": "blocks",
        "features": ["blocks", "rectangular", "flat_faces", "stackable", "support_surface", "alignment_sensitive"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "stack",
            "support_affordance": "can_support",
            "required_contact_region": "block_body",
            "precision_requirement": "high",
            "failure_sensitive_property": "unstable_stack",
        },
    },
    "place_cups": {
        "object": "cups_and_holder",
        "features": ["cups", "hollow", "rim", "holder_spokes", "alignment_sensitive", "can_hang"],
        "affordance": {
            "grasp_affordance": "rim_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "place",
            "support_affordance": "can_hang",
            "containment_affordance": "holder_spoke",
            "required_contact_region": "cup_body_or_rim",
            "precision_requirement": "high",
            "failure_sensitive_property": "missed_holder",
        },
    },
    "place_wine_at_rack_location": {
        "object": "wine_and_rack",
        "features": ["wine_bottle", "cylindrical", "rack", "slot", "target_location", "alignment_sensitive"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "place",
            "containment_affordance": "rack_slot",
            "required_contact_region": "bottle_body_or_neck",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        },
    },
    "meat_off_grill": {
        "object": "meat_and_grill",
        "features": ["meat", "irregular", "flat_grill", "support_surface", "lift_off_surface"],
        "affordance": {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_top",
            "motion_affordance": "lift",
            "support_affordance": "none",
            "required_contact_region": "meat_body",
            "failure_sensitive_property": "wrong_grasp",
        },
    },
    "stack_cups": {
        "object": "cups",
        "features": ["cups", "hollow", "rim", "open_container", "stackable", "nesting", "alignment_sensitive"],
        "affordance": {
            "grasp_affordance": "rim_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "stack",
            "containment_affordance": "open_container",
            "required_contact_region": "cup_body_or_rim",
            "precision_requirement": "high",
            "failure_sensitive_property": "unstable_stack",
        },
    },
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def task_from_id(demo_id: str) -> str:
    return demo_id.split("_episode", 1)[0]


def episode_path_from_images(images: list[str]) -> str | None:
    if not images:
        return None
    path = Path(images[0])
    if len(path.parents) >= 2:
        return str(path.parents[1])
    return None


def episode_id_from_episode_path(episode_path: str | None) -> int | None:
    if not episode_path:
        return None
    name = Path(episode_path).name
    if not name.startswith("episode"):
        return None
    try:
        return int(name.replace("episode", "", 1))
    except ValueError:
        return None


def frame_index_from_images(images: list[str]) -> int:
    if not images:
        return 10**9
    try:
        return int(Path(images[0]).stem)
    except ValueError:
        return 10**9


def clean_json(text: str) -> tuple[dict[str, Any], str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed, text
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed, text
            except Exception:
                pass
    return {"parse_error": True, "raw_text": text}, text


def unique(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item).strip().lower().replace(" ", "_")
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_points(text: str | None) -> list[list[float]]:
    if not text:
        return []
    points = []
    for x, y in re.findall(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text):
        try:
            points.append([float(x), float(y)])
        except ValueError:
            pass
    return points


def infer_geometry_key_features(task: str, geometry: dict[str, Any]) -> list[str]:
    if task in TASK_RULES:
        return unique(TASK_RULES[task]["features"])

    text = " ".join([task or "", json.dumps(geometry or {}, sort_keys=True)]).lower()
    vocab = {
        "round": ["round", "circular", "cylinder", "cylindrical", "sphere", "radial", "bulb", "jar", "cup"],
        "square": ["square"],
        "rectangular": ["rectangular", "box", "drawer", "safe", "cupboard", "block"],
        "flat": ["flat", "panel", "thin", "plate"],
        "elongated": ["elongated", "long", "handle", "stick", "tap"],
        "hollow": ["hollow", "opening", "container", "inside", "drawer", "cupboard", "jar", "safe", "dustpan", "hole", "slot", "cup"],
        "solid": ["solid", "block", "cube"],
        "handle": ["handle", "tap", "drawer"],
        "knob": ["knob", "button", "tap"],
        "lid": ["lid", "jar"],
        "rim": ["rim", "jar", "cup", "opening"],
        "hole": ["hole", "socket", "peg"],
        "slot": ["slot", "shape_sorter", "safe", "rack"],
        "hinge": ["hinge"],
        "sliding_axis": ["sliding_axis", "drawer", "slide"],
        "rotational_axis": ["axis", "rotate", "turn", "twist", "tap"],
        "open_container": ["open", "drawer", "cupboard", "dustpan", "cup"],
        "target_region": ["target", "button", "peg", "sorter", "rack"],
        "alignment_sensitive": ["insert", "peg", "socket", "shape_sorter", "light_bulb", "stack", "rack"],
    }
    features = [feature for feature, cues in vocab.items() if any(cue in text for cue in cues)]
    blocked = {"robot", "arm", "arms", "head", "gripper", "red", "green", "blue"}
    return [feature for feature in unique(features) if feature not in blocked]


def infer_manipulated_object(task: str) -> str:
    return str(TASK_RULES.get(task, {}).get("object", "unknown"))


def infer_action_primitive(task: str) -> str:
    if task == "push_buttons":
        return "press"
    if task == "sweep_to_dustpan_of_size":
        return "sweep"
    motion = str(TASK_RULES.get(task, {}).get("affordance", {}).get("motion_affordance", "unknown"))
    return {
        "rotate": "twist",
        "insert_then_twist": "insert",
        "lift_then_insert": "insert",
        "lift_then_place": "place",
        "lift_and_place": "place",
        "push": "push",
    }.get(motion, motion)


def infer_target_part(task: str) -> str:
    region = str(TASK_RULES.get(task, {}).get("affordance", {}).get("required_contact_region", "unknown"))
    if "body" in region:
        return "body"
    if "side" in region or "surface" in region:
        return "surface"
    if "handle" in region:
        return "handle"
    if "knob" in region:
        return "knob"
    if "rim" in region:
        return "rim"
    if "button" in region:
        return "button_top"
    if "edge" in region:
        return "edge"
    if "slot" in region:
        return "slot"
    if "hole" in region or "socket" in region:
        return "hole"
    if "opening" in region:
        return "opening"
    return region


def infer_contact_mode(task: str) -> str:
    action = infer_action_primitive(task)
    if action in {"push", "press", "slide", "sweep", "drag"}:
        return "single_contact"
    if action in {"pull", "twist", "lift", "place", "insert", "stack", "scoop", "pour"}:
        return "grasp_pair"
    return "none"


def infer_object_category(task: str, geometry: dict[str, Any], features: list[str]) -> str:
    text = " ".join([task, json.dumps(geometry or {}, sort_keys=True), " ".join(features)]).lower()
    if any(token in text for token in ["drawer", "door", "lid", "hinge", "knob", "tap", "switch", "button", "jar"]):
        return "articulated_or_control"
    if any(token in text for token in ["slot", "hole", "socket", "peg", "shape_sorter"]):
        return "alignment_target"
    if any(token in text for token in ["cupboard", "safe", "dustpan", "shelf", "rack", "container"]):
        return "receptacle_or_support"
    if any(token in text for token in ["spatula", "knife", "broom", "tool", "handle"]):
        return "elongated_or_tool"
    return "rigid_object"


def infer_primary_shape(task: str, geometry: dict[str, Any], features: list[str]) -> str:
    text = " ".join([task, json.dumps(geometry or {}, sort_keys=True), " ".join(features)]).lower()
    if task == "push_buttons":
        return "button"
    if task == "insert_onto_square_peg":
        return "peg"
    if task in {"close_jar", "light_bulb_in"}:
        return "round_object"
    if task in {"open_drawer", "put_item_in_drawer", "put_money_in_safe", "put_groceries_in_cupboard", "place_shape_in_shape_sorter", "slide_block_to_color_target"}:
        return "box_like"
    if task == "sweep_to_dustpan_of_size":
        return "thin_flat_object"
    if task == "turn_tap":
        return "elongated_tool"
    if any(token in text for token in ["button"]):
        return "button"
    if any(token in text for token in ["peg"]):
        return "peg"
    if any(token in text for token in ["drawer", "safe", "cupboard", "block", "box", "rectangular", "cube"]):
        return "box_like"
    if any(token in text for token in ["money", "thin", "flat", "plate"]):
        return "thin_flat_object"
    if any(token in text for token in ["tap", "handle", "spatula", "tool", "sweeping"]):
        return "elongated_tool"
    if any(token in text for token in ["jar", "bulb", "round", "circular", "cylinder", "cylindrical", "rim"]):
        return "round_object"
    if any(token in text for token in ["shape_sorter", "shape_profile", "irregular"]):
        return "irregular"
    return "unknown"


def normalize_state(value: Any) -> str:
    state = str(value or "").strip().lower().replace(" ", "_")
    if state in {"open", "closed", "attached", "detached", "inside", "on_surface", "free", "unknown"}:
        return state
    if "open" in state:
        return "open"
    if "closed" in state:
        return "closed"
    if "attached" in state or "grasp" in state or "held" in state:
        return "attached"
    if "inside" in state:
        return "inside"
    if "surface" in state:
        return "on_surface"
    if "free" in state:
        return "free"
    return "unknown"


def infer_contact_hints(task: str, points: list[list[float]], raw_text: str | None, vision_tower_note: str | None = None) -> dict[str, Any]:
    contact_mode = infer_contact_mode(task)
    if contact_mode == "single_contact":
        points = points[:1]
    elif contact_mode == "grasp_pair":
        points = points[:2]
    else:
        points = []
    hints = {
        "contact_mode": contact_mode,
        "source_view": "front_rgb_initial",
        "target_object": infer_manipulated_object(task),
        "target_part": infer_target_part(task),
        "points_2d_normalized": points,
        "contact_region_text": infer_target_part(task),
        "raw_robopoint_text": raw_text,
        "source_model": "RoboPoint",
        "source_role": "keypoint_contact_hint_only",
        "use_as": "final_prompt_contact_hint_only_not_retrieval_score",
    }
    if vision_tower_note:
        hints["vision_tower_note"] = vision_tower_note
    return hints


def infer_primitive_geometry(task: str, geometry: dict[str, Any]) -> dict[str, Any]:
    # The seen task label is the reliable source for the primitive. Qwen often
    # sees the contact shape correctly but confuses nearby primitives.
    action = str(infer_action_primitive(task) or geometry.get("action_primitive") or "unknown")
    features = infer_geometry_key_features(task, geometry)
    target_part = str(geometry.get("target_part") or infer_target_part(task) or "unknown")
    if target_part not in {"handle", "lid", "rim", "knob", "button_top", "slot", "hole", "opening", "body", "edge", "spout", "socket", "surface", "unknown"}:
        target_part = infer_target_part(task)
    axis = str(geometry.get("axis_geometry") or geometry.get("motion_axis") or "unknown")
    opening = str(geometry.get("opening_geometry") or geometry.get("constraint_type") or "unknown")

    if action in {"twist", "rotate"} or "rotat" in axis or "tilt" in axis:
        motion_type = "rotational"
        motion_axis = "rotational"
    elif action == "insert":
        motion_type = "insertion"
        motion_axis = "into_opening"
    elif action in {"place", "stack", "lift", "pour"}:
        motion_type = "vertical"
        motion_axis = "vertical"
    elif action in {"slide", "sweep", "drag", "scoop"}:
        motion_type = "planar"
        motion_axis = "across_surface"
    elif action == "press":
        motion_type = "linear"
        motion_axis = "surface_normal"
    elif action in {"push", "pull"}:
        motion_type = "linear"
        motion_axis = "horizontal"
    else:
        motion_type = str(geometry.get("motion_type") or "unknown")
        motion_axis = str(geometry.get("motion_axis") or "unknown")

    if action in {"twist", "rotate"}:
        constraint_type = "joint"
    elif any(token in features for token in ["hinge", "sliding_axis", "rotational_axis"]):
        constraint_type = "joint"
    elif "slot" in opening:
        constraint_type = "slot"
    elif "hole" in opening or "socket" in opening:
        constraint_type = "hole"
    elif "opening" in opening:
        constraint_type = "container"
    elif any(token in features for token in ["slot", "hole"]):
        constraint_type = "slot" if "slot" in features else "hole"
    elif any(token in features for token in ["open_container", "hollow"]):
        constraint_type = "container"
    else:
        constraint_type = "support_surface" if action in {"place", "stack"} else "none"

    if any(token in features for token in ["alignment_sensitive", "slot", "hole"]):
        alignment = "high"
    elif any(token in features for token in ["target_region", "rim"]):
        alignment = "medium"
    else:
        alignment = "low"

    contact_type = {
        "press": "press",
        "push": "surface_contact",
        "slide": "surface_contact",
        "sweep": "surface_contact",
        "drag": "surface_contact",
        "scoop": "tool_contact",
    }.get(action, "grasp" if action not in {"none", "unknown"} else "unknown")

    return {
        "manipulated_object": str(geometry.get("manipulated_object") or infer_manipulated_object(task)),
        "object_category": str(geometry.get("object_category") or infer_object_category(task, geometry, features)),
        "primary_shape": infer_primary_shape(task, geometry, features),
        "target_part": target_part,
        "secondary_parts": unique(geometry.get("secondary_parts") or geometry.get("part_geometry") or []),
        "action_primitive": action,
        "motion_type": motion_type,
        "motion_axis": motion_axis,
        "contact_type": contact_type,
        "contact_region": str(geometry.get("contact_region") or target_part),
        "constraint_type": constraint_type,
        "alignment_requirement": str(geometry.get("alignment_requirement") or alignment),
        "state": normalize_state(geometry.get("state") or geometry.get("pose_relation") or "unknown"),
        "geometry_tags": unique(features),
        "execution_clearance_hint": str(geometry.get("execution_clearance_hint") or geometry.get("clearance_geometry") or "unknown"),
    }


def geometry_path(demo: dict[str, Any]) -> Path:
    return Path(demo["demo_dir"]) / "geometry_qwen2_5_vl.json"


def affordance_path(demo: dict[str, Any]) -> Path:
    return Path(demo["demo_dir"]) / "affordance_robopoint.json"


def combined_review_path(root: Path, demo: dict[str, Any]) -> Path:
    return root / "human_check_bundle" / demo["id"] / "combined_review.json"


def progress_counts(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    demos = manifest.get("selected", [])
    total = len(demos)
    geometry_done = sum(1 for demo in demos if geometry_path(demo).exists())
    affordance_done = sum(1 for demo in demos if affordance_path(demo).exists())
    combined_done = sum(1 for demo in demos if combined_review_path(root, demo).exists())
    by_task: dict[str, dict[str, int]] = {}
    for demo in demos:
        task = demo.get("task", "unknown")
        item = by_task.setdefault(task, {"total": 0, "geometry": 0, "affordance": 0})
        item["total"] += 1
        item["geometry"] += int(geometry_path(demo).exists())
        item["affordance"] += int(affordance_path(demo).exists())
    return {
        "total_demos": total,
        "geometry_done": geometry_done,
        "affordance_done": affordance_done,
        "combined_done": combined_done,
        "by_task": by_task,
    }


def update_progress(root: Path, manifest: dict[str, Any], stage: str, current_demo: str = "", extra: dict[str, Any] | None = None) -> None:
    previous = read_json(root / "progress.json", {})
    payload = {
        "root": str(root),
        "started_at": previous.get("started_at") or now_iso(),
        "updated_at": now_iso(),
        "current_stage": stage,
        "current_demo": current_demo,
        "geometry_only": bool(manifest.get("geometry_only")),
        **progress_counts(root, manifest),
    }
    if previous.get("errors") and stage != "manifest_built":
        payload["errors"] = previous["errors"][-100:]
    if extra:
        payload.update(extra)
    write_json(root / "progress.json", payload)


def add_error(root: Path, message: str) -> None:
    progress = read_json(root / "progress.json", {})
    errors = progress.get("errors", [])
    errors.append({"time": now_iso(), "message": message})
    progress["errors"] = errors[-100:]
    progress["updated_at"] = now_iso()
    write_json(root / "progress.json", progress)


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"
    filled = min(width, int(width * done / total))
    pct = 100 * done / total
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({pct:5.1f}%)"


def print_status(root: Path) -> None:
    progress = read_json(root / "progress.json", {})
    manifest = read_json(root / "manifest.json", {"selected": []})
    counts = progress_counts(root, manifest)
    geometry_only = bool(progress.get("geometry_only") or manifest.get("geometry_only"))
    total = counts["total_demos"]
    print(f"root: {root}")
    print(f"stage: {progress.get('current_stage', 'unknown')}")
    print(f"current_demo: {progress.get('current_demo', '')}")
    print(f"updated_at: {progress.get('updated_at', 'never')}")
    print(f"geometry_only: {geometry_only}")
    print(f"geometry   {progress_bar(counts['geometry_done'], total)}")
    if not geometry_only:
        print(f"affordance {progress_bar(counts['affordance_done'], total)}")
    print(f"combined   {progress_bar(counts['combined_done'], total)}")
    if progress.get("errors"):
        print("recent_errors:")
        for err in progress["errors"][-5:]:
            print(f"- {err.get('time')}: {err.get('message')}")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest_source == "episodes":
        return build_episode_manifest(args)
    return build_train_json_manifest(args)


def read_episode_language(episode_path: Path, fallback: str) -> str:
    desc_path = episode_path / "variation_descriptions.pkl"
    if not desc_path.exists():
        return fallback.replace("_", " ")
    try:
        with desc_path.open("rb") as f:
            descriptions = pickle.load(f)
        if descriptions:
            return str(descriptions[0])
    except Exception:
        return fallback.replace("_", " ")
    return fallback.replace("_", " ")


def image_for_camera(episode_path: Path, camera: str) -> Path | None:
    preferred = episode_path / camera / "0.png"
    if preferred.exists():
        return preferred
    camera_dir = episode_path / camera
    if not camera_dir.exists():
        return None
    images = sorted(camera_dir.glob("*.png"), key=lambda path: int(path.stem) if path.stem.isdigit() else 10**9)
    return images[0] if images else None


def geometry_view_images_from_episode(episode_path: Path) -> dict[str, str]:
    front = image_for_camera(episode_path, "front_rgb")
    overhead = image_for_camera(episode_path, "overhead_rgb")
    wrist = image_for_camera(episode_path, "wrist_rgb")
    views: dict[str, str] = {}
    if front is not None:
        views["front"] = str(front)
    if overhead is not None:
        views["overhead"] = str(overhead)
    elif wrist is not None:
        views["overhead"] = str(wrist)
    return views


def geometry_view_images(demo: dict[str, Any]) -> dict[str, str]:
    views = demo.get("geometry_view_images")
    if isinstance(views, dict) and views:
        return {str(key): str(value) for key, value in views.items() if value}

    episode_path = demo.get("episode_path")
    if episode_path:
        inferred = geometry_view_images_from_episode(Path(episode_path))
        if inferred:
            return inferred

    images = demo.get("review_images") or demo.get("absolute_images") or []
    if not images:
        return {}
    fallback = {"front": str(images[0])}
    if len(images) > 1:
        fallback["overhead"] = str(images[1])
    return fallback


def build_episode_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    seen_root = Path(args.data_root) / "seen_tasks"
    selected = []
    missing = []
    wanted_tasks = set(args.task or [])

    task_dirs = sorted(path for path in seen_root.iterdir() if path.is_dir())
    for task_dir in task_dirs:
        task = task_dir.name
        if wanted_tasks and task not in wanted_tasks:
            continue
        episodes_root = task_dir / "all_variations" / "episodes"
        if not episodes_root.exists():
            continue
        episode_dirs = sorted(
            (path for path in episodes_root.iterdir() if path.is_dir() and path.name.startswith("episode")),
            key=lambda path: episode_id_from_episode_path(str(path)) if episode_id_from_episode_path(str(path)) is not None else 10**9,
        )
        for episode_path in episode_dirs:
            episode_id = episode_id_from_episode_path(str(episode_path))
            views = geometry_view_images_from_episode(episode_path)
            images = [views[key] for key in ["front", "overhead"] if key in views]
            if not images:
                if len(missing) < 100:
                    missing.append({"episode_path": str(episode_path), "reason": "missing front_rgb/overhead_rgb images"})
                continue
            demo_id = f"{task}_episode{episode_id}" if episode_id is not None else f"{task}_{episode_path.name}"
            scene_id = f"{task}:episode{episode_id}" if episode_id is not None else demo_id
            demo_dir = root / "demos" / task / episode_path.name
            record = {
                "id": demo_id,
                "scene_id": scene_id,
                "task": task,
                "episode_id": episode_id,
                "language_description": read_episode_language(episode_path, task),
                "relative_images": [str(Path(path).relative_to(Path(args.data_root))) for path in images],
                "absolute_images": images,
                "review_images": images,
                "geometry_view_images": views,
                "images_exist": [Path(path).exists() for path in images],
                "episode_path": str(episode_path),
                "frame_index": 0,
                "source_order": len(selected),
                "demo_dir": str(demo_dir),
                "input_state": "initial/current seen observation frame only; no future frames or after-state frames",
            }
            selected.append(record)
            if args.limit and len(selected) >= args.limit:
                break
        if args.limit and len(selected) >= args.limit:
            break

    for record in selected:
        demo_dir = Path(record["demo_dir"])
        demo_dir.mkdir(parents=True, exist_ok=True)
        write_json(demo_dir / "demo_metadata.json", record)

    manifest = {
        "created_at": now_iso(),
        "manifest_source": "episodes",
        "data_root": args.data_root,
        "train_json": args.train_json,
        "geometry_only": args.geometry_only,
        "all_train_rows": False,
        "dedupe_rule": "episode folder scan: one descriptor per seen demonstration episode",
        "candidate_rows_before_dedup": len(selected),
        "num_selected": len(selected),
        "selected": selected,
        "missing_examples_preview": missing,
        "task_counts": {},
    }
    for demo in selected:
        manifest["task_counts"][demo["task"]] = manifest["task_counts"].get(demo["task"], 0) + 1
    write_json(root / "manifest.json", manifest)
    update_progress(root, manifest, "manifest_built")
    print(json.dumps({"manifest": str(root / "manifest.json"), "num_selected": len(selected), "tasks": len(manifest["task_counts"])}, indent=2))
    return manifest


def build_train_json_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    rows = read_json(Path(args.train_json))
    if not isinstance(rows, list):
        raise ValueError(f"expected train JSON list at {args.train_json}")

    wanted_tasks = set(args.task or [])
    candidates = []
    missing = []
    for source_order, item in enumerate(rows):
        demo_id = item["id"]
        task = task_from_id(demo_id)
        if wanted_tasks and task not in wanted_tasks:
            continue
        relative_images = item.get("image", [])
        absolute_images = [str(Path(args.data_root) / rel) for rel in relative_images]
        exists = [Path(path).exists() for path in absolute_images]
        if not all(exists) or not absolute_images:
            if len(missing) < 100:
                missing.append({"id": demo_id, "images": absolute_images, "exists": exists})
            continue

        episode_path = episode_path_from_images(absolute_images)
        episode_id = episode_id_from_episode_path(episode_path)
        views = geometry_view_images_from_episode(Path(episode_path)) if episode_path else {}
        if not views and absolute_images:
            views = {"front": absolute_images[0]}
            if len(absolute_images) > 1:
                views["overhead"] = absolute_images[1]
        scene_id = f"{task}:episode{episode_id}" if episode_id is not None else demo_id
        demo_dir_name = f"episode{episode_id}" if episode_id is not None else demo_id
        demo_dir = root / "demos" / task / demo_dir_name
        demo_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": demo_id,
            "scene_id": scene_id,
            "task": task,
            "episode_id": episode_id,
            "language_description": item.get("language_description", ""),
            "relative_images": relative_images,
            "absolute_images": list(views.values()) or absolute_images,
            "review_images": list(views.values()) or absolute_images,
            "geometry_view_images": views,
            "images_exist": [Path(path).exists() for path in (list(views.values()) or absolute_images)],
            "episode_path": episode_path,
            "frame_index": frame_index_from_images(relative_images),
            "source_order": source_order,
            "demo_dir": str(demo_dir),
            "input_state": "initial/current seen observation frame only; no future frames or after-state frames",
        }
        candidates.append(record)

    if args.all_train_rows:
        selected = candidates
    else:
        by_episode: dict[str, dict[str, Any]] = {}
        for record in candidates:
            key = record.get("episode_path") or record["id"]
            previous = by_episode.get(key)
            if previous is None or record["frame_index"] < previous["frame_index"]:
                by_episode[key] = record
        selected = sorted(by_episode.values(), key=lambda item: item["source_order"])

    if args.limit:
        selected = selected[: args.limit]

    for record in selected:
        demo_dir = Path(record["demo_dir"])
        demo_dir.mkdir(parents=True, exist_ok=True)
        write_json(demo_dir / "demo_metadata.json", record)

    manifest = {
        "created_at": now_iso(),
        "manifest_source": "train-json",
        "data_root": args.data_root,
        "train_json": args.train_json,
        "geometry_only": args.geometry_only,
        "all_train_rows": args.all_train_rows,
        "dedupe_rule": "one descriptor per task episode, preferring the lowest first-image frame index",
        "candidate_rows_before_dedup": len(candidates),
        "num_selected": len(selected),
        "selected": selected,
        "missing_examples_preview": missing,
        "task_counts": {},
    }
    for demo in selected:
        manifest["task_counts"][demo["task"]] = manifest["task_counts"].get(demo["task"], 0) + 1
    write_json(root / "manifest.json", manifest)
    update_progress(root, manifest, "manifest_built")
    print(json.dumps({"manifest": str(root / "manifest.json"), "num_selected": len(selected), "tasks": len(manifest["task_counts"])}, indent=2))
    return manifest


def load_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if not manifest:
        raise SystemExit(f"Missing manifest: {root / 'manifest.json'}. Run --stage manifest first.")
    return manifest


def run_geometry(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from run_qwen_dual_view_geometry_target_pose import (
        apply_task_language_rules,
        fuse_descriptors,
        run_view,
    )

    root = Path(args.root)
    demos = manifest.get("selected", [])
    pending = [demo for demo in demos if args.force_geometry or not geometry_path(demo).exists()]
    update_progress(root, manifest, "geometry_start", extra={"geometry_pending": len(pending)})
    if not pending:
        print("geometry cache already complete")
        return

    processor = AutoProcessor.from_pretrained(args.qwen_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    for idx, demo in enumerate(pending, start=1):
        out_path = geometry_path(demo)
        update_progress(root, manifest, "geometry", demo["id"], {"stage_index": idx, "stage_total": len(pending)})
        try:
            views = geometry_view_images(demo)
            if "front" not in views and views:
                first_view, first_image = next(iter(views.items()))
                views = {"front": first_image, **{key: value for key, value in views.items() if key != first_view}}
            if "overhead" not in views and len(views) == 1:
                views["overhead"] = views["front"]
            if not views:
                raise ValueError("no geometry view images found")

            task_instruction = demo.get("language_description", "")
            view_outputs = {
                view_name: run_view(
                    processor,
                    model,
                    image_path,
                    task_instruction,
                    view_name,
                    args.max_new_tokens,
                )
                for view_name, image_path in views.items()
                if view_name in {"front", "overhead"}
            }
            if "front" not in view_outputs:
                raise ValueError("missing front view output")
            if "overhead" not in view_outputs:
                view_outputs["overhead"] = view_outputs["front"]

            fused, source_by_field, conflicts = fuse_descriptors(
                view_outputs["front"]["normalized_descriptor"],
                view_outputs["overhead"]["normalized_descriptor"],
            )
            rule_adjustments = apply_task_language_rules(
                demo.get("task"),
                task_instruction,
                fused,
                source_by_field,
            )
            write_json(
                out_path,
                {
                    "demo_id": demo["id"],
                    "task": demo.get("task"),
                    "language_description": task_instruction,
                    "model": args.qwen_model,
                    "schema_version": "clean_geometry_target_pose_v2_no_start_no_tags",
                    "image_inputs": {
                        "front_image": views.get("front"),
                        "overhead_image": views.get("overhead"),
                    },
                    "view_outputs": view_outputs,
                    "geometry_g_i": fused["geometry"],
                    "target_pose_i": fused["target_pose"],
                    "fused_descriptor": fused,
                    "source_by_field": source_by_field,
                    "front_overhead_conflicts": conflicts,
                    "rule_adjustments": rule_adjustments,
                    "target_pose_in_retrieval": False,
                    "target_pose_use": "final_llm_prompt_only",
                },
            )
            if idx % args.empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - keep long runs moving.
            write_json(out_path.with_name("geometry_qwen2_5_vl.error.json"), {"demo_id": demo["id"], "error": repr(exc), "time": now_iso()})
            add_error(root, f"geometry {demo['id']}: {exc!r}")
            if args.stop_on_error:
                raise
        if idx % args.progress_every == 0 or idx == len(pending):
            print(f"geometry {progress_bar(progress_counts(root, manifest)['geometry_done'], len(demos))} current={demo['id']}", flush=True)

    update_progress(root, manifest, "geometry_done")


def make_robopoint_questions(root: Path, demos: list[dict[str, Any]], question_file: Path) -> Path:
    image_folder = root / "robopoint_images"
    image_folder.mkdir(parents=True, exist_ok=True)
    with question_file.open("w") as f:
        for demo in demos:
            src = Path((demo.get("review_images") or demo.get("absolute_images"))[0])
            dst_name = f"{demo['id']}_{src.name}"
            dst = image_folder / dst_name
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(src, dst)
            row = {
                "question_id": demo["id"],
                "image": dst_name,
                "text": QUESTION_TEMPLATE.format(
                    task=demo.get("language_description", ""),
                    action_primitive=infer_action_primitive(demo.get("task", "")),
                    contact_mode=infer_contact_mode(demo.get("task", "")),
                    target_object=infer_manipulated_object(demo.get("task", "")),
                    target_part=infer_target_part(demo.get("task", "")),
                ),
                "category": "agnostos_contact_hint_full_cache",
            }
            f.write(json.dumps(row) + "\n")
    return image_folder


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    answers = {}
    if not path.exists():
        return answers
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                answers[row["question_id"]] = row
    return answers


def merge_answers(root: Path, answer_files: list[Path]) -> dict[str, dict[str, Any]]:
    master = root / "robopoint_answers_master.jsonl"
    answers = load_answers(master)
    for path in answer_files:
        answers.update(load_answers(path))
    with master.open("w") as f:
        for key in sorted(answers):
            f.write(json.dumps(answers[key]) + "\n")
    return answers


def write_affordance_outputs(args: argparse.Namespace, demos: list[dict[str, Any]], answers: dict[str, dict[str, Any]]) -> None:
    for demo in demos:
        ans = answers.get(demo["id"])
        if not ans:
            continue
        write_json(
            affordance_path(demo),
            {
                "demo_id": demo["id"],
                "task": demo.get("task"),
                "language_description": demo.get("language_description"),
                "model": args.robopoint_model,
                "images": demo.get("review_images") or demo.get("absolute_images"),
                "contact_hints_i": {
                    "raw_robopoint_text": ans.get("text"),
                    "expected_schema": CONTACT_HINT_SCHEMA,
                    "note": "RoboPoint predicts contact/keypoint hints only. These are not symbolic affordance descriptors and are not used for retrieval scoring.",
                    "vision_tower_note": "Pilot/full-cache run may use the RoboPoint LLM checkpoint with a local CLIP ViT-L/14 vision tower fallback on CAIR if openai/clip-vit-large-patch14-336 PyTorch weights cannot be fetched through Hugging Face SSL. Treat affordance outputs as human-check candidates.",
                },
                "raw_answer_record": ans,
            },
        )


def run_affordance(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    root = Path(args.root)
    demos = manifest.get("selected", [])
    pending = [demo for demo in demos if args.force_affordance or not affordance_path(demo).exists()]
    update_progress(root, manifest, "affordance_start", extra={"affordance_pending": len(pending)})
    if not pending:
        print("affordance cache already complete")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_file = root / f"robopoint_questions_pending_{run_id}.jsonl"
    answer_file = root / f"robopoint_answers_pending_{run_id}.jsonl"
    image_folder = make_robopoint_questions(root, pending, question_file)

    cmd = [
        sys.executable,
        "-m",
        "robopoint.eval.model_vqa",
        "--model-path",
        args.robopoint_model,
        "--image-folder",
        str(image_folder),
        "--question-file",
        str(question_file),
        "--answer-file",
        str(answer_file),
        "--conv-mode",
        args.robopoint_conv_mode,
        "--temperature",
        str(args.robopoint_temperature),
    ]
    print("running", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", args.hf_home)
    started = time.time()
    proc = subprocess.Popen(cmd, env=env)
    last_print = 0.0
    while proc.poll() is None:
        answered = sum(1 for line in answer_file.open()) if answer_file.exists() else 0
        if time.time() - last_print >= max(5, args.progress_seconds):
            update_progress(
                root,
                manifest,
                "affordance",
                extra={
                    "robopoint_answered_this_run": answered,
                    "robopoint_pending_this_run": len(pending),
                    "robopoint_elapsed_seconds": int(time.time() - started),
                },
            )
            print(f"robopoint {progress_bar(answered, len(pending))}", flush=True)
            last_print = time.time()
        time.sleep(5)
    rc = proc.returncode

    answers = merge_answers(root, [answer_file])
    write_affordance_outputs(args, pending, answers)
    update_progress(root, manifest, "affordance_done" if rc == 0 else "affordance_failed", extra={"robopoint_returncode": rc})
    print(f"affordance {progress_bar(progress_counts(root, manifest)['affordance_done'], len(demos))}", flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def normalize_outputs(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    root = Path(args.root)
    rows = []
    combined_rows = []
    bundle_root = root / "human_check_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    for demo in manifest.get("selected", []):
        task = demo.get("task") or ""
        geometry_doc = read_json(geometry_path(demo), {})
        geometry = geometry_doc.get("geometry_g_i") or {}
        target_pose_i = geometry_doc.get("target_pose_i") or (geometry_doc.get("fused_descriptor") or {}).get("target_pose") or {}
        if geometry and geometry_doc.get("schema_version") != "clean_geometry_target_pose_v2_no_start_no_tags":
            geometry_doc.setdefault("raw_geometry_g_i", geometry)
            geometry_doc["geometry_g_i"] = infer_primitive_geometry(task, geometry)
            geometry = geometry_doc["geometry_g_i"]
            write_json(geometry_path(demo), geometry_doc)

        affordance_doc = {} if args.geometry_only else read_json(affordance_path(demo), {})
        if affordance_doc:
            old_contact = affordance_doc.get("contact_hints_i") or affordance_doc.get("affordance_a_i", {})
            raw_text = old_contact.get("raw_robopoint_text")
            points = parse_points(raw_text)
            vision_tower_note = old_contact.get("vision_tower_note")
            affordance_doc["contact_hints_i"] = infer_contact_hints(task, points, raw_text, vision_tower_note)
            affordance_doc.pop("affordance_a_i", None)
            write_json(affordance_path(demo), affordance_doc)

        geometry_g_i = (geometry_doc or {}).get("geometry_g_i") or {}
        contact_hints_i = (affordance_doc or {}).get("contact_hints_i") or {}
        combined = {
            "demo_id": demo["id"],
            "scene_id": demo.get("scene_id"),
            "task": task,
            "episode_id": demo.get("episode_id"),
            "language_description": demo.get("language_description"),
            "episode_path": demo.get("episode_path"),
            "demo_dir": demo.get("demo_dir"),
            "images": demo.get("review_images") or demo.get("absolute_images") or [],
            "geometry_model": (geometry_doc or {}).get("model"),
            "geometry_schema_version": (geometry_doc or {}).get("schema_version"),
            "raw_geometry_g_i": (geometry_doc or {}).get("raw_geometry_g_i"),
            "geometry_g_i": geometry_g_i,
            "target_pose_i": target_pose_i,
            "target_pose_in_retrieval": False,
            "target_pose_use": "final_llm_prompt_only",
            "source_by_field": (geometry_doc or {}).get("source_by_field"),
            "front_overhead_conflicts": (geometry_doc or {}).get("front_overhead_conflicts"),
            "rule_adjustments": (geometry_doc or {}).get("rule_adjustments"),
            "source_files": {
                "geometry_qwen2_5_vl": str(geometry_path(demo)),
            },
        }
        if not args.geometry_only and affordance_doc:
            combined["robopoint_model"] = affordance_doc.get("model")
            combined["contact_hints_i"] = contact_hints_i
            combined["source_files"]["contact_hints_robopoint"] = str(affordance_path(demo))
        combined_path = combined_review_path(root, demo)
        write_json(combined_path, combined)
        combined_rows.append(combined)
        rows.append(
            {
                "demo_id": demo["id"],
                "scene_id": demo.get("scene_id"),
                "task": task,
                "episode_id": demo.get("episode_id"),
                "language_description": demo.get("language_description"),
                "episode_path": demo.get("episode_path"),
                "geometry_done": geometry_path(demo).exists(),
                "action_primitive": geometry_g_i.get("action_primitive"),
                "motion_type": geometry_g_i.get("motion_type"),
                "motion_axis": geometry_g_i.get("motion_axis"),
                "contact_type": geometry_g_i.get("contact_type"),
                "contact_region": geometry_g_i.get("contact_region"),
                "constraint_type": geometry_g_i.get("constraint_type"),
                "alignment_requirement": geometry_g_i.get("alignment_requirement"),
                "target_pose_goal_state": target_pose_i.get("goal_state_type"),
                "target_pose_relation": target_pose_i.get("required_final_relation"),
                "combined_review_file": str(combined_path),
            }
        )
        if not args.geometry_only:
            rows[-1]["contact_hints_done"] = affordance_path(demo).exists()
            rows[-1]["contact_mode"] = contact_hints_i.get("contact_mode")
            rows[-1]["points_2d_normalized"] = contact_hints_i.get("points_2d_normalized", [])

    write_json(root / "review_index.json", rows)
    with (root / "review_bundle.jsonl").open("w") as f:
        for row in combined_rows:
            f.write(json.dumps(row) + "\n")
    write_summary_markdown(root, manifest, rows)
    update_progress(root, manifest, "normalize_done")
    print(root / "review_bundle.jsonl")


def write_summary_markdown(root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    counts = progress_counts(root, manifest)
    with (root / "cache_summary.md").open("w") as f:
        geometry_only = bool(rows) and "contact_hints_done" not in rows[0]
        title = "Full Seen Clean Geometry + Target-Pose Cache" if geometry_only else "Full Seen Clean Geometry + Target-Pose / Contact-Hint Cache"
        f.write(f"# {title}\n\n")
        f.write(f"- Root: `{root}`\n")
        f.write(f"- Total demos: `{counts['total_demos']}`\n")
        f.write(f"- Geometry complete: `{counts['geometry_done']}`\n")
        f.write("- Geometry schema: `clean_geometry_target_pose_v2_no_start_no_tags`\n")
        f.write("- Target pose use: final LLM prompt only, not retrieval scoring\n")
        if not geometry_only:
            f.write(f"- Contact hints complete: `{counts['affordance_done']}`\n")
        f.write(f"- Review bundle: `{root / 'review_bundle.jsonl'}`\n\n")
        f.write("## Task Counts\n\n")
        if geometry_only:
            f.write("| Task | Total | Geometry |\n")
            f.write("| --- | ---: | ---: |\n")
        else:
            f.write("| Task | Total | Geometry | Contact hints |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
        for task, item in sorted(counts["by_task"].items()):
            if geometry_only:
                f.write(f"| {task} | {item['total']} | {item['geometry']} |\n")
            else:
                f.write(f"| {task} | {item['total']} | {item['geometry']} | {item['affordance']} |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache")
    parser.add_argument("--train-json", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/train.json")
    parser.add_argument("--data-root", default="/data/yf23/datasets/ICRA27-ROBOT")
    parser.add_argument("--qwen-model", default="/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--robopoint-model", default="/data/yf23/checkpoints/ICRA27-ROBOT/robopoint-v1-vicuna-v1.5-13b")
    parser.add_argument("--hf-home", default="/data/yf23/checkpoints/ICRA27-ROBOT/hf_home")
    parser.add_argument("--stage", choices=["all", "manifest", "geometry", "affordance", "normalize", "status"], default="all")
    parser.add_argument("--manifest-source", choices=["episodes", "train-json"], default="episodes")
    parser.add_argument("--task", action="append", help="Optional seen task filter; repeat for multiple tasks")
    parser.add_argument("--limit", type=int, help="Optional limit for smoke tests")
    parser.add_argument("--all-train-rows", action="store_true", help="Cache every train.json row instead of one descriptor per episode")
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-affordance", action="store_true")
    parser.add_argument("--geometry-only", action="store_true", help="Normalize/write review bundle without RoboPoint/contact-hint fields")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--empty-cache-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10, help="Print every N completed geometry demos")
    parser.add_argument("--progress-seconds", type=int, default=30, help="Print RoboPoint subprocess progress every N seconds")
    parser.add_argument("--robopoint-conv-mode", default="llava_v1")
    parser.add_argument("--robopoint-temperature", default="0")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage == "status":
        print_status(root)
        return

    if args.stage in {"all", "manifest"}:
        manifest = build_manifest(args)
    else:
        manifest = load_manifest(root)

    if args.stage in {"all", "geometry"}:
        run_geometry(args, manifest)
    if args.stage in {"all", "affordance"}:
        run_affordance(args, manifest)
    if args.stage in {"all", "normalize"}:
        normalize_outputs(args, manifest)
    if args.stage == "manifest":
        print_status(root)


if __name__ == "__main__":
    main()
