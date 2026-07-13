"""Prompt builder for one phase-step VLM action."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .long_term_memory import format_long_term_memory


PHASE_STEP_SYSTEM_PROMPT = (
    "You are a phase-step robot policy for a Franka Panda with a parallel gripper. "
    "You receive one current phase, compact long-term retrieved-demo memory, "
    "short-term execution memory, and precompiled candidate primitive actions. "
    "Select exactly one provided candidate action id, or report done/failed. "
    "Never invent coordinates."
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _image_content(path: str) -> Dict[str, Any]:
    content: Dict[str, Any] = {"type": "image", "image": path}
    max_pixels = os.environ.get("XICM_VL_MAX_PIXELS")
    min_pixels = os.environ.get("XICM_VL_MIN_PIXELS")
    resized_width = os.environ.get("XICM_VL_RESIZED_WIDTH")
    resized_height = os.environ.get("XICM_VL_RESIZED_HEIGHT")
    if max_pixels:
        content["max_pixels"] = int(max_pixels)
    if min_pixels:
        content["min_pixels"] = int(min_pixels)
    if resized_width:
        content["resized_width"] = int(resized_width)
    if resized_height:
        content["resized_height"] = int(resized_height)
    return content


def _phase_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "phase_index": item.get("phase_index"),
        "phase": item.get("phase"),
        "purpose": item.get("purpose"),
        "anchor_role": item.get("anchor_role"),
        "anchor_object": item.get("anchor_object"),
        "anchor_part": item.get("anchor_part"),
        "anchor_voxel_xyz": item.get("anchor_voxel_xyz"),
        "resolved_voxel_xyz": item.get("resolved_voxel_xyz"),
        "gripper": item.get("gripper"),
        "motion_constraint": item.get("motion_constraint"),
        "stop_condition": item.get("stop_condition"),
    }


def _candidate_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_id": item.get("action_id"),
        "phase_index": item.get("phase_index"),
        "phase": item.get("phase"),
        "action_7d": item.get("action_7d"),
        "anchor_role": item.get("anchor_role"),
        "anchor_voxel_xyz": item.get("anchor_voxel_xyz"),
        "resolved_voxel_xyz": item.get("resolved_voxel_xyz"),
        "offset_voxel_xyz": item.get("extra_motion_offset_voxel_xyz"),
        "gripper": item.get("gripper"),
    }


def _short_memory_brief(memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "current_phase_index": memory.get("current_phase_index"),
        "last_action_7d": memory.get("last_action_7d"),
        "recent_actions": (memory.get("recent_actions") or [])[-3:],
        "phase_history": (memory.get("phase_history") or [])[-4:],
        "phase_failures": (memory.get("phase_failures") or [])[-2:],
    }


def build_phase_step_user_prompt(
    *,
    task: str,
    instruction: str,
    phase: Dict[str, Any],
    all_phases: List[Dict[str, Any]],
    action_candidates: List[Dict[str, Any]],
    long_term_memory: Dict[str, Any],
    short_term_memory: Dict[str, Any],
    observation_summary: str,
    max_actions_per_phase: int,
) -> str:
    lines = [
        "Choose the next control decision for the current phase only.",
        "",
        "Rules:",
        f"- At most {max_actions_per_phase} actions may execute for this phase.",
        "- For phase_status=continue, selected_action_id must exactly match one candidate id below.",
        "- For phase_status=done or failed, selected_action_id must be null.",
        "- Do not output next_action_7d, raw coordinates, or extra JSON fields.",
        "- The candidate action_7d values are already valid voxel actions; choose by id only.",
        "- Do not mark a required manipulation phase done before an action was attempted unless the state summary proves it is already satisfied.",
        "- If uncertain, choose the best candidate id rather than marking done.",
        "",
        "Return exactly one JSON object:",
        "{",
        '  "phase_status": "continue|done|failed",',
        '  "selected_action_id": "candidate id or null",',
        '  "phase_done_evidence": "state evidence, or not_done_yet",',
        '  "why_this_action": "short reason",',
        '  "expected_scene_change": "short expected change",',
        '  "safe_to_advance": true or false,',
        '  "short_term_memory_note": "short note"',
        "}",
        "",
        f"Task: {task}",
        f"Instruction: {instruction}",
        f"Robot state summary: {observation_summary}",
        "",
        "Ordered phase plan:",
        _json([_phase_brief(item) for item in all_phases]),
        "",
        "Current phase:",
        _json(_phase_brief(phase)),
        "",
        "Candidate actions for this phase:",
        _json([_candidate_brief(item) for item in action_candidates]),
        "",
        "Retrieved-demo motion memory:",
        format_long_term_memory(long_term_memory),
        "",
        "Short-term execution memory:",
        _json(_short_memory_brief(short_term_memory)),
        "",
        "Now output only the JSON object.",
    ]
    return "\n".join(lines)


def build_phase_step_messages(
    user_prompt: str,
    *,
    front_rgb_path: Optional[str],
    overhead_rgb_path: Optional[str],
) -> List[Dict[str, Any]]:
    use_images = os.environ.get("XICM_PHASE_MEMORY_USE_IMAGES", "0") == "1"
    content: List[Dict[str, Any]] = []
    if use_images and front_rgb_path:
        content.append(_image_content(front_rgb_path))
    if use_images and overhead_rgb_path:
        content.append(_image_content(overhead_rgb_path))
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": PHASE_STEP_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
