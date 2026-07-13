"""Query-only phase interpretation for manipulation tasks.

This module keeps task understanding separate from retrieval. It asks a VLM to
describe the current unseen query as structured manipulation phases, while the
later compiler/verifier will be responsible for turning phases into actions.
"""

import json
import re
from typing import Any, Dict, List, Optional


PHASE_INTERPRETER_SYSTEM_PROMPT = (
    "You are a robotics task phase interpreter for a Franka Panda robot with a "
    "parallel gripper. Interpret only the current unseen query scene. You do "
    "not receive retrieved demonstrations, and you must not infer phases by "
    "copying from seen demos. Your job is to output a structured phase plan "
    "that states what the robot must do, which object roles are involved, and "
    "what success relation must be achieved. Do not output raw 7D robot actions."
)


PHASE_SCHEMA_TEXT = """
Return exactly one JSON object with this schema:
{
  "task_family": "short_family_name",
  "manipulated_object": "object the robot directly grasps/touches",
  "target_object_or_region": "goal/support/receptacle/control/region",
  "success_condition": "physical relation that means the task is done",
  "phase_plan": [
    {
      "phase": "pregrasp|grasp|lift|align|move|lower|insert|press|pull|rotate|slide|sweep|pour|release|retract|stop",
      "purpose": "why this phase is needed",
      "target_role": "manipulated_object_contact|goal_region|secondary_object_to_move|constraint_reference|tool_working_edge|none",
      "anchor_index": "integer point index from c_j if this phase uses a specific point, otherwise null",
      "anchor_role": "same role as the bound c_j anchor, otherwise none",
      "anchor_offset": "none|safe_z|precontact_z|retract_z|task_axis_offset",
      "target_object": "object/part used in this phase",
      "gripper": "open|close|closed|unchanged",
      "motion_constraint": "free_motion|vertical_axis|linear_axis|rotation_axis|surface_plane|hinge_arc|slot_axis|none",
      "stop_condition": "how to know this phase is complete"
    }
  ],
  "execution_constraints": {
    "requires_grasp_before_transport": true,
    "requires_lift_clearance": true,
    "release_only_at_goal": true,
    "axis_alignment": "none|low|medium|high",
    "critical_contact_side_or_direction": "short direction/contact note"
  },
  "role_bindings": [
    {
      "role": "manipulated_object_contact|goal_region|secondary_object_to_move|constraint_reference|tool_working_edge",
      "object": "object name",
      "part": "part name",
      "voxel_xyz": [x, y, z],
      "why_it_matters": "how the compiler should use this point"
    }
  ],
  "failure_modes_to_avoid": ["short physical mistakes to avoid"]
}
"""


def _format_jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def build_phase_interpreter_user_prompt(
    *,
    task_key: str,
    instruction: str,
    current_observation: str,
    query_geometry: Dict[str, Any],
    goal_state: Dict[str, Any],
    contact_hints: Dict[str, Any],
    interaction_profile: Dict[str, Any],
) -> str:
    """Build the query-only VLM prompt for phase interpretation."""

    contact_text = contact_hints.get("llm_contact_hint_text")
    if not contact_text:
        contact_text = "No role-labeled 3D contact hints available."

    lines = [
        "Interpret the current unseen query scene into manipulation phases.",
        "",
        "Important rules:",
        "- Use only the current query instruction, images, observation, masks/object names, geometry, goal descriptor, and role-labeled contact hints.",
        "- Retrieved demonstrations are intentionally not provided to this phase interpreter.",
        "- Decide WHAT the task is and which phases are required. Do not produce raw 7D actions.",
        "- Bind each phase to a role-labeled c_j scene anchor when that phase needs a physical point.",
        "- Use anchor_index from the numbered c_j list when a phase clearly uses that point.",
        "- Use anchor_role=none and anchor_index=null for phases that do not need a new scene point.",
        "- Do not invent exact action keyframes; only state phase-to-anchor bindings and offsets.",
        "- If a task has no goal point, leave goal-specific phases out instead of forcing a goal_region.",
        "- The output will later be compiled into primitive templates, so phases must be mechanically executable.",
        "",
        PHASE_SCHEMA_TEXT.strip(),
        "",
        "Current query:",
        f"- task_key: {task_key}",
        f"- instruction: {instruction}",
        "",
        "Current observation:",
        str(current_observation),
        "",
        "Primitive geometry/action descriptor g_j:",
    ]
    for key in [
        "action_primitive",
        "motion_type",
        "motion_axis",
        "contact_type",
        "contact_region",
        "constraint_type",
        "alignment_requirement",
        "execution_clearance_hint",
    ]:
        if key in query_geometry:
            lines.append(f"- {key}: {_format_jsonish(query_geometry.get(key))}")

    lines.extend(["", "Goal-state/contact-pose descriptor h_j:"])
    for key in [
        "goal_state_type",
        "manipulated_object",
        "target_object_or_region",
        "required_final_relation",
        "contact_or_release_target",
        "required_motion_constraint",
        "required_orientation_or_alignment",
        "release_or_stop_condition",
        "success_check",
        "goal_tags",
    ]:
        if key in goal_state:
            lines.append(f"- {key}: {_format_jsonish(goal_state.get(key))}")

    lines.extend(["", "Role-labeled query contact hints c_j:", str(contact_text)])

    lines.extend(["", "Current interaction profile p_j:"])
    for key in [
        "interaction_family",
        "motion_sequence",
        "contact_strategy",
        "target_relation",
        "axis_constraint",
        "articulation_model",
        "precision_driver",
        "transfer_caution",
    ]:
        if key in interaction_profile:
            lines.append(f"- {key}: {_format_jsonish(interaction_profile.get(key))}")

    lines.extend(
        [
            "",
            "Now output only the JSON phase plan. Do not include markdown fences or explanation.",
        ]
    )
    return "\n".join(lines)


def build_phase_messages(
    user_prompt: str,
    front_rgb_path: Optional[str] = None,
    overhead_rgb_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build QwenVL-style chat messages for the phase interpreter."""

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Current unseen query images are attached. First image: front RGB. "
                "Second image: overhead/top RGB. Interpret this query only."
            ),
        }
    ]
    if front_rgb_path:
        content.append({"type": "image", "image": front_rgb_path})
    if overhead_rgb_path:
        content.append({"type": "image", "image": overhead_rgb_path})
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": PHASE_INTERPRETER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _strip_markdown_fences(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str:
    cleaned = _strip_markdown_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in model output.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ValueError("JSON object was opened but not closed.")


def parse_phase_interpreter_output(text: str) -> Dict[str, Any]:
    """Parse and lightly validate phase interpreter JSON output."""

    json_text = _extract_first_json_object(text)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Phase interpreter output must be a JSON object.")
    if not isinstance(parsed.get("phase_plan"), list):
        raise ValueError("Phase interpreter output must include phase_plan as a list.")
    if not parsed.get("task_family"):
        parsed["task_family"] = "unknown"
    if not parsed.get("role_bindings"):
        parsed["role_bindings"] = []
    if not parsed.get("failure_modes_to_avoid"):
        parsed["failure_modes_to_avoid"] = []
    return parsed
