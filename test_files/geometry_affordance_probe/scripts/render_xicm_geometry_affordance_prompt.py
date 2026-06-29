#!/usr/bin/env python3
"""Render the clean v1 primitive-geometry X-ICM prompt.

This renderer is intentionally separate from the vanilla X-ICM baseline path.
It expects retrieved seen demonstrations to be pre-rendered as per-key-action
observation/action steps, matching the AGNOSTOS/X-ICM paper format. Clean v1
uses primitive geometry for retrieval and optional RoboPoint contact hints only
as final-prompt evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = """You are a Franka Panda robot with a parallel gripper.

You will receive the top-k retrieved in-context demonstrations from seen robot manipulation tasks. Each seen demonstration contains:
- a task instruction,
- a sequence of key action steps,
- for each key action step, an X-ICM observation text summarizing object names and discretized 3D positions,
- for each key action step, the corresponding next 7D action,
- a primitive geometry/action description g_i covering the object, part, action primitive, motion type/axis, contact region, constraint, and alignment requirement,
- optional RoboPoint contact hints c_i with visible contact points when available.

You will then receive one unseen query. The unseen query contains only the current/initial observation, task instruction, primitive geometry/action description g_j, and optional contact hints c_j. It does not contain future observations, after-states, ground-truth actions, or unseen demonstrations.

Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the top-k retrieved seen demonstrations. Use action trends and primitive manipulation geometry to choose the best physical analogy: action primitive, contact region, motion axis, insertion/containment/alignment needs, and mechanical constraint. Use contact hints only to ground where to touch in the current scene. Do not invent objects, future states, or actions that are not supported by the current unseen observation and instruction.

Return only a Python-style list of 7D action lists. Do not output explanations, labels, markdown, or any other text."""


GEOMETRY_FIELDS = [
    "manipulated_object",
    "object_category",
    "primary_shape",
    "target_part",
    "secondary_parts",
    "action_primitive",
    "motion_type",
    "motion_axis",
    "contact_type",
    "contact_region",
    "constraint_type",
    "alignment_requirement",
    "state",
    "geometry_tags",
    "execution_clearance_hint",
]

CONTACT_HINT_FIELDS = [
    "contact_mode",
    "source_view",
    "target_object",
    "target_part",
    "points_2d_normalized",
    "contact_region_text",
    "source_role",
]

DISALLOWED_QUERY_KEYS = {
    "action",
    "actions",
    "steps",
    "trajectory",
    "future_observations",
    "after_state",
    "after_states",
    "ground_truth_actions",
}


class PromptFormatError(ValueError):
    """Raised when prompt input would violate the intended X-ICM format."""


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _field_default(field: str) -> Any:
    if field in {"secondary_parts", "geometry_tags", "points_2d_normalized"}:
        return []
    if field in {"execution_clearance_hint"}:
        return "none"
    return "unknown"


def _format_feature_block(title: str, values: dict[str, Any], fields: Iterable[str]) -> str:
    lines = [f"{title}:"]
    for field in fields:
        lines.append(f"- {field}: {_jsonish(values.get(field, _field_default(field)))}")
    return "\n".join(lines)


def _coerce_action(action: Any, context: str) -> list[Any]:
    if not isinstance(action, list) or len(action) != 7:
        raise PromptFormatError(f"{context} must be a 7-element action list, got {action!r}")
    for value in action:
        if not isinstance(value, (int, float)):
            raise PromptFormatError(f"{context} action values must be numeric, got {action!r}")
    return action


def _validate_steps(demo: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    steps = demo.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PromptFormatError(f"seen demonstration {rank} must contain a non-empty steps list")
    cleaned = []
    for step_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise PromptFormatError(f"seen demonstration {rank} step {step_index} must be an object")
        observation = step.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            raise PromptFormatError(f"seen demonstration {rank} step {step_index} needs observation text")
        action = _coerce_action(step.get("action"), f"seen demonstration {rank} step {step_index}")
        cleaned.append({"observation": observation, "action": action})
    return cleaned


def _format_demo(demo: dict[str, Any], rank: int, include_geometry: bool, include_affordance: bool) -> str:
    task = demo.get("task_instruction") or demo.get("language_description") or demo.get("task") or "unknown"
    lines = [
        f"Seen demonstration {rank}:",
        "Task instruction:",
        str(task),
        "",
    ]

    if include_geometry:
        geometry = demo.get("geometry_g_i") or demo.get("geometry") or {}
        lines.extend([_format_feature_block("Primitive geometry/action descriptor g_i", geometry, GEOMETRY_FIELDS), ""])

    if include_affordance:
        contact_hints = demo.get("contact_hints_i") or demo.get("affordance_a_i") or demo.get("affordance") or {}
        lines.extend([_format_feature_block("RoboPoint contact hints c_i", contact_hints, CONTACT_HINT_FIELDS), ""])

    lines.append("Key observation-action trajectory:")
    for step_index, step in enumerate(_validate_steps(demo, rank), start=1):
        lines.extend(
            [
                f"Step {step_index} observation:",
                step["observation"],
                f"Step {step_index} 7D action:",
                json.dumps(step["action"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _validate_query(query: dict[str, Any]) -> None:
    blocked = sorted(DISALLOWED_QUERY_KEYS.intersection(query))
    if blocked:
        raise PromptFormatError(f"unseen query must not include future/action keys: {', '.join(blocked)}")
    observation = query.get("observation") or query.get("current_observation")
    if not isinstance(observation, str) or not observation.strip():
        raise PromptFormatError("unseen query needs current observation text")


def render_user_prompt(payload: dict[str, Any], include_geometry: bool = True, include_affordance: bool = True) -> str:
    demos = payload.get("retrieved_demos") or payload.get("seen_demonstrations") or []
    if not isinstance(demos, list) or not demos:
        raise PromptFormatError("payload must contain retrieved_demos")

    query = payload.get("query") or payload.get("unseen_query")
    if not isinstance(query, dict):
        raise PromptFormatError("payload must contain query")
    _validate_query(query)

    lines = [
        f"You will receive {len(demos)} top-k retrieved seen demonstrations from the AGNOSTOS seen-task training set. Use all of them as in-context examples for the current unseen query.",
        "",
        "Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the retrieved seen demonstrations using action trends and primitive manipulation geometry.",
        "",
        "Important rules:",
        "- Each seen demonstration includes per-key-action observations paired with the corresponding 7D action.",
        "- Each seen demonstration includes primitive geometry/action descriptor g_i.",
        "- Optional RoboPoint contact hints c_i/c_j are contact evidence only, not retrieval scores.",
        "- The unseen query includes only the current/initial observation, task instruction, primitive geometry/action descriptor g_j, and optional contact hints c_j.",
        "- Do not use unseen demonstrations, unseen future frames, unseen ground-truth actions, or after-states.",
        "- If geometry/contact hints conflict with a seen demo's action trend, prioritize the current unseen observation and task instruction.",
        "- Preserve the X-ICM output format: only a list of 7D action lists, such as [[x, y, z, roll, pitch, yaw, gripper], ...].",
        "",
    ]

    for rank, demo in enumerate(demos, start=1):
        lines.extend([_format_demo(demo, rank, include_geometry, include_affordance), ""])

    task = query.get("task_instruction") or query.get("language_description") or query.get("task") or "unknown"
    observation = query.get("observation") or query.get("current_observation")
    lines.extend(
        [
            "Unseen task:",
            "Task instruction:",
            str(task),
            "",
            "Current observation:",
            str(observation),
            "",
        ]
    )

    if include_geometry:
        geometry = query.get("geometry_g_j") or query.get("geometry") or {}
        lines.extend([_format_feature_block("Primitive geometry/action descriptor g_j", geometry, GEOMETRY_FIELDS), ""])

    if include_affordance:
        contact_hints = query.get("contact_hints_j") or query.get("affordance_a_j") or query.get("affordance") or {}
        lines.extend([_format_feature_block("RoboPoint contact hints c_j", contact_hints, CONTACT_HINT_FIELDS), ""])

    lines.append("Predict the key 7D action sequence for the unseen task. Return only a Python-style list of 7D action lists:")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON payload with retrieved_demos and query")
    parser.add_argument("--output", help="Optional path for rendered prompt")
    parser.add_argument("--include-system", action="store_true", help="Write system prompt followed by user prompt")
    parser.add_argument("--omit-geometry", action="store_true", help="Render affordance-only ablation")
    parser.add_argument("--omit-affordance", action="store_true", help="Omit optional RoboPoint contact hints")
    parser.add_argument("--validate-only", action="store_true", help="Validate the payload without writing a prompt")
    args = parser.parse_args()

    payload = _load_json(args.input)
    user_prompt = render_user_prompt(
        payload,
        include_geometry=not args.omit_geometry,
        include_affordance=not args.omit_affordance,
    )
    rendered = f"System prompt:\n{SYSTEM_PROMPT}\n\nUser prompt:\n{user_prompt}" if args.include_system else user_prompt

    if args.validate_only:
        return

    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
