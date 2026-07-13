"""Build compact long-term memory from retrieved seen-demo prompts."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from phase_anchor_pipeline.anchor_utils import normalized_text


ACTION_RE = re.compile(r"\[[^\[\]]*?\d[^\[\]]*?\]")


def _first_line_after(label: str, text: str) -> str:
    idx = text.find(label)
    if idx < 0:
        return ""
    tail = text[idx + len(label) :].strip().splitlines()
    return tail[0].strip() if tail else ""


def _parse_action_list(text: str) -> List[List[int]]:
    actions: List[List[int]] = []
    for match in ACTION_RE.findall(text):
        try:
            value = json.loads(match)
        except Exception:
            continue
        if isinstance(value, list) and len(value) == 7:
            try:
                actions.append([int(round(float(item))) for item in value])
            except (TypeError, ValueError):
                continue
    return actions


def _gripper_rhythm(actions: Iterable[List[int]]) -> str:
    states = ["open" if action[-1] == 1 else "closed" for action in actions if len(action) == 7]
    if not states:
        return "unknown"
    collapsed = []
    for state in states:
        if not collapsed or collapsed[-1] != state:
            collapsed.append(state)
    return " -> ".join(collapsed)


def _motion_delta_summary(actions: List[List[int]]) -> Dict[str, Any]:
    if len(actions) < 2:
        return {"delta_voxel": [0, 0, 0], "z_change": 0, "xy_span": 0}
    xs = [a[0] for a in actions]
    ys = [a[1] for a in actions]
    zs = [a[2] for a in actions]
    return {
        "delta_voxel": [actions[-1][0] - actions[0][0], actions[-1][1] - actions[0][1], actions[-1][2] - actions[0][2]],
        "z_change": actions[-1][2] - actions[0][2],
        "xy_span": max(max(xs) - min(xs), max(ys) - min(ys)),
    }


def _demo_blocks(retrieval_prompt: str, max_examples: int) -> List[str]:
    matches = list(re.finditer(r"^Seen demonstration\s+\d+:", retrieval_prompt, flags=re.MULTILINE))
    blocks = []
    for index, match in enumerate(matches[:max_examples]):
        end = matches[index + 1].start() if index + 1 < len(matches) else retrieval_prompt.find("\nUnseen query:", match.end())
        if end < 0:
            end = len(retrieval_prompt)
        blocks.append(retrieval_prompt[match.start() : end].strip())
    return blocks


def _summarize_demo(block: str, rank: int) -> Dict[str, Any]:
    task_instruction = _first_line_after("Task instruction:", block)
    score_line = next((line.strip() for line in block.splitlines() if line.strip().startswith("Retrieval scores:")), "")
    plan_line = next((line.strip() for line in block.splitlines() if line.strip().startswith("Plan compatibility:")), "")
    actions = _parse_action_list(block)
    return {
        "rank": rank,
        "task_instruction": task_instruction,
        "retrieval_scores": score_line,
        "plan_compatibility": plan_line,
        "action_count": len(actions),
        "gripper_rhythm": _gripper_rhythm(actions),
        "motion_delta_summary": _motion_delta_summary(actions),
        "first_actions_7d": actions[:4],
    }


def _phase_names(normalized_plan: Dict[str, Any]) -> List[str]:
    return [str(phase.get("phase") or "unknown") for phase in normalized_plan.get("anchored_phase_plan") or []]


def _anchor_summary(normalized_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    anchors = []
    for anchor in normalized_plan.get("scene_anchors") or []:
        anchors.append(
            {
                "index": anchor.get("index"),
                "role": anchor.get("role"),
                "object": anchor.get("object"),
                "part": anchor.get("part"),
                "voxel_xyz": anchor.get("voxel_xyz"),
                "quality": anchor.get("quality"),
            }
        )
    return anchors[:12]


def build_long_term_memory(
    *,
    task_name: str,
    instruction: str,
    normalized_plan: Dict[str, Any],
    retrieval_prompt: str = "",
    retrieval_method: str = "",
    max_examples: int = 4,
) -> Dict[str, Any]:
    """Create the compact long-term memory used by phase-step prompts.

    The original retrieved prompt can be huge. This function keeps only the
    action rhythm and compatibility hints so the phase-step policy receives an
    overall guide without being asked to copy full demonstrations.
    """

    examples = [
        _summarize_demo(block, rank)
        for rank, block in enumerate(_demo_blocks(retrieval_prompt or "", max_examples), start=1)
    ]
    family = normalized_text(normalized_plan.get("task_family")) or "unknown"
    return {
        "schema_version": "phase_memory_long_term_v1",
        "task": task_name,
        "instruction": instruction,
        "task_family": family,
        "success_condition": normalized_plan.get("success_condition"),
        "retrieval_method": retrieval_method,
        "phase_template": _phase_names(normalized_plan),
        "scene_anchor_summary": _anchor_summary(normalized_plan),
        "retrieved_demo_summaries": examples,
        "controller_rules": [
            "Retrieved demos are long-term motion priors only.",
            "Current query phase and anchors decide the next action.",
            "Do not copy a retrieved final goal if it conflicts with the query success condition.",
        ],
    }


def format_long_term_memory(memory: Optional[Dict[str, Any]]) -> str:
    if not memory:
        return "No long-term memory available."
    lines = [
        f"- task_family: {memory.get('task_family', 'unknown')}",
        f"- success_condition: {memory.get('success_condition', 'unknown')}",
        f"- phase_template: {memory.get('phase_template')}",
        "- retrieved_demo_summaries:",
    ]
    demos = memory.get("retrieved_demo_summaries") or []
    if not demos:
        lines.append("  - none")
    for demo in demos:
        lines.append(
            "  - "
            f"rank={demo.get('rank')}; task={demo.get('task_instruction')}; "
            f"actions={demo.get('action_count')}; gripper={demo.get('gripper_rhythm')}; "
            f"motion={demo.get('motion_delta_summary')}; "
            f"compat={demo.get('plan_compatibility') or demo.get('retrieval_scores')}"
        )
    return "\n".join(lines)
