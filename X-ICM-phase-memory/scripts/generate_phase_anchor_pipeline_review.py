#!/usr/bin/env python3
"""Compile saved phase-interpreter packets into anchored phases and 7D actions."""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from phase_anchor_pipeline import run_phase_anchor_pipeline


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path, text):
    with open(path, "w") as handle:
        handle.write(str(text).rstrip() + "\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def phase_review_dirs(root):
    root_path = Path(root)
    dirs = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "01_extracted_query.json").exists() and (child / "04_phase_interpreter_parsed.json").exists():
            dirs.append(child)
    return dirs


def copy_light_review_assets(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "query_front_rgb_0.png",
        "query_overhead_rgb_0.png",
        "contact_points_overlay_front.png",
        "contact_points_overlay_overhead.png",
        "contact_points_overlay_front_overhead.png",
        "01_extracted_query.md",
        "02_phase_interpreter_prompt.txt",
        "03_phase_interpreter_raw_output.txt",
        "04_phase_interpreter_parsed.json",
    ]:
        src = source_dir / name
        if src.exists() and output_dir / name != src:
            shutil.copy2(src, output_dir / name)


def load_execution_prior(path, task_name):
    if not path:
        return {}
    data = read_json(path)
    if isinstance(data, dict) and task_name in data:
        return data[task_name] or {}
    return data if isinstance(data, dict) else {}


def phase_table_md(normalized):
    lines = [
        "# Normalized Anchored Phase Plan",
        "",
        f"Task: `{normalized.get('task')}`",
        f"Task family: `{normalized.get('task_family')}`",
        f"Success condition: `{normalized.get('success_condition')}`",
        "",
        "## Notes",
    ]
    notes = normalized.get("normalization_notes") or ["none"]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Scene Anchors",
            "| index | role | object | part | voxel | quality | source |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for anchor in normalized.get("scene_anchors") or []:
        lines.append(
            f"| {anchor.get('index')} | {anchor.get('role')} | {anchor.get('object')} | "
            f"{anchor.get('part')} | {anchor.get('voxel_xyz')} | {anchor.get('quality')} | {anchor.get('source')} |"
        )
    lines.extend(
        [
            "",
            "## Anchored Phases",
            "| # | phase | anchor role | anchor index | anchor voxel | offset | resolved voxel | gripper | constraint |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for phase in normalized.get("anchored_phase_plan") or []:
        lines.append(
            f"| {phase.get('phase_index')} | {phase.get('phase')} | {phase.get('anchor_role')} | "
            f"{phase.get('anchor_index')} | {phase.get('anchor_voxel_xyz')} | {phase.get('offset_voxel_xyz')} | "
            f"{phase.get('resolved_voxel_xyz')} | {phase.get('gripper')} | {phase.get('motion_constraint')} |"
        )
    return "\n".join(lines)


def compiled_actions_text(compilation):
    lines = [
        "# Compiled Open-Loop 7D Actions",
        "",
        "Python-style action list:",
        "",
        "```python",
        json.dumps(compilation.get("actions_7d") or []),
        "```",
        "",
        "## Phase Trace",
        "| # | phase | anchor role | anchor index | action_7d |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in compilation.get("compiled_steps") or []:
        lines.append(
            f"| {step.get('phase_index')} | {step.get('phase')} | {step.get('anchor_role')} | "
            f"{step.get('anchor_index')} | {step.get('action_7d')} |"
        )
    skipped = compilation.get("skipped_steps") or []
    if skipped:
        lines.extend(["", "## Skipped Phases"])
        for item in skipped:
            lines.append(f"- {item}")
    return "\n".join(lines)


def verifier_md(verification):
    lines = [
        "# Action Verifier",
        "",
        f"Passed: `{verification.get('passed')}`",
        f"Phase plan passed: `{verification.get('phase_plan_passed')}`",
        f"Compiled actions passed: `{verification.get('compiled_actions_passed')}`",
        "",
        "## Issues",
    ]
    issues = verification.get("issues") or []
    if not issues:
        lines.append("- none")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}`: {issue.get('message')}")
    lines.extend(["", "## Repair Recommendations"])
    for rec in verification.get("repair_recommendations") or []:
        lines.append(f"- {rec}")
    return "\n".join(lines)


def run_one(source_dir, output_dir, execution_prior):
    extracted = read_json(source_dir / "01_extracted_query.json")
    phase_output = read_json(source_dir / "04_phase_interpreter_parsed.json")
    query_context = {
        "task": extracted.get("task"),
        "task_key": extracted.get("task"),
        "instruction": extracted.get("instruction"),
        "current_observation": extracted.get("current_observation"),
        "geometry_g_j": extracted.get("geometry_g_j") or {},
        "goal_state_h_j": extracted.get("goal_state_h_j") or {},
        "contact_hints_c_j": extracted.get("contact_hints_c_j") or {},
        "interaction_profile_p_j": extracted.get("interaction_profile_p_j") or {},
    }
    result = run_phase_anchor_pipeline(
        phase_output,
        query_context,
        execution_prior=execution_prior,
    )
    copy_light_review_assets(source_dir, output_dir)
    write_json(output_dir / "05_normalized_anchored_phase_plan.json", result["normalized_phase_plan"])
    write_text(output_dir / "05_normalized_anchored_phase_plan.md", phase_table_md(result["normalized_phase_plan"]))
    write_json(output_dir / "06_compiled_open_loop_actions.json", result["compiled_actions"])
    write_text(output_dir / "06_compiled_open_loop_actions.txt", compiled_actions_text(result["compiled_actions"]))
    write_json(output_dir / "07_action_verifier.json", result["verification"])
    write_text(output_dir / "07_action_verifier.md", verifier_md(result["verification"]))
    write_json(output_dir / "08_phase_anchor_pipeline_combined.json", result)
    return {
        "task": query_context["task"],
        "instruction": query_context["instruction"],
        "task_family": result["normalized_phase_plan"].get("task_family"),
        "phase_count": len(result["normalized_phase_plan"].get("anchored_phase_plan") or []),
        "action_count": len(result["compiled_actions"].get("actions_7d") or []),
        "verifier_passed": result["verification"].get("passed"),
        "issue_count": len(result["verification"].get("issues") or []),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-review-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--execution-prior-json", default="")
    args = parser.parse_args()

    source_root = Path(args.phase_review_root).resolve()
    if args.output_root:
        run_name = args.name or datetime.now().strftime("phase_anchor_pipeline_%Y%m%d_%H%M%S")
        output_root = Path(args.output_root).resolve() / run_name
    else:
        output_root = source_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_dir in phase_review_dirs(source_root):
        task_name = source_dir.name.split("_episode", 1)[0]
        if task_name[:3].isdigit() and "_" in task_name:
            task_name = task_name.split("_", 1)[1]
        output_dir = output_root / source_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        execution_prior = load_execution_prior(args.execution_prior_json, task_name)
        print(f"Compiling {source_dir.name}", flush=True)
        rows.append(run_one(source_dir, output_dir, execution_prior))

    write_csv(
        output_root / "phase_anchor_pipeline_summary.csv",
        rows,
        ["task", "instruction", "task_family", "phase_count", "action_count", "verifier_passed", "issue_count", "output_dir"],
    )
    write_json(output_root / "phase_anchor_pipeline_summary.json", rows)
    write_text(
        output_root / "README.md",
        "\n".join(
            [
                "# Phase Anchor Pipeline Review",
                "",
                "This folder contains query-only phase outputs normalized into anchor-bound phases, compiled open-loop 7D actions, and verifier reports.",
                "",
                "Pipeline rule: Query decides WHAT. Anchors provide WHERE. Retrieval helps HOW. Compiler produces 7D actions.",
                "",
                "Summary:",
                *[
                    f"- {row['task']}: family={row['task_family']}, phases={row['phase_count']}, "
                    f"actions={row['action_count']}, verifier_passed={row['verifier_passed']}, issues={row['issue_count']}"
                    for row in rows
                ],
            ]
        ),
    )
    print(f"Wrote phase-anchor pipeline review to {output_root}", flush=True)


if __name__ == "__main__":
    main()
