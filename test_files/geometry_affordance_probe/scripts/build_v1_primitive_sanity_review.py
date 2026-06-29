#!/usr/bin/env python3
"""Build a small clean-v1 primitive-geometry sanity review batch.

The input is the existing Qwen geometry human-check bundle. This script does not
rerun Qwen or RoboPoint. It keeps the original model input/output intact, adds
the proposed clean-v1 primitive descriptor, and groups everything by demo for
manual checking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cache_all_seen_geometry_affordance import GEOMETRY_PROMPT, infer_contact_hints, infer_primitive_geometry  # noqa: E402


DEFAULT_SOURCE = Path(
    "test_files/geometry_affordance_probe/review/"
    "2026-06-23_qwen_geometry_cache_human_checks/qwen_geometry_cache_human_check_bundle.json"
)
DEFAULT_OUT = Path(
    "test_files/geometry_affordance_probe/review/"
    "2026-06-23_v1_primitive_sanity_review_batch"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def copy_images(item: dict[str, Any], demo_dir: Path) -> list[str]:
    copied = []
    for image_path in item.get("image_inputs", {}).get("local_review_copies", []):
        src = Path(image_path)
        if not src.exists():
            continue
        dst = demo_dir / "inputs" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst.resolve()))
    return copied


def human_check_template() -> dict[str, Any]:
    return {
        "geometry_matches_task": None,
        "primitive_action_correct": None,
        "motion_axis_correct": None,
        "target_part_correct": None,
        "constraint_type_correct": None,
        "clearance_hint_should_be_final_prompt_only": True,
        "notes": "",
    }


def build_item(item: dict[str, Any], out_root: Path) -> dict[str, Any]:
    review_id = item["review_id"]
    task = item.get("task") or ""
    old_geometry = item.get("llm_output", {}).get("parsed_geometry_json_from_source") or {}
    primitive = infer_primitive_geometry(task, old_geometry)
    contact_hints = infer_contact_hints(task, [], None)

    demo_dir = out_root / "demos" / review_id
    copied_images = copy_images(item, demo_dir)

    old_prompt = item.get("llm_input", {}).get("prompt_text") or ""
    new_prompt = GEOMETRY_PROMPT.replace("{task}", item.get("language_description") or task)

    write_json(demo_dir / "inputs" / "old_qwen_input_messages.json", item.get("llm_input", {}).get("messages") or [])
    (demo_dir / "inputs" / "old_qwen_prompt.txt").write_text(old_prompt)
    (demo_dir / "inputs" / "clean_v1_qwen_prompt.txt").write_text(new_prompt)
    write_json(
        demo_dir / "inputs" / "input_manifest.json",
        {
            "task": task,
            "episode_id": item.get("episode_id"),
            "language_description": item.get("language_description"),
            "old_qwen_runner": item.get("llm_input", {}).get("runner"),
            "old_qwen_model": item.get("llm_input", {}).get("model"),
            "qwen_image_inputs": item.get("image_inputs", {}),
            "local_image_copies": copied_images,
            "note": "This sanity batch reuses the prior Qwen output; it does not rerun Qwen.",
        },
    )

    write_json(demo_dir / "outputs" / "old_qwen_geometry_output.json", old_geometry)
    (demo_dir / "outputs" / "old_qwen_raw_output.txt").write_text(item.get("llm_output", {}).get("raw_output") or "")
    write_json(demo_dir / "outputs" / "clean_v1_primitive_geometry.json", primitive)
    write_json(demo_dir / "outputs" / "clean_v1_contact_hints_stub.json", contact_hints)
    write_json(demo_dir / "outputs" / "human_check_template.json", human_check_template())

    record = {
        "review_id": review_id,
        "task": task,
        "episode_id": item.get("episode_id"),
        "language_description": item.get("language_description"),
        "demo_dir": str(demo_dir.resolve()),
        "inputs": {
            "old_qwen_prompt": str((demo_dir / "inputs" / "old_qwen_prompt.txt").resolve()),
            "clean_v1_qwen_prompt": str((demo_dir / "inputs" / "clean_v1_qwen_prompt.txt").resolve()),
            "old_qwen_messages": str((demo_dir / "inputs" / "old_qwen_input_messages.json").resolve()),
            "images": copied_images,
        },
        "outputs": {
            "old_qwen_geometry": str((demo_dir / "outputs" / "old_qwen_geometry_output.json").resolve()),
            "clean_v1_primitive_geometry": str((demo_dir / "outputs" / "clean_v1_primitive_geometry.json").resolve()),
            "clean_v1_contact_hints_stub": str((demo_dir / "outputs" / "clean_v1_contact_hints_stub.json").resolve()),
            "human_check_template": str((demo_dir / "outputs" / "human_check_template.json").resolve()),
        },
        "primitive_summary": {
            "action_primitive": primitive.get("action_primitive"),
            "motion_type": primitive.get("motion_type"),
            "motion_axis": primitive.get("motion_axis"),
            "target_part": primitive.get("target_part"),
            "constraint_type": primitive.get("constraint_type"),
            "alignment_requirement": primitive.get("alignment_requirement"),
            "execution_clearance_hint": primitive.get("execution_clearance_hint"),
        },
    }
    write_json(demo_dir / "complete_record.json", record)
    (demo_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {review_id}",
                "",
                f"Task: `{item.get('language_description') or task}`",
                "",
                "Check the old Qwen input/output against the clean-v1 primitive descriptor.",
                "The contact-hints file is only a stub here because this batch does not rerun RoboPoint.",
                "",
                "Review files:",
                "- `inputs/old_qwen_prompt.txt`",
                "- `inputs/clean_v1_qwen_prompt.txt`",
                "- `outputs/old_qwen_geometry_output.json`",
                "- `outputs/clean_v1_primitive_geometry.json`",
                "- `outputs/human_check_template.json`",
                "",
            ]
        )
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    payload = read_json(args.source)
    items = payload.get("items", [])[: args.limit]
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    records = [build_item(item, out_root) for item in items]
    write_json(
        out_root / "review_index.json",
        {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_bundle": str(args.source.resolve()),
            "item_count": len(records),
            "purpose": "Sanity check clean-v1 primitive geometry conversion before rerunning the full cache.",
            "records": records,
        },
    )
    with (out_root / "review_bundle.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    with (out_root / "README.md").open("w") as f:
        f.write("# Clean v1 Primitive Geometry Sanity Review Batch\n\n")
        f.write("This batch reuses the prior Qwen geometry cache examples and converts them into the proposed clean-v1 primitive schema. It does not rerun Qwen or RoboPoint.\n\n")
        f.write("Each demo folder contains the old LLM input, old output, new prompt, converted primitive output, and a human-check template.\n\n")
        f.write("| Demo | Task | Action | Motion axis | Target part | Constraint |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for record in records:
            summary = record["primitive_summary"]
            f.write(
                f"| `{record['review_id']}` | {record['language_description']} | "
                f"`{summary['action_primitive']}` | `{summary['motion_axis']}` | "
                f"`{summary['target_part']}` | `{summary['constraint_type']}` |\n"
            )
    print(out_root.resolve())


if __name__ == "__main__":
    main()
