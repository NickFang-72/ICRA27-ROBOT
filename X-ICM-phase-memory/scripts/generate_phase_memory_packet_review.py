#!/usr/bin/env python3
"""Create inspectable phase-memory prompt previews from phase-anchor packets."""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from phase_memory_pipeline.long_term_memory import build_long_term_memory, format_long_term_memory
from phase_memory_pipeline.phase_step_prompt import build_phase_step_user_prompt


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text).rstrip() + "\n")


def phase_packet_dirs(root):
    root = Path(root)
    result = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "05_normalized_anchored_phase_plan.json").exists():
            result.append(child)
    return result


def copy_assets(source_dir, output_dir):
    for name in [
        "query_front_rgb_0.png",
        "query_overhead_rgb_0.png",
        "contact_points_overlay_front_overhead.png",
        "05_normalized_anchored_phase_plan.md",
    ]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def run_one(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_assets(source_dir, output_dir)
    normalized = read_json(source_dir / "05_normalized_anchored_phase_plan.json")
    extracted = read_json(source_dir / "01_extracted_query.json") if (source_dir / "01_extracted_query.json").exists() else {}
    task = extracted.get("task") or normalized.get("task") or source_dir.name
    instruction = extracted.get("instruction") or normalized.get("instruction") or task
    long_term = build_long_term_memory(
        task_name=task,
        instruction=instruction,
        normalized_plan=normalized,
        retrieval_prompt="",
        retrieval_method="preview_no_retrieval_prompt",
    )
    write_json(output_dir / "09_long_term_memory_preview.json", long_term)
    write_text(output_dir / "09_long_term_memory_preview.md", format_long_term_memory(long_term))
    phases = normalized.get("anchored_phase_plan") or []
    if phases:
        prompt = build_phase_step_user_prompt(
            task=task,
            instruction=instruction,
            phase=phases[0],
            all_phases=phases,
            long_term_memory=long_term,
            short_term_memory={
                "current_phase_index": phases[0].get("phase_index"),
                "last_action_7d": None,
                "recent_actions": [],
                "phase_history": [],
                "phase_failures": [],
            },
            observation_summary="preview_only_no_runtime_state",
            max_actions_per_phase=2,
        )
        write_text(output_dir / "10_phase_step_prompt_preview.txt", prompt)
    return {
        "task": task,
        "instruction": instruction,
        "phase_count": len(phases),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-review-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_root = Path(args.phase_review_root).resolve()
    run_name = args.name or datetime.now().strftime("phase_memory_packet_review_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / run_name if args.output_root else source_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    dirs = phase_packet_dirs(source_root)
    if args.limit > 0:
        dirs = dirs[: args.limit]
    for source_dir in dirs:
        out = output_root / source_dir.name
        print(f"Reviewing {source_dir.name}", flush=True)
        rows.append(run_one(source_dir, out))

    with (output_root / "phase_memory_packet_review_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "instruction", "phase_count", "output_dir"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "phase_memory_packet_review_summary.json", rows)
    print(f"Wrote phase-memory packet review to {output_root}", flush=True)


if __name__ == "__main__":
    main()
