#!/usr/bin/env python3
"""Build a human review index for clean v1 geometry/contact-hint cache files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = root / "manifest.json"
    bundle_path = root / "human_check_bundle"
    bundle_path.mkdir(parents=True, exist_ok=True)
    rows = []
    combined_rows = []

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    for demo in manifest.get("selected", []):
        demo_dir = Path(demo["demo_dir"])
        geometry_file = demo_dir / "geometry_qwen2_5_vl.json"
        contact_file = demo_dir / "affordance_robopoint.json"
        geometry_doc = read_json(geometry_file) or {}
        contact_doc = read_json(contact_file) or {}

        geometry_g_i = geometry_doc.get("geometry_g_i") or {}
        contact_hints_i = contact_doc.get("contact_hints_i") or contact_doc.get("affordance_a_i") or {}
        raw_robopoint_text = contact_hints_i.get("raw_robopoint_text")
        points = contact_hints_i.get("points_2d_normalized") or parse_points(raw_robopoint_text)

        combined = {
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "demo_dir": str(demo_dir),
            "images": demo.get("review_images") or demo.get("absolute_images") or [],
            "geometry_model": geometry_doc.get("model"),
            "geometry_g_i": geometry_g_i,
            "robopoint_model": contact_doc.get("model"),
            "contact_hints_i": contact_hints_i,
            "source_files": {
                "geometry_qwen2_5_vl": str(geometry_file),
                "contact_hints_robopoint": str(contact_file),
            },
        }
        demo_bundle_dir = bundle_path / demo["id"]
        demo_bundle_dir.mkdir(parents=True, exist_ok=True)
        combined_path = demo_bundle_dir / "combined_review.json"
        write_json(combined_path, combined)
        combined_rows.append(combined)
        rows.append(
            {
                "demo_id": demo["id"],
                "task": demo.get("task"),
                "language_description": demo.get("language_description"),
                "demo_dir": str(demo_dir),
                "images": demo.get("review_images") or demo.get("absolute_images") or [],
                "geometry_file": str(geometry_file),
                "geometry_done": geometry_file.exists(),
                "contact_hints_file": str(contact_file),
                "contact_hints_done": contact_file.exists(),
                "action_primitive": geometry_g_i.get("action_primitive"),
                "motion_type": geometry_g_i.get("motion_type"),
                "motion_axis": geometry_g_i.get("motion_axis"),
                "target_part": geometry_g_i.get("target_part"),
                "constraint_type": geometry_g_i.get("constraint_type"),
                "alignment_requirement": geometry_g_i.get("alignment_requirement"),
                "execution_clearance_hint": geometry_g_i.get("execution_clearance_hint"),
                "contact_mode": contact_hints_i.get("contact_mode"),
                "contact_region_text": contact_hints_i.get("contact_region_text"),
                "points_2d_normalized": points,
                "combined_review_file": str(combined_path),
            }
        )

    write_json(root / "review_index.json", rows)
    with (root / "review_bundle.jsonl").open("w") as f:
        for row in combined_rows:
            f.write(json.dumps(row) + "\n")

    with (root / "review_index.md").open("w") as f:
        f.write("# Clean v1 Geometry/Contact-Hint Human Review Index\n\n")
        f.write(
            "Qwen2.5-VL is used for primitive manipulation geometry. RoboPoint is used only "
            "for optional contact/keypoint hints; these hints are not retrieval scores.\n\n"
        )
        for row in rows:
            f.write(f"## {row['demo_id']}\n\n")
            f.write(f"- Task: {row['language_description']}\n")
            f.write(f"- Demo folder: `{row['demo_dir']}`\n")
            f.write(f"- Geometry, Qwen2.5-VL: `{row['geometry_file']}` ({'done' if row['geometry_done'] else 'missing'})\n")
            f.write(
                "- Primitive fields: "
                f"action=`{row['action_primitive']}`, motion=`{row['motion_type']}`, "
                f"axis=`{row['motion_axis']}`, target_part=`{row['target_part']}`, "
                f"constraint=`{row['constraint_type']}`\n"
            )
            f.write(f"- Clearance hint, final prompt only: `{row['execution_clearance_hint']}`\n")
            f.write(f"- Contact hints, RoboPoint: `{row['contact_hints_file']}` ({'done' if row['contact_hints_done'] else 'missing'})\n")
            f.write(
                f"- Contact mode: `{row['contact_mode']}`, region=`{row['contact_region_text']}`, "
                f"points=`{row['points_2d_normalized']}`\n"
            )
            f.write(f"- Combined review: `{row['combined_review_file']}`\n")
            for image in row["images"]:
                f.write(f"- Image: `{image}`\n")
            f.write("\n")

    print(root / "review_index.md")


if __name__ == "__main__":
    main()
