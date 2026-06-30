#!/usr/bin/env python3
"""Normalize cached geometry outputs into the clean v1 primitive schema.

This compatibility utility rewrites old pilot cache files without rerunning the
models. Qwen geometry output is converted into primitive manipulation geometry
for retrieval; RoboPoint text is converted into contact hints for final-prompt
human review only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cache_all_seen_geometry_affordance import (  # noqa: E402
    affordance_path,
    geometry_path,
    infer_contact_hints,
    infer_primitive_geometry,
    parse_points,
)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = read_json(root / "manifest.json", {"selected": []})
    for demo in manifest.get("selected", []):
        task = demo.get("task") or ""

        g_path = geometry_path(demo)
        if g_path.exists():
            geometry_doc = read_json(g_path, {})
            old_geometry = geometry_doc.get("geometry_g_i") or geometry_doc.get("geometry") or {}
            geometry_doc["geometry_g_i"] = infer_primitive_geometry(task, old_geometry)
            write_json(g_path, geometry_doc)

        c_path = affordance_path(demo)
        if c_path.exists():
            contact_doc = read_json(c_path, {})
            old_contact = contact_doc.get("contact_hints_i") or contact_doc.get("affordance_a_i") or {}
            raw_text = old_contact.get("raw_robopoint_text") or contact_doc.get("raw_text")
            points = old_contact.get("points_2d_normalized") or parse_points(raw_text)
            contact_doc["contact_hints_i"] = infer_contact_hints(
                task,
                points,
                raw_text,
                old_contact.get("vision_tower_note"),
            )
            contact_doc.pop("affordance_a_i", None)
            write_json(c_path, contact_doc)

        print("normalized", demo.get("id", task))


if __name__ == "__main__":
    main()
