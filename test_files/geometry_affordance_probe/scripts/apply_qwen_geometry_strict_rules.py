#!/usr/bin/env python3
"""Apply strict retrieval-geometry rules to an existing Qwen bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_retrieval_geometry_rules import apply_strict_rules


def apply_to_record(record: dict[str, Any]) -> dict[str, Any]:
    before = (
        record.get("fused_before_strict_rules")
        or record.get("fused_retrieval_geometry")
        or record.get("result")
        or {}
    )
    final, adjustments = apply_strict_rules(
        record.get("task"),
        record.get("language_description"),
        before,
    )
    record["fused_before_strict_rules"] = before
    record["fused_retrieval_geometry"] = final
    record["strict_rule_adjustments"] = adjustments
    source = dict(record.get("source_by_field") or {})
    for adjustment in adjustments:
        source[adjustment["field"]] = f"strict_rule:{adjustment['rule']}"
    record["source_by_field"] = source
    return record


def compact_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for record in records:
        compact.append(
            {
                "task": record.get("task"),
                "language_description": record.get("language_description"),
                "front": (record.get("view_outputs") or {}).get("front", {}).get("normalized_descriptor"),
                "overhead": (record.get("view_outputs") or {}).get("overhead", {}).get("normalized_descriptor"),
                "fused_before_strict_rules": record.get("fused_before_strict_rules"),
                "fused": record.get("fused_retrieval_geometry"),
                "source_by_field": record.get("source_by_field"),
                "strict_rule_adjustments": record.get("strict_rule_adjustments"),
                "conflicts": record.get("conflicts") or [],
            }
        )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Input bundle JSON.")
    parser.add_argument("--out-bundle", help="Output bundle JSON. Defaults to --bundle.")
    parser.add_argument("--out-compact", help="Optional compact task-results JSON.")
    parser.add_argument("--items-dir", help="Optional items directory to update by record id.")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    out_bundle = Path(args.out_bundle) if args.out_bundle else bundle_path
    records = json.loads(bundle_path.read_text())
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list bundle: {bundle_path}")

    updated = [apply_to_record(dict(record)) for record in records]
    out_bundle.write_text(json.dumps(updated, indent=2) + "\n")

    if args.out_compact:
        Path(args.out_compact).write_text(json.dumps(compact_from_records(updated), indent=2) + "\n")

    if args.items_dir:
        items_dir = Path(args.items_dir)
        for record in updated:
            record_id = record.get("id")
            if not record_id:
                continue
            item_path = items_dir / f"{record_id}.json"
            if item_path.exists():
                item_path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"updated records: {len(updated)}")
    print(f"total strict rule adjustments: {sum(len(record.get('strict_rule_adjustments') or []) for record in updated)}")


if __name__ == "__main__":
    main()
