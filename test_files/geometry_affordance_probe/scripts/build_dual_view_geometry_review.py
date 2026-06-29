#!/usr/bin/env python3
"""Build Markdown and PNG review artifacts for dual-view Qwen geometry output."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCALAR_FIELDS = [
    "action_primitive",
    "motion_type",
    "target_pose_type",
    "manipulated_object_family",
    "target_object_family",
    "manipulated_part",
    "target_part",
    "articulation_type",
    "required_alignment",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap_lines(text: Any, width: int) -> list[str]:
    if text is None:
        text = "None"
    if isinstance(text, list):
        text = ", ".join(map(str, text))
    return textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False) or [""]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
    width: int,
    max_lines: int | None = None,
    gap: int = 3,
) -> int:
    x, y = xy
    lines = wrap_lines(text, width)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += getattr(font, "size", 14) + gap
    return y


def fit_image(path: str | None, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if not path:
        image = Image.new("RGB", size, "#e5e7eb")
        draw = ImageDraw.Draw(image)
        draw.text((12, height // 2 - 8), "missing", fill="#4b5563", font=load_font(14))
        return image
    try:
        source = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", size, "#e5e7eb")
        draw = ImageDraw.Draw(image)
        draw.text((12, height // 2 - 8), "missing", fill="#4b5563", font=load_font(14))
        return image
    source.thumbnail(size, Image.LANCZOS)
    image = Image.new("RGB", size, "#eef1f4")
    image.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return image


def field_summary(descriptor: dict[str, Any]) -> str:
    return (
        f"action={descriptor.get('action_primitive')}<br>"
        f"motion={descriptor.get('motion_type')}<br>"
        f"pose={descriptor.get('target_pose_type')}<br>"
        f"obj={descriptor.get('manipulated_object_family')}<br>"
        f"target={descriptor.get('target_object_family')}"
    )


def build_markdown(records: list[dict[str, Any]], out_path: Path) -> None:
    conflict_records = [record for record in records if record.get("conflicts")]
    rule_records = [record for record in records if record.get("strict_rule_adjustments")]
    lines = [
        "# Qwen Dual-View Retrieval Geometry Review",
        "",
        "Each row contains the front descriptor, overhead descriptor, and final descriptor. Fusion uses agreement first, filled-over-unknown second, front-on-conflict third, then strict task-language rules for fields that are structurally determined by the instruction.",
        "",
        "## Summary",
        "",
        f"- Records: {len(records)}",
        f"- Records with descriptor conflicts: {len(conflict_records)}",
        f"- Total conflicted fields: {sum(len(record.get('conflicts') or []) for record in records)}",
        f"- Records changed by strict rules: {len(rule_records)}",
        f"- Total strict-rule field adjustments: {sum(len(record.get('strict_rule_adjustments') or []) for record in records)}",
        "",
        "## Compact Overview",
        "",
        "| Task | Front | Overhead | Final | Conflicts | Strict Rules |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        task = record.get("task")
        front = record["view_outputs"]["front"]["normalized_descriptor"]
        overhead = record["view_outputs"]["overhead"]["normalized_descriptor"]
        fused = record["fused_retrieval_geometry"]
        conflicts = ", ".join(conflict["field"] for conflict in record.get("conflicts") or []) or "none"
        rules = ", ".join(
            f"{adjustment['field']}={adjustment['after']}"
            for adjustment in record.get("strict_rule_adjustments") or []
        ) or "none"
        lines.append(
            f"| `{task}` | {field_summary(front)} | {field_summary(overhead)} | "
            f"{field_summary(fused)} | {conflicts} | {rules} |"
        )

    lines.extend(["", "## Per-Demo Results", ""])
    for index, record in enumerate(records, 1):
        image_inputs = record.get("image_inputs") or {}
        front_image = image_inputs.get("local_front_image") or image_inputs.get("front_image")
        overhead_image = image_inputs.get("local_overhead_image") or image_inputs.get("overhead_image")
        lines.extend(
            [
                f"### {index}. `{record.get('task')}`",
                "",
                f"- Instruction: {record.get('language_description')}",
                f"- Front image: [{Path(front_image).name}](<{front_image}>)" if front_image else "- Front image: missing",
                f"- Overhead image: [{Path(overhead_image).name}](<{overhead_image}>)" if overhead_image else "- Overhead image: missing",
                f"- Conflicts: {json.dumps(record.get('conflicts') or [])}",
                f"- Strict rule adjustments: {json.dumps(record.get('strict_rule_adjustments') or [])}",
                "",
                "Front normalized:",
                "",
                "```json",
                json.dumps(record["view_outputs"]["front"]["normalized_descriptor"], indent=2),
                "```",
                "",
                "Overhead normalized:",
                "",
                "```json",
                json.dumps(record["view_outputs"]["overhead"]["normalized_descriptor"], indent=2),
                "```",
                "",
                "Fused descriptor:",
                "",
                "```json",
                json.dumps(record["fused_retrieval_geometry"], indent=2),
                "```",
                "",
                "Source by field:",
                "",
                "```json",
                json.dumps(record["source_by_field"], indent=2),
                "```",
                "",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_png(records: list[dict[str, Any]], out_path: Path) -> None:
    title_font = load_font(38, True)
    subtitle_font = load_font(19)
    task_font = load_font(21, True)
    body_font = load_font(16)
    small_font = load_font(14)

    cols = 1
    card_w = 1720
    card_h = 330
    gap = 26
    margin = 42
    header_h = 150
    rows = (len(records) + cols - 1) // cols
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = header_h + rows * card_h + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, header_h], fill="#101820")
    draw.text((margin, 28), "Qwen Dual-View Retrieval Geometry", fill="white", font=title_font)
    conflicts = sum(len(record.get("conflicts") or []) for record in records)
    rule_adjustments = sum(len(record.get("strict_rule_adjustments") or []) for record in records)
    subtitle = f"{len(records)} demos | front + overhead descriptors | strict rules after fusion | conflicts: {conflicts} | rule adjustments: {rule_adjustments}"
    draw.text((margin, 88), subtitle, fill="#d8e6f3", font=subtitle_font)

    for index, record in enumerate(records):
        row = index // cols
        col = index % cols
        x0 = margin + col * (card_w + gap)
        y0 = header_h + row * (card_h + gap)
        x1 = x0 + card_w
        y1 = y0 + card_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill="white", outline="#cbd5df", width=2)

        title = f"{index + 1}. {record.get('task', '').replace('_', ' ')}"
        draw_wrapped(draw, (x0 + 18, y0 + 18), title, task_font, "#17202a", 52, max_lines=1)
        draw_wrapped(draw, (x0 + 18, y0 + 48), record.get("language_description"), small_font, "#52606d", 60, max_lines=1)

        image_inputs = record.get("image_inputs") or {}
        front_path = image_inputs.get("local_front_image") or image_inputs.get("front_image")
        overhead_path = image_inputs.get("local_overhead_image") or image_inputs.get("overhead_image")
        front = fit_image(front_path, (170, 128))
        overhead = fit_image(overhead_path, (170, 128))
        canvas.paste(front, (x0 + 18, y0 + 86))
        canvas.paste(overhead, (x0 + 208, y0 + 86))
        draw.rectangle([x0 + 18, y0 + 86, x0 + 188, y0 + 214], outline="#b5bec8")
        draw.rectangle([x0 + 208, y0 + 86, x0 + 378, y0 + 214], outline="#b5bec8")
        draw.text((x0 + 18, y0 + 220), "front", fill="#4b5563", font=small_font)
        draw.text((x0 + 208, y0 + 220), "overhead", fill="#4b5563", font=small_font)

        front_desc = record["view_outputs"]["front"]["normalized_descriptor"]
        overhead_desc = record["view_outputs"]["overhead"]["normalized_descriptor"]
        fused = record["fused_retrieval_geometry"]
        source = record["source_by_field"]

        tx = x0 + 410
        ty = y0 + 84
        field_x = tx
        front_x = tx + 185
        overhead_x = tx + 475
        fused_x = tx + 765
        source_x = tx + 1055
        draw.text((field_x, ty), "field", fill="#0f172a", font=body_font)
        draw.text((front_x, ty), "front", fill="#0f172a", font=body_font)
        draw.text((overhead_x, ty), "overhead", fill="#0f172a", font=body_font)
        draw.text((fused_x, ty), "fused", fill="#0f172a", font=body_font)
        draw.text((source_x, ty), "source", fill="#0f172a", font=body_font)
        ty += 26
        rows = [
            ("action", "action_primitive"),
            ("motion", "motion_type"),
            ("target pose", "target_pose_type"),
            ("object family", "manipulated_object_family"),
            ("target family", "target_object_family"),
            ("manipulated part", "manipulated_part"),
            ("target part", "target_part"),
            ("articulation", "articulation_type"),
            ("alignment", "required_alignment"),
        ]
        for label, field in rows:
            draw.text((field_x, ty), label, fill="#64748b", font=small_font)
            draw_wrapped(draw, (front_x, ty), front_desc.get(field), small_font, "#334155", 28, max_lines=1)
            draw_wrapped(draw, (overhead_x, ty), overhead_desc.get(field), small_font, "#334155", 28, max_lines=1)
            draw_wrapped(draw, (fused_x, ty), fused.get(field), small_font, "#111827", 28, max_lines=1)
            draw_wrapped(draw, (source_x, ty), source.get(field), small_font, "#475569", 32, max_lines=1)
            ty += 23

        tag_y = y0 + 252
        draw.text((x0 + 18, tag_y), "fused tags:", fill="#64748b", font=small_font)
        draw_wrapped(draw, (x0 + 100, tag_y), fused.get("geometry_tags"), small_font, "#334155", 62, max_lines=1)

        notes = []
        if record.get("conflicts"):
            notes.append("conflicts: " + ", ".join(conflict["field"] for conflict in record["conflicts"]))
        else:
            notes.append("no descriptor conflicts")
        if record.get("strict_rule_adjustments"):
            rules = ", ".join(
                f"{adjustment['field']}->{adjustment['after']}"
                for adjustment in record["strict_rule_adjustments"][:4]
            )
            if len(record["strict_rule_adjustments"]) > 4:
                rules += ", ..."
            notes.append("strict rules: " + rules)
        front_notes = record["view_outputs"]["front"].get("normalization_notes") or []
        overhead_notes = record["view_outputs"]["overhead"].get("normalization_notes") or []
        if front_notes or overhead_notes:
            notes.append(f"normalized labels: front {len(front_notes)}, overhead {len(overhead_notes)}")
        fill = "#fee2e2" if record.get("conflicts") else "#ecfdf3"
        outline = "#fca5a5" if record.get("conflicts") else "#9fd8b1"
        text_color = "#991b1b" if record.get("conflicts") else "#166534"
        draw.rounded_rectangle([x0 + 18, y1 - 44, x1 - 18, y1 - 14], radius=8, fill=fill, outline=outline)
        draw_wrapped(draw, (x0 + 30, y1 - 39), " | ".join(notes), small_font, text_color, 92, max_lines=1)

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-png", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.bundle).read_text())
    build_markdown(records, Path(args.out_md))
    build_png(records, Path(args.out_png))


if __name__ == "__main__":
    main()
