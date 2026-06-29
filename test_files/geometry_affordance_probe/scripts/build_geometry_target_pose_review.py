#!/usr/bin/env python3
"""Build Markdown and PNG review sheets for cleaned geometry/target-pose Qwen outputs."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


GEOMETRY_ROWS = [
    ("action", "action_primitive"),
    ("motion", "motion_type"),
    ("axis", "motion_axis"),
    ("contact", "contact_type"),
    ("region", "contact_region"),
    ("constraint", "constraint_type"),
    ("align", "alignment_requirement"),
]

TARGET_ROWS = [
    ("goal", "goal_state_type"),
    ("relation", "required_final_relation"),
    ("target", "target_object_or_region"),
    ("orient", "required_orientation_or_alignment"),
    ("stop/release", "release_or_stop_condition"),
    ("success", "success_check"),
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


def local_image(record: dict[str, Any], key: str) -> str | None:
    image_inputs = record.get("image_inputs") or {}
    local_key = f"local_{key}"
    return image_inputs.get(local_key) or image_inputs.get(key)


def compact_descriptor(descriptor: dict[str, Any]) -> str:
    geometry = descriptor.get("geometry") or {}
    target = descriptor.get("target_pose") or {}
    return (
        f"action={geometry.get('action_primitive')}<br>"
        f"motion={geometry.get('motion_type')} / {geometry.get('motion_axis')}<br>"
        f"contact={geometry.get('contact_type')} / {geometry.get('contact_region')}<br>"
        f"constraint={geometry.get('constraint_type')}<br>"
        f"goal={target.get('goal_state_type')}<br>"
        f"relation={target.get('required_final_relation')}"
    )


def build_markdown(records: list[dict[str, Any]], out_path: Path) -> None:
    conflict_count = sum(len(record.get("conflicts") or []) for record in records)
    rule_count = sum(len(record.get("rule_adjustments") or []) for record in records)
    lines = [
        "# Qwen Clean Geometry + Target Pose Quick Review",
        "",
        "Each demo was run through Qwen2.5-VL using the front view and overhead view separately. The fused descriptor uses agreement first, filled-over-unknown second, front-on-conflict third, then strict task-language cleanup for retrieval-critical fields.",
        "",
        "The `geometry` block is intended for retrieval. The `target_pose` block is intended for the final LLM prompt and should not be scored as retrieval similarity.",
        "",
        "## Summary",
        "",
        f"- Records: {len(records)}",
        f"- Total front/overhead field conflicts: {conflict_count}",
        f"- Total strict rule adjustments: {rule_count}",
        "",
        "## Compact Overview",
        "",
        "| Task | Front | Overhead | Fused | Conflicts | Rules |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        front = record["view_outputs"]["front"]["normalized_descriptor"]
        overhead = record["view_outputs"]["overhead"]["normalized_descriptor"]
        fused = record["fused_descriptor"]
        conflicts = ", ".join(conflict["field"] for conflict in record.get("conflicts") or []) or "none"
        rules = ", ".join(adjustment["rule"] for adjustment in record.get("rule_adjustments") or []) or "none"
        lines.append(
            f"| `{record.get('task')}` | {compact_descriptor(front)} | "
            f"{compact_descriptor(overhead)} | {compact_descriptor(fused)} | {conflicts} | {rules} |"
        )

    lines.extend(["", "## Per-Demo Results", ""])
    for index, record in enumerate(records, 1):
        front_image = local_image(record, "front_image")
        overhead_image = local_image(record, "overhead_image")
        item_json = out_path.parent / "items" / f"{record['id']}.json"
        lines.extend(
            [
                f"### {index}. `{record.get('task')}`",
                "",
                f"- Instruction: {record.get('language_description')}",
                f"- Front image: [{Path(front_image).name}](<{front_image}>)" if front_image else "- Front image: missing",
                f"- Overhead image: [{Path(overhead_image).name}](<{overhead_image}>)" if overhead_image else "- Overhead image: missing",
                f"- Item JSON: [{item_json.name}](<{item_json}>)",
                f"- Conflicts: {json.dumps(record.get('conflicts') or [])}",
                f"- Rule adjustments: {json.dumps(record.get('rule_adjustments') or [])}",
                "",
                "Fused descriptor:",
                "",
                "```json",
                json.dumps(record["fused_descriptor"], indent=2),
                "```",
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
            ]
        )
    out_path.write_text("\n".join(lines) + "\n")


def draw_key_values(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    rows: list[tuple[str, str]],
    values: dict[str, Any],
    width_chars: int,
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    draw.text((x, y), title, fill="#0f172a", font=title_font)
    y += 25
    for label, field in rows:
        draw.text((x, y), label, fill="#64748b", font=small_font)
        draw_wrapped(draw, (x + 92, y), values.get(field), small_font, "#17202a", width_chars, max_lines=1)
        y += 22
    return y


def build_png(records: list[dict[str, Any]], out_path: Path) -> None:
    title_font = load_font(36, True)
    subtitle_font = load_font(18)
    task_font = load_font(20, True)
    section_font = load_font(16, True)
    small_font = load_font(14)

    card_w = 1860
    card_h = 390
    gap = 24
    margin = 38
    header_h = 136
    width = margin * 2 + card_w
    height = header_h + len(records) * card_h + (len(records) - 1) * gap + margin

    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, header_h], fill="#101820")
    draw.text((margin, 24), "Qwen Clean Geometry + Target Pose", fill="white", font=title_font)
    conflict_count = sum(len(record.get("conflicts") or []) for record in records)
    rule_count = sum(len(record.get("rule_adjustments") or []) for record in records)
    subtitle = f"{len(records)} demos | front + overhead views | geometry for retrieval | target pose for final prompt | conflicts: {conflict_count} | rule fixes: {rule_count}"
    draw.text((margin, 82), subtitle, fill="#d8e6f3", font=subtitle_font)

    for index, record in enumerate(records):
        x0 = margin
        y0 = header_h + index * (card_h + gap)
        x1 = x0 + card_w
        y1 = y0 + card_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill="white", outline="#cbd5df", width=2)

        task = f"{index + 1}. {record.get('task', '').replace('_', ' ')}"
        draw_wrapped(draw, (x0 + 18, y0 + 16), task, task_font, "#17202a", 58, max_lines=1)
        draw_wrapped(draw, (x0 + 18, y0 + 46), record.get("language_description"), small_font, "#52606d", 74, max_lines=1)

        front_path = local_image(record, "front_image")
        overhead_path = local_image(record, "overhead_image")
        front = fit_image(front_path, (195, 146))
        overhead = fit_image(overhead_path, (195, 146))
        canvas.paste(front, (x0 + 18, y0 + 86))
        canvas.paste(overhead, (x0 + 236, y0 + 86))
        draw.rectangle([x0 + 18, y0 + 86, x0 + 213, y0 + 232], outline="#b5bec8")
        draw.rectangle([x0 + 236, y0 + 86, x0 + 431, y0 + 232], outline="#b5bec8")
        draw.text((x0 + 18, y0 + 238), "front RGB", fill="#4b5563", font=small_font)
        draw.text((x0 + 236, y0 + 238), "overhead RGB", fill="#4b5563", font=small_font)

        fused = record["fused_descriptor"]
        geometry = fused.get("geometry") or {}
        target = fused.get("target_pose") or {}
        gx = x0 + 470
        tx = x0 + 815
        draw_key_values(draw, gx, y0 + 84, "Fused geometry", GEOMETRY_ROWS, geometry, 27, section_font, small_font)
        draw_key_values(draw, tx, y0 + 84, "Fused target pose", TARGET_ROWS, target, 54, section_font, small_font)

        source = record.get("source_by_field") or {}
        sx = x0 + 1325
        sy = y0 + 84
        draw.text((sx, sy), "Fusion sources", fill="#0f172a", font=section_font)
        sy += 25
        source_rows = [
            "geometry.action_primitive",
            "geometry.motion_type",
            "geometry.motion_axis",
            "geometry.contact_region",
            "geometry.constraint_type",
            "target_pose.goal_state_type",
            "target_pose.required_final_relation",
            "target_pose.target_object_or_region",
            "target_pose.required_orientation_or_alignment",
        ]
        for field in source_rows:
            short = field.replace("geometry.", "g.").replace("target_pose.", "t.")
            draw.text((sx, sy), short, fill="#64748b", font=small_font)
            draw_wrapped(draw, (sx + 195, sy), source.get(field), small_font, "#334155", 28, max_lines=1)
            sy += 22

        tag_y = y0 + 274
        uncertain = fused.get("uncertain_fields") or []
        draw.text((x0 + 18, tag_y), "uncertain:", fill="#64748b", font=small_font)
        draw_wrapped(draw, (x0 + 95, tag_y), uncertain or "none", small_font, "#334155", 100, max_lines=1)
        rule_adjustments = record.get("rule_adjustments") or []
        draw.text((x0 + 18, tag_y + 24), "rules:", fill="#64748b", font=small_font)
        rule_names = []
        for adjustment in rule_adjustments:
            if adjustment["rule"] not in rule_names:
                rule_names.append(adjustment["rule"])
        draw_wrapped(draw, (x0 + 65, tag_y + 24), rule_names or "none", small_font, "#334155", 100, max_lines=1)

        notes: list[str] = []
        if record.get("conflicts"):
            notes.append("conflicts: " + ", ".join(conflict["field"] for conflict in record["conflicts"][:6]))
            if len(record["conflicts"]) > 6:
                notes[-1] += ", ..."
        else:
            notes.append("no front/overhead conflicts")
        front_notes = record["view_outputs"]["front"].get("normalization_notes") or []
        overhead_notes = record["view_outputs"]["overhead"].get("normalization_notes") or []
        if front_notes or overhead_notes:
            notes.append(f"normalized labels: front {len(front_notes)}, overhead {len(overhead_notes)}")
        if rule_adjustments:
            notes.append(f"rule fixes: {len(rule_adjustments)}")
        fill = "#fff7ed" if record.get("conflicts") else "#ecfdf3"
        outline = "#fdba74" if record.get("conflicts") else "#9fd8b1"
        text_color = "#9a3412" if record.get("conflicts") else "#166534"
        draw.rounded_rectangle([x0 + 18, y1 - 44, x1 - 18, y1 - 14], radius=8, fill=fill, outline=outline)
        draw_wrapped(draw, (x0 + 30, y1 - 39), " | ".join(notes), small_font, text_color, 142, max_lines=1)

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-png", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.bundle).read_text())
    out_md = Path(args.out_md)
    out_png = Path(args.out_png)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    build_markdown(records, out_md)
    build_png(records, out_png)


if __name__ == "__main__":
    main()
