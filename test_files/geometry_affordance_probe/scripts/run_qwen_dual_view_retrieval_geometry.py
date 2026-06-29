#!/usr/bin/env python3
"""Run Qwen2.5-VL front/overhead retrieval-geometry descriptors and fuse them."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_retrieval_geometry_rules import apply_strict_rules


SCHEMA: dict[str, set[str]] = {
    "action_primitive": {
        "push",
        "pull",
        "press",
        "twist",
        "lift",
        "place",
        "insert",
        "remove",
        "slide",
        "sweep",
        "scoop",
        "pour",
        "open",
        "close",
        "dock",
        "unknown",
    },
    "motion_type": {
        "linear_push",
        "linear_pull",
        "vertical_lift",
        "lift_then_release",
        "planar_slide",
        "rotational_turn",
        "hinge_swing",
        "press_down",
        "align_then_insert",
        "align_then_dock",
        "sweep_across_surface",
        "scoop_under_then_lift",
        "pour_tilt",
        "none",
        "unknown",
    },
    "target_pose_type": {
        "none",
        "support_surface",
        "open_receptacle",
        "slot",
        "peg_or_post",
        "hole_or_socket",
        "base_or_dock",
        "shelf",
        "rack",
        "button_or_switch",
        "hinge_or_joint",
        "screw_or_twist_socket",
        "pour_target",
        "free_space_target",
        "unknown",
    },
    "manipulated_object_family": {
        "rigid_compact_object",
        "thin_rigid_object",
        "elongated_tool",
        "container_or_lid",
        "round_or_cylindrical_object",
        "ring_or_hollow_object",
        "flat_deformable_object",
        "deformable_rope",
        "button_or_switch",
        "door_drawer_or_lid",
        "tool_or_utensil",
        "unknown",
    },
    "target_object_family": {
        "flat_support_surface",
        "open_receptacle",
        "slot_or_hole",
        "peg_or_post",
        "dock_or_base",
        "shelf_or_rack",
        "button_panel",
        "hinged_articulation",
        "drawer_or_container",
        "plant_or_pour_target",
        "free_space",
        "none",
        "unknown",
    },
    "manipulated_part": {
        "body",
        "handle",
        "rim",
        "lid",
        "button_top",
        "knob",
        "edge",
        "tip",
        "opening",
        "hole",
        "cord",
        "surface",
        "none",
        "unknown",
    },
    "target_part": {
        "top_surface",
        "inside_volume",
        "opening",
        "slot",
        "hole",
        "peg_post",
        "dock_cradle",
        "shelf_surface",
        "button_top",
        "handle",
        "hinge_lid",
        "socket",
        "plant_soil",
        "none",
        "unknown",
    },
    "articulation_type": {
        "none",
        "hinge",
        "slider",
        "button",
        "switch",
        "knob",
        "lid",
        "drawer",
        "door",
        "socket",
        "unknown",
    },
    "required_alignment": {
        "none",
        "low",
        "medium",
        "high",
        "unknown",
    },
}

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
LIST_FIELDS = ["geometry_tags", "uncertain_fields"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS

ALIASES: dict[str, dict[str, str]] = {
    "action_primitive": {
        "turn": "twist",
        "rotate": "twist",
        "pick": "lift",
        "pick_up": "lift",
        "put": "place",
        "drop": "place",
        "release": "place",
        "water": "pour",
        "switch": "press",
    },
    "motion_type": {
        "push": "linear_push",
        "pull": "linear_pull",
        "lift": "vertical_lift",
        "place": "lift_then_release",
        "insert": "align_then_insert",
        "dock": "align_then_dock",
        "twist": "rotational_turn",
        "turn": "rotational_turn",
        "rotate": "rotational_turn",
        "press": "press_down",
        "hinge": "hinge_swing",
        "open": "hinge_swing",
        "close": "hinge_swing",
        "sweep": "sweep_across_surface",
        "scoop": "scoop_under_then_lift",
        "pour": "pour_tilt",
    },
    "target_pose_type": {
        "receptacle": "open_receptacle",
        "bin": "open_receptacle",
        "basket": "open_receptacle",
        "container": "open_receptacle",
        "slot_or_hole": "hole_or_socket",
        "slot_or_socket": "hole_or_socket",
        "hole": "hole_or_socket",
        "socket": "hole_or_socket",
        "peg": "peg_or_post",
        "post": "peg_or_post",
        "stand": "peg_or_post",
        "dock": "base_or_dock",
        "base": "base_or_dock",
        "button": "button_or_switch",
        "switch": "button_or_switch",
        "hinge": "hinge_or_joint",
        "joint": "hinge_or_joint",
        "plant_soil": "pour_target",
        "plant": "pour_target",
        "pour": "pour_target",
        "free_space": "free_space_target",
    },
    "manipulated_object_family": {
        "rigid_object": "rigid_compact_object",
        "compact_object": "rigid_compact_object",
        "object": "rigid_compact_object",
        "plate": "thin_rigid_object",
        "flat_rigid_object": "thin_rigid_object",
        "tool": "tool_or_utensil",
        "utensil": "tool_or_utensil",
        "lid": "container_or_lid",
        "container": "container_or_lid",
        "door": "door_drawer_or_lid",
        "drawer": "door_drawer_or_lid",
        "rope": "deformable_rope",
        "cord": "deformable_rope",
        "cable": "deformable_rope",
        "cylindrical_object": "round_or_cylindrical_object",
        "round_object": "round_or_cylindrical_object",
    },
    "target_object_family": {
        "receptacle": "open_receptacle",
        "bin": "open_receptacle",
        "basket": "open_receptacle",
        "container": "drawer_or_container",
        "slot": "slot_or_hole",
        "hole": "slot_or_hole",
        "socket": "slot_or_hole",
        "peg": "peg_or_post",
        "post": "peg_or_post",
        "stand": "peg_or_post",
        "dock": "dock_or_base",
        "base": "dock_or_base",
        "shelf": "shelf_or_rack",
        "rack": "shelf_or_rack",
        "button": "button_panel",
        "switch": "button_panel",
        "hinge": "hinged_articulation",
        "plant": "plant_or_pour_target",
        "plant_soil": "plant_or_pour_target",
    },
    "manipulated_part": {
        "top": "surface",
        "top_surface": "surface",
        "side": "edge",
        "front": "surface",
        "object": "body",
        "none": "none",
    },
    "target_part": {
        "inside": "inside_volume",
        "interior": "inside_volume",
        "top": "top_surface",
        "surface": "top_surface",
        "peg": "peg_post",
        "post": "peg_post",
        "dock": "dock_cradle",
        "cradle": "dock_cradle",
        "button": "button_top",
        "switch": "button_top",
        "plant": "plant_soil",
        "soil": "plant_soil",
    },
    "articulation_type": {
        "none": "none",
        "hinged": "hinge",
        "slide": "slider",
        "sliding": "slider",
        "push_button": "button",
        "switch_button": "switch",
    },
    "required_alignment": {
        "no": "none",
        "not_required": "none",
        "minimal": "low",
        "moderate": "medium",
        "precise": "high",
        "strict": "high",
    },
}

PROMPT_TEMPLATE = """You are extracting a compact robot retrieval-geometry descriptor from one camera view.
Return ONLY valid JSON. Do not include markdown.

Goal:
Describe the reusable action structure needed to retrieve similar robot demonstrations.
Prioritize the action primitive and target/goal pose over exact object identity.
Use only the allowed enum labels below. Do not invent new labels. If no allowed label fits, use "unknown".
Do not use camera-relative directions like left, right, front, back, facing up, or facing down.
Do not include clearance/path-planning hints; those belong in the final robot prompt, not retrieval.

Decision rules:
- action_primitive is the main retrieval key: the physical action the robot should perform.
- target_pose_type describes the final relation after success, especially for place, insert, dock, pour, open, and close tasks.
- motion_type describes the geometric movement pattern, not the object category.
- object family fields should be primitive families, not exact object names.
- articulation_type is only for joints, hinges, buttons, doors, drawers, sockets, knobs, lids, or switches.
- required_alignment should be high when success requires seating, docking, inserting, pouring into/onto a target, aligning with a peg/post/hole/socket, or placing inside a constrained target.
- Always include every schema field. Use "unknown" or [] when unsure.

Allowed schema:
{
  "action_primitive": "push|pull|press|twist|lift|place|insert|remove|slide|sweep|scoop|pour|open|close|dock|unknown",
  "motion_type": "linear_push|linear_pull|vertical_lift|lift_then_release|planar_slide|rotational_turn|hinge_swing|press_down|align_then_insert|align_then_dock|sweep_across_surface|scoop_under_then_lift|pour_tilt|none|unknown",
  "target_pose_type": "none|support_surface|open_receptacle|slot|peg_or_post|hole_or_socket|base_or_dock|shelf|rack|button_or_switch|hinge_or_joint|screw_or_twist_socket|pour_target|free_space_target|unknown",
  "manipulated_object_family": "rigid_compact_object|thin_rigid_object|elongated_tool|container_or_lid|round_or_cylindrical_object|ring_or_hollow_object|flat_deformable_object|deformable_rope|button_or_switch|door_drawer_or_lid|tool_or_utensil|unknown",
  "target_object_family": "flat_support_surface|open_receptacle|slot_or_hole|peg_or_post|dock_or_base|shelf_or_rack|button_panel|hinged_articulation|drawer_or_container|plant_or_pour_target|free_space|none|unknown",
  "manipulated_part": "body|handle|rim|lid|button_top|knob|edge|tip|opening|hole|cord|surface|none|unknown",
  "target_part": "top_surface|inside_volume|opening|slot|hole|peg_post|dock_cradle|shelf_surface|button_top|handle|hinge_lid|socket|plant_soil|none|unknown",
  "articulation_type": "none|hinge|slider|button|switch|knob|lid|drawer|door|socket|unknown",
  "required_alignment": "none|low|medium|high|unknown",
  "geometry_tags": [short primitive tags],
  "uncertain_fields": [schema field names that are uncertain]
}

Calibration examples:
- "turn off the light" -> action_primitive press, motion_type press_down, target_pose_type button_or_switch.
- "open the grill" -> action_primitive open, motion_type hinge_swing, target_pose_type hinge_or_joint, articulation_type hinge.
- "put the knife on the chopping board" -> action_primitive place, motion_type lift_then_release, target_pose_type support_surface.
- "put the phone on the base" -> action_primitive dock, motion_type align_then_dock, target_pose_type base_or_dock.
- "put the rubbish in the bin" -> action_primitive place, motion_type lift_then_release, target_pose_type open_receptacle.
- "put the toilet roll on the stand" -> action_primitive insert, motion_type align_then_insert, target_pose_type peg_or_post.
- "water the plant" -> action_primitive pour, motion_type pour_tilt, target_pose_type pour_target, target_part plant_soil.
- "take the USB out of the computer" -> action_primitive remove, motion_type linear_pull, target_pose_type none, target_part socket.

Task instruction: {task}
Camera view: {view}
Return the descriptor JSON only.
"""


def clean_json(text: str) -> tuple[dict[str, Any], str]:
    raw = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", raw).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw_json": parsed}, raw
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw_json": parsed}, raw
            except Exception:
                pass
    return {"parse_error": True, "raw_text": raw}, raw


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("examples") or data.get("records") or data.get("selected") or []
    raise SystemExit(f"Unsupported manifest format: {path}")


def norm_label(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        value = value[0] if value else "unknown"
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def normalize_scalar(field: str, value: Any, notes: list[str]) -> str:
    text = norm_label(value)
    if text in SCHEMA[field]:
        return text
    aliased = ALIASES.get(field, {}).get(text)
    if aliased and aliased in SCHEMA[field]:
        notes.append(f"{field}: {text} -> {aliased}")
        return aliased
    notes.append(f"{field}: {text} -> unknown")
    return "unknown"


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        tag = norm_label(item)
        if tag and tag != "unknown" and tag not in out:
            out.append(tag)
    return out[:8]


def normalize_descriptor(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    if raw.get("parse_error"):
        raw = {}
        notes.append("parse_error -> empty descriptor")
    normalized: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        normalized[field] = normalize_scalar(field, raw.get(field), notes)
    normalized["geometry_tags"] = normalize_list(raw.get("geometry_tags"))
    uncertain = normalize_list(raw.get("uncertain_fields"))
    normalized["uncertain_fields"] = [field for field in uncertain if field in ALL_FIELDS]
    return normalized, notes


def known(value: Any) -> bool:
    return value not in (None, "", "unknown", [], ["unknown"])


def fuse_descriptors(front: dict[str, Any], overhead: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    fused: dict[str, Any] = {}
    source: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for field in SCALAR_FIELDS:
        f_value = front.get(field, "unknown")
        o_value = overhead.get(field, "unknown")
        if f_value == o_value:
            fused[field] = f_value
            source[field] = "agreement" if known(f_value) else "both_unknown"
        elif known(f_value) and not known(o_value):
            fused[field] = f_value
            source[field] = "front_filled_overhead_unknown"
        elif known(o_value) and not known(f_value):
            fused[field] = o_value
            source[field] = "overhead_filled_front_unknown"
        elif known(f_value) and known(o_value):
            fused[field] = f_value
            source[field] = "front_on_conflict"
            conflicts.append({"field": field, "front": f_value, "overhead": o_value, "chosen": f_value})
        else:
            fused[field] = "unknown"
            source[field] = "both_unknown"

    front_tags = normalize_list(front.get("geometry_tags"))
    overhead_tags = normalize_list(overhead.get("geometry_tags"))
    fused["geometry_tags"] = front_tags if front_tags else overhead_tags
    if front_tags and overhead_tags and front_tags != overhead_tags:
        source["geometry_tags"] = "front_on_conflict"
        conflicts.append({"field": "geometry_tags", "front": front_tags, "overhead": overhead_tags, "chosen": front_tags})
    elif front_tags == overhead_tags and front_tags:
        source["geometry_tags"] = "agreement"
    elif front_tags:
        source["geometry_tags"] = "front_filled_overhead_unknown"
    elif overhead_tags:
        source["geometry_tags"] = "overhead_filled_front_unknown"
    else:
        source["geometry_tags"] = "both_unknown"

    uncertain = sorted(set(normalize_list(front.get("uncertain_fields")) + normalize_list(overhead.get("uncertain_fields"))))
    fused["uncertain_fields"] = [field for field in uncertain if field in ALL_FIELDS]
    source["uncertain_fields"] = "union"
    return fused, source, conflicts


def run_view(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    image_path: str,
    task_instruction: str,
    view: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.replace("{task}", task_instruction).replace("{view}", view)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed, raw = clean_json(decoded)
    normalized, notes = normalize_descriptor(parsed)
    return {
        "image": image_path,
        "prompt_text": prompt,
        "messages": messages,
        "raw_descriptor": parsed,
        "raw_output": raw,
        "normalized_descriptor": normalized,
        "normalization_notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_dir = out_dir / "items"
    items_dir.mkdir(exist_ok=True)

    examples = read_manifest(manifest_path)
    if not examples:
        raise SystemExit(f"No examples in {manifest_path}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    records: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for example in examples:
        review_id = example["id"]
        task_instruction = example["language_description"]
        views = {
            "front": run_view(
                processor,
                model,
                example["front_image"],
                task_instruction,
                "front",
                args.max_new_tokens,
            ),
            "overhead": run_view(
                processor,
                model,
                example["overhead_image"],
                task_instruction,
                "overhead",
                args.max_new_tokens,
            ),
        }
        fused_before_rules, source_by_field, conflicts = fuse_descriptors(
            views["front"]["normalized_descriptor"],
            views["overhead"]["normalized_descriptor"],
        )
        fused, strict_rule_adjustments = apply_strict_rules(
            example.get("task"),
            task_instruction,
            fused_before_rules,
        )
        for adjustment in strict_rule_adjustments:
            source_by_field[adjustment["field"]] = f"strict_rule:{adjustment['rule']}"
        record = {
            "id": review_id,
            "task": example.get("task"),
            "language_description": task_instruction,
            "model": args.model,
            "image_inputs": {
                "front_image": example["front_image"],
                "overhead_image": example["overhead_image"],
                "local_front_image": example.get("local_front_image"),
                "local_overhead_image": example.get("local_overhead_image"),
            },
            "view_outputs": views,
            "fused_before_strict_rules": fused_before_rules,
            "fused_retrieval_geometry": fused,
            "source_by_field": source_by_field,
            "strict_rule_adjustments": strict_rule_adjustments,
            "conflicts": conflicts,
            "human_check": {
                "front_descriptor_ok": None,
                "overhead_descriptor_ok": None,
                "fused_descriptor_ok": None,
                "front_on_conflict_reasonable": None,
                "strict_rules_reasonable": None,
                "useful_for_retrieval": None,
                "notes": "",
            },
        }
        (items_dir / f"{review_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        compact.append(
            {
                "task": example.get("task"),
                "language_description": task_instruction,
                "front": views["front"]["normalized_descriptor"],
                "overhead": views["overhead"]["normalized_descriptor"],
                "fused_before_strict_rules": fused_before_rules,
                "fused": fused,
                "source_by_field": source_by_field,
                "strict_rule_adjustments": strict_rule_adjustments,
                "conflicts": conflicts,
            }
        )
        print("wrote", items_dir / f"{review_id}.json")

    (out_dir / "qwen_dual_view_retrieval_geometry_bundle.json").write_text(json.dumps(records, indent=2) + "\n")
    (out_dir / "qwen_dual_view_retrieval_geometry_task_results.json").write_text(json.dumps(compact, indent=2) + "\n")
    with (out_dir / "qwen_dual_view_retrieval_geometry_bundle.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
