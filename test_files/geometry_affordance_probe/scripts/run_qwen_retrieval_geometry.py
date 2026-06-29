#!/usr/bin/env python3
"""Run Qwen2.5-VL on the compact retrieval-geometry schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


RETRIEVAL_GEOMETRY_PROMPT = """You are extracting a compact robot retrieval-geometry descriptor.
Return ONLY valid JSON. Do not include markdown.

Goal:
Describe the reusable action structure needed to retrieve similar robot demos.
Prioritize the action primitive and final target pose over exact object category.
Use compact primitive labels that can compare an unseen query to seen demonstrations.
Do not use camera-relative directions like left, right, front, back, facing up, or facing down.
Do not include clearance/path-planning hints; those belong in the final robot prompt, not retrieval.
If unsure, use "unknown" rather than guessing.

Decision rules:
- action_primitive is the main retrieval key. Choose the physical action: push, pull, press, twist, lift, place, insert, remove, slide, sweep, scoop, pour, open, close, dock, or unknown.
- target_pose_type describes the goal relation after success, especially for place/insert/dock tasks.
- motion_type describes the geometric movement pattern, not the object category.
- manipulated_object_family and target_object_family should be primitive families, not exact names when a family label is enough.
- articulation_type is only for objects with joints, hinges, buttons, lids, doors, drawers, switches, knobs, or similar mechanisms.
- required_alignment should increase when success requires seating, docking, inserting through an opening, aligning with a peg/post, or placing inside a constrained target.

Schema:
{
  "action_primitive": "push|pull|press|twist|lift|place|insert|remove|slide|sweep|scoop|pour|open|close|dock|unknown",
  "motion_type": "linear_push|linear_pull|vertical_lift|lift_then_release|planar_slide|rotational_turn|hinge_swing|press_down|align_then_insert|align_then_dock|sweep_across_surface|scoop_under_then_lift|pour_tilt|none|unknown",
  "target_pose_type": "none|support_surface|receptacle|slot|peg_or_post|hole_or_socket|base_or_dock|shelf|bin|rack|button_or_switch|hinge_or_joint|screw_or_twist_socket|free_space_target|unknown",
  "manipulated_object_family": "rigid_compact_object|thin_rigid_object|elongated_tool|container_or_lid|round_or_cylindrical_object|ring_or_hollow_object|flat_deformable_object|deformable_rope|button_or_switch|door_drawer_or_lid|tool_or_utensil|unknown",
  "target_object_family": "flat_support_surface|open_receptacle|slot_or_hole|peg_or_post|dock_or_base|shelf_or_rack|button_panel|hinged_articulation|drawer_or_container|plant_or_pour_target|free_space|none|unknown",
  "manipulated_part": "body|handle|rim|lid|button_top|knob|edge|tip|opening|hole|cord|surface|unknown",
  "target_part": "top_surface|inside_volume|opening|slot|hole|peg_post|dock_cradle|shelf_surface|button_top|handle|hinge_lid|socket|plant_soil|none|unknown",
  "articulation_type": "none|hinge|slider|button|switch|knob|lid|drawer|door|socket|unknown",
  "required_alignment": "none|low|medium|high|unknown",
  "geometry_tags": [string],
  "uncertain_fields": [string]
}

Calibration examples:
- "turn off the light" -> action_primitive press, motion_type press_down, target_pose_type button_or_switch, manipulated_object_family button_or_switch.
- "open the grill" -> action_primitive open, motion_type hinge_swing, target_pose_type hinge_or_joint, articulation_type hinge.
- "put the knife on the chopping board" -> action_primitive place, motion_type lift_then_release, target_pose_type support_surface, manipulated_object_family thin_rigid_object, target_object_family flat_support_surface.
- "put the phone on the base" -> action_primitive dock, motion_type align_then_dock, target_pose_type base_or_dock, target_object_family dock_or_base, required_alignment high.
- "put the rubbish in the bin" -> action_primitive place, motion_type lift_then_release, target_pose_type bin, target_object_family open_receptacle.
- "put the toilet roll on the stand" -> action_primitive insert, motion_type align_then_insert, target_pose_type peg_or_post, manipulated_object_family ring_or_hollow_object, required_alignment high.
- "take the USB out of the computer" -> action_primitive remove, motion_type linear_pull, target_pose_type none, target_part socket.

Task instruction: {task}
Use the current front observation image and task instruction.
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
        return data.get("examples") or data.get("selected") or []
    raise SystemExit(f"Unsupported manifest format: {path}")


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
        image_path = example["front_image"]
        prompt = RETRIEVAL_GEOMETRY_PROMPT.replace("{task}", task_instruction)
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
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed, raw = clean_json(decoded)
        record = {
            "id": review_id,
            "task": example.get("task"),
            "language_description": task_instruction,
            "model": args.model,
            "image_inputs": {"front_image": image_path},
            "llm_input": {"prompt_text": prompt, "messages": messages},
            "llm_output": {
                "retrieval_geometry": parsed,
                "raw_output": raw,
            },
            "human_check": {
                "correct_action_primitive": None,
                "correct_motion_type": None,
                "correct_target_pose_type": None,
                "correct_object_families": None,
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
                "result": parsed,
            }
        )
        print("wrote", items_dir / f"{review_id}.json")

    (out_dir / "qwen_retrieval_geometry_review_bundle.json").write_text(json.dumps(records, indent=2) + "\n")
    (out_dir / "qwen_retrieval_geometry_task_results.json").write_text(json.dumps(compact, indent=2) + "\n")
    with (out_dir / "qwen_retrieval_geometry_review_bundle.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
