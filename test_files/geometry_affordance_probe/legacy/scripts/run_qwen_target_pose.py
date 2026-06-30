#!/usr/bin/env python3
"""Run Qwen2.5-VL target-pose descriptor checks for a small review batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


TARGET_POSE_PROMPT = """You are extracting a target-pose descriptor for robot placement-task retrieval.
Return ONLY valid JSON. Do not include markdown.

Goal:
Describe the desired final pose/relation of the manipulated object after the task succeeds.
Infer the final goal state from the task instruction plus the image.
Do NOT describe where the object is currently resting unless that is already the requested final state.
Do NOT use "on_surface" just because the target object is sitting on a table in the image.
Do not describe the object's visual geometry unless it affects the final target pose.
Use compact primitive labels so descriptors can be compared across unseen and seen tasks.
If unsure, use "unknown" rather than guessing.

Decision rules:
- If the task says "in/inside <bin, stand, basket, cupboard, drawer, safe>", prefer inside_receptacle or inserted_in_slot over on_surface.
- If the task says "on <base, dock, charger, cradle>", prefer docked with support_type cradle_or_dock.
- If the task says "on <peg, spoke, stand, rod>" and the object has a hole/ring/roll, prefer on_peg_or_spoke with containment_requirement around_support.
- If the task says "on <board, table, shelf, surface>" and the object should rest flat, use on_surface with support_type flat_surface or shelf.
- If the task says "books on bookshelf" or similar, use on_surface/shelf unless the goal is explicitly stacking multiple movable objects.

Release-condition rule:
The "release_condition" must be a short concrete final-state test for when the robot can open the gripper.
Do not write "unknown" unless the target object or target site cannot be identified.
Use physical language such as:
- object rests flat within the target boundary
- object is inside the receptacle opening
- object is seated/aligned in the dock
- object's hole is around the peg/post/spoke
- object is upright inside the stand
- object is supported by the shelf surface

Calibration examples:
- "put the phone on the base" -> docked, cradle_or_dock, dock, aligned_with_dock, seated_in, release_condition: phone is seated and stable in the base/dock.
- "put the knife on the chopping board" -> on_surface, flat_surface, lay_flat, flat, within_boundary, release_condition: knife rests flat within the chopping board boundary.
- "put the rubbish in the bin" -> inside_receptacle, open_receptacle, drop_or_place_inside, any, inside, release_condition: rubbish is fully inside the bin opening.
- "put the umbrella in the umbrella stand" -> inside_receptacle, vertical_stand, insert, upright, object_axis_inside_receptacle, release_condition: umbrella shaft is upright inside the stand.
- "put the toilet roll on the stand" -> on_peg_or_spoke, peg_or_spoke, thread_onto, hole_aligned_with_post, around_peg, release_condition: toilet roll center hole is around the stand post.
- "put the books on the bookshelf" -> on_surface, shelf, lay_flat, flat, on_support, release_condition: books are supported by the shelf surface.

Schema:
{
  "manipulated_object": string,
  "target_object_or_region": string,
  "target_pose_type": "on_surface|inside_receptacle|inserted_in_slot|on_peg_or_spoke|docked|nested|stacked|hung|mounted|leaning|unknown",
  "support_type": "flat_surface|open_receptacle|vertical_stand|slot|hole|peg_or_spoke|cradle_or_dock|shelf|rack|hook|container_interior|unknown",
  "placement_mode": "lay_flat|place_on_top|drop_or_place_inside|insert|stand_upright_inside|thread_onto|dock|nest|stack|hang|slide_into|unknown",
  "required_object_orientation": "any|flat|upright|vertical|horizontal|aligned_with_slot|aligned_with_dock|hole_aligned_with_post|opening_up|handle_accessible|unknown",
  "required_spatial_relation": "centered_on|within_boundary|inside|through_opening|seated_in|around_peg|object_axis_inside_receptacle|on_support|against_surface|unknown",
  "alignment_requirement": "none|low|medium|high|unknown",
  "containment_requirement": "none|inside_target|partially_inserted|fully_inserted|seated_in_target|around_support|unknown",
  "release_condition": string,
  "release_condition_confidence": "low|medium|high|unknown",
  "target_pose_tags": [string],
  "uncertain_fields": [string]
}

Task instruction: {task}
Use the image and task instruction to infer the desired final pose.
Return the descriptor JSON only.
"""


def clean_json(text: str) -> tuple[dict[str, Any], str]:
    raw = text.strip()
    text = re.sub(r"^```(?:json)?", "", raw).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw_json": parsed}, raw
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
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
    for example in examples:
        review_id = example["id"]
        task = example["language_description"]
        image_path = example["front_image"]
        prompt = TARGET_POSE_PROMPT.replace("{task}", task)
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
            "language_description": task,
            "model": args.model,
            "image_inputs": {
                "front_image": image_path,
            },
            "llm_input": {
                "prompt_text": prompt,
                "messages": messages,
            },
            "llm_output": {
                "target_pose_descriptor": parsed,
                "raw_output": raw,
            },
            "human_check": {
                "correct_target_object_or_region": None,
                "correct_target_pose_type": None,
                "correct_required_orientation": None,
                "correct_release_condition": None,
                "notes": "",
            },
        }
        (items_dir / f"{review_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print("wrote", items_dir / f"{review_id}.json")

    (out_dir / "target_pose_review_bundle.json").write_text(json.dumps(records, indent=2) + "\n")
    with (out_dir / "target_pose_review_bundle.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
