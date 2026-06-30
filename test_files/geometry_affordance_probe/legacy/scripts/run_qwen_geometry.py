import argparse, json, re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

GEOMETRY_PROMPT = """You are extracting a primitive manipulation-geometry descriptor for robot demo retrieval.
Return ONLY valid JSON. Do not include markdown.

Goal:
Describe the object, part, contact region, action primitive, and mechanical constraint needed for the task.
Use only compact primitive labels. Do not describe camera-relative directions such as left, right, front, back, or facing upward.
Only use coarse motion geometry: horizontal, vertical, rotational, into_opening, across_surface, surface_normal, none, unknown.
If unsure, use "unknown" rather than guessing.

Schema:
{
  "manipulated_object": string,
  "object_category": "rigid_object|articulated_or_control|alignment_target|receptacle_or_support|elongated_or_tool|deformable|unknown",
  "primary_shape": "round_object|box_like|thin_flat_object|elongated_tool|button|peg|irregular|unknown",
  "target_part": "handle|lid|rim|knob|button_top|slot|hole|opening|body|edge|spout|socket|surface|unknown",
  "secondary_parts": [string],
  "action_primitive": "push|pull|press|twist|lift|place|insert|slide|sweep|stack|drag|scoop|pour|none|unknown",
  "motion_type": "linear|rotational|vertical|planar|insertion|none|unknown",
  "motion_axis": "horizontal|vertical|rotational|into_opening|across_surface|surface_normal|none|unknown",
  "contact_type": "grasp|pinch|press|surface_contact|tool_contact|none|unknown",
  "contact_region": string,
  "constraint_type": "slot|hole|container|joint|support_surface|surface_target|free_space|none|unknown",
  "alignment_requirement": "none|low|medium|high|unknown",
  "state": "open|closed|attached|detached|inside|on_surface|free|unknown",
  "geometry_tags": [string],
  "execution_clearance_hint": "none|open_path|narrow_path|swing_path|requires_lift|requires_slide_under|unknown"
}

Task instruction: {task}
Use the current observation images and task instruction.
Return the descriptor JSON only.
"""

def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text), text
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0)), text
            except Exception:
                pass
    return {"parse_error": True, "raw_text": text}, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/manifest.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    demos = manifest.get("selected", [])
    if not demos:
        raise SystemExit("No selected demos in manifest. Run sample_seen_demos.py after seen images are available.")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    for demo in demos:
        demo_dir = Path(demo["demo_dir"])
        images = demo.get("review_images") or demo.get("absolute_images")
        content = []
        for img in images[:2]:
            content.append({"type": "image", "image": img})
        prompt = GEOMETRY_PROMPT.replace("{task}", demo.get("language_description", ""))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed, raw = clean_json(decoded)
        if isinstance(parsed, dict):
            parsed.pop("short_rationale", None)
        out = {
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "model": args.model,
            "geometry_g_i": parsed,
            "raw_output": raw,
        }
        with (demo_dir / "geometry_qwen2_5_vl.json").open("w") as f:
            json.dump(out, f, indent=2)
        print("wrote", demo_dir / "geometry_qwen2_5_vl.json")

if __name__ == "__main__":
    main()
