import argparse, json, os, subprocess, sys
from pathlib import Path

CONTACT_HINT_SCHEMA = {
    "contact_mode": "single_contact | grasp_pair | none",
    "source_view": "front_rgb_initial",
    "target_object": "Task object named by primitive geometry/task rule",
    "target_part": "Task part/contact region named by primitive geometry/task rule",
    "points_2d_normalized": "RoboPoint normalized image points, if produced",
    "contact_region_text": "Short RoboPoint phrase naming contact region",
    "raw_robopoint_text": "Original RoboPoint answer"
}

QUESTION_TEMPLATE = """<image>
Task instruction: {task}
Primitive action: {action_primitive}
Contact mode: {contact_mode}
Target object: {target_object}
Target part: {target_part}

For robot manipulation, identify the best visible contact points for accomplishing this task.
If contact_mode is single_contact, return one normalized image point.
If contact_mode is grasp_pair, return two normalized image points for the two parallel-gripper finger contacts when visible.
If the contact is not visible or unreliable, return no points and say none.
Use normalized tuples [(x1, y1), ...] where x and y are between 0 and 1. After the list, add a short phrase naming the contact region only."""


TASK_CONTACT_RULES = {
    "turn_tap": ("twist", "grasp_pair", "tap", "tap_handle_or_knob"),
    "close_jar": ("twist", "grasp_pair", "jar", "lid_or_rim"),
    "light_bulb_in": ("insert", "grasp_pair", "light_bulb", "bulb_body_or_base"),
    "slide_block_to_color_target": ("slide", "single_contact", "block", "block_side_or_top"),
    "sweep_to_dustpan_of_size": ("sweep", "single_contact", "dustpan", "sweeping_tool_or_object_side"),
    "push_buttons": ("press", "single_contact", "button", "button_top"),
    "put_groceries_in_cupboard": ("place", "grasp_pair", "groceries_and_cupboard", "object_body"),
    "put_money_in_safe": ("insert", "grasp_pair", "money_and_safe", "thin_object_edge"),
    "place_shape_in_shape_sorter": ("insert", "grasp_pair", "shape_and_shape_sorter", "shape_body"),
    "put_item_in_drawer": ("place", "grasp_pair", "item_and_drawer", "object_body"),
    "insert_onto_square_peg": ("insert", "grasp_pair", "shape_and_square_peg", "object_body"),
    "open_drawer": ("pull", "grasp_pair", "drawer", "handle"),
}


def contact_rule(task):
    return TASK_CONTACT_RULES.get(task, ("unknown", "none", task or "unknown", "unknown"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/manifest.json")
    ap.add_argument("--model-path", default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b")
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--temperature", default="0")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open() as f:
        manifest = json.load(f)
    demos = manifest.get("selected", [])
    if not demos:
        raise SystemExit("No selected demos in manifest. Run sample_seen_demos.py after seen images are available.")

    root = manifest_path.parent
    image_folder = root / "robopoint_images"
    image_folder.mkdir(parents=True, exist_ok=True)
    question_file = root / "robopoint_questions.jsonl"
    answer_file = root / "robopoint_answers.jsonl"

    with question_file.open("w") as f:
        for demo in demos:
            # Use the front view when available; train.json stores front first, wrist second.
            src = Path((demo.get("review_images") or demo.get("absolute_images"))[0])
            dst_name = f"{demo['id']}_{src.name}"
            dst = image_folder / dst_name
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            action_primitive, contact_mode, target_object, target_part = contact_rule(demo.get("task"))
            row = {
                "question_id": demo["id"],
                "image": dst_name,
                "text": QUESTION_TEMPLATE.format(
                    task=demo.get("language_description", ""),
                    action_primitive=action_primitive,
                    contact_mode=contact_mode,
                    target_object=target_object,
                    target_part=target_part,
                ),
                "category": "agnostos_contact_hint_probe",
            }
            f.write(json.dumps(row) + "\n")

    cmd = [
        sys.executable, "-m", "robopoint.eval.model_vqa",
        "--model-path", args.model_path,
        "--image-folder", str(image_folder),
        "--question-file", str(question_file),
        "--answer-file", str(answer_file),
        "--conv-mode", args.conv_mode,
        "--temperature", str(args.temperature),
    ]
    print("running", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/data/yf23/checkpoints/ICRA27-ROBOT/hf_home")
    subprocess.run(cmd, check=True, env=env)

    answers = {}
    with answer_file.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                answers[row["question_id"]] = row

    for demo in demos:
        ans = answers.get(demo["id"], {})
        action_primitive, contact_mode, target_object, target_part = contact_rule(demo.get("task"))
        out = {
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "model": args.model_path,
            "images": demo.get("review_images") or demo.get("absolute_images"),
            "contact_hints_i": {
                "contact_mode": contact_mode,
                "source_view": "front_rgb_initial",
                "target_object": target_object,
                "target_part": target_part,
                "raw_robopoint_text": ans.get("text"),
                "expected_schema": CONTACT_HINT_SCHEMA,
                "note": "RoboPoint predicts contact/keypoint hints only. These are not symbolic affordance descriptors and are not used for retrieval scoring.",
                "vision_tower_note": "Pilot run uses the RoboPoint LLM checkpoint with a local CLIP ViT-L/14 vision tower fallback on CAIR because openai/clip-vit-large-patch14-336 PyTorch weights could not be fetched through Hugging Face SSL. Treat affordance outputs as human-check candidates.",
            },
            "raw_answer_record": ans,
        }
        demo_dir = Path(demo["demo_dir"])
        with (demo_dir / "affordance_robopoint.json").open("w") as f:
            json.dump(out, f, indent=2)
        print("wrote", demo_dir / "affordance_robopoint.json")

if __name__ == "__main__":
    main()
