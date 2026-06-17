import argparse, json, os, subprocess, sys
from pathlib import Path

AFFORDANCE_SCHEMA = {
    "grasp_affordance": "handle_grasp | knob_grasp | rim_grasp | body_grasp | edge_grasp | pinch | none | unknown",
    "contact_affordance": "push_surface | pull_handle | lift_top | rotate_part | slide_part | insert_object | unknown",
    "motion_affordance": "push | pull | lift | rotate | slide | twist | insert | scoop | pour | unknown",
    "required_contact_region": "handle | knob | rim | top | side | front_face | free_end | unknown",
    "preferred_contact_points": "RoboPoint normalized image points, if produced",
    "raw_robopoint_text": "Original RoboPoint answer"
}

QUESTION_TEMPLATE = """<image>\nTask instruction: {task}\nFor robot manipulation, identify the best visible contact or grasp affordance points for accomplishing this task. Return several normalized image keypoints as a list of tuples [(x1, y1), ...] where x and y are between 0 and 1. After the list, add a short phrase naming the contact region and motion affordance."""


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
            row = {
                "question_id": demo["id"],
                "image": dst_name,
                "text": QUESTION_TEMPLATE.format(task=demo.get("language_description", "")),
                "category": "agnostos_affordance_probe",
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
        out = {
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "model": args.model_path,
            "images": demo.get("review_images") or demo.get("absolute_images"),
            "affordance_a_i": {
                "raw_robopoint_text": ans.get("text"),
                "expected_schema": AFFORDANCE_SCHEMA,
                "note": "RoboPoint primarily predicts spatial affordance keypoints; symbolic affordance labels should be human-checked or post-processed.",
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
