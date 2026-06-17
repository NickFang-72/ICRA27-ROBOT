import argparse
import json
import shutil
from pathlib import Path


def task_from_id(item):
    return item["id"].split("_episode", 1)[0]


def load_excluded_ids(paths):
    excluded = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        for demo in payload.get("selected", []):
            excluded.add(demo["id"])
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/train.json")
    ap.add_argument("--data-root", default="/data/yf23/datasets/ICRA27-ROBOT")
    ap.add_argument("--out", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/manifest.json")
    ap.add_argument("--exclude-manifest", action="append", default=[])
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--max-demos", type=int, default=12)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    train_json = Path(args.train_json)
    data_root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    excluded_ids = load_excluded_ids(args.exclude_manifest)

    rows = json.loads(train_json.read_text())
    selected = []
    per_task_counts = {}
    missing_examples = []
    skipped_existing = []

    for item in rows:
        task = task_from_id(item)
        if item["id"] in excluded_ids:
            if len(skipped_existing) < 20:
                skipped_existing.append(item["id"])
            continue
        if per_task_counts.get(task, 0) >= args.per_task:
            continue

        image_paths = [data_root / p for p in item.get("image", [])]
        exists = [p.exists() for p in image_paths]
        record = {
            "id": item["id"],
            "task": task,
            "language_description": item.get("language_description", ""),
            "relative_images": item.get("image", []),
            "absolute_images": [str(p) for p in image_paths],
            "images_exist": exists,
            "conversation": item.get("conversations", []),
            "input_state": "initial/current observation frame only; no future frames or after-state frames",
        }

        if all(exists) and image_paths:
            demo_dir = out.parent / "demos" / item["id"]
            demo_dir.mkdir(parents=True, exist_ok=True)
            record["demo_dir"] = str(demo_dir)
            if args.copy_images:
                record["review_images"] = []
                for idx, p in enumerate(image_paths):
                    dst = demo_dir / f"view{idx}_{p.name}"
                    shutil.copy2(p, dst)
                    record["review_images"].append(str(dst))
            else:
                record["review_images"] = [str(p) for p in image_paths]
            (demo_dir / "demo_metadata.json").write_text(json.dumps(record, indent=2))
            selected.append(record)
            per_task_counts[task] = per_task_counts.get(task, 0) + 1
            if len(selected) >= args.max_demos:
                break
        elif len(missing_examples) < 20:
            missing_examples.append(record)

    payload = {
        "data_root": str(data_root),
        "train_json": str(train_json),
        "num_selected": len(selected),
        "selected": selected,
        "excluded_manifest": args.exclude_manifest,
        "excluded_ids_count": len(excluded_ids),
        "skipped_existing_preview": skipped_existing,
        "missing_examples_preview": missing_examples,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"manifest": str(out), "num_selected": len(selected), "excluded_ids_count": len(excluded_ids)}, indent=2))


if __name__ == "__main__":
    main()
