#!/usr/bin/env python3
"""Bundle phase-memory runtime inputs/outputs into inspectable demo folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def find_runtime_manifests(log_root: Path, task_filter: Optional[Set[str]]) -> List[Path]:
    manifests = sorted(log_root.rglob("phase_memory_runtime/*/episode*/00_phase_memory_manifest.json"))
    if task_filter is None:
        return manifests
    kept = []
    for manifest in manifests:
        try:
            data = read_json(manifest)
        except Exception:
            continue
        if str(data.get("task") or "") in task_filter:
            kept.append(manifest)
    return kept


def iter_message_image_paths(value: Any) -> Iterable[Path]:
    if isinstance(value, dict):
        if value.get("type") == "image" and value.get("image"):
            yield Path(str(value["image"]))
        for child in value.values():
            yield from iter_message_image_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_message_image_paths(child)


def copy_referenced_prompt_images(runtime_copy: Path, output_dir: Path) -> int:
    image_dir = output_dir / "02_runtime_prompt_images"
    copied = 0
    seen: Set[str] = set()
    for prompt_packet in sorted(runtime_copy.rglob("01_prompt_packet.json")):
        try:
            packet = read_json(prompt_packet)
        except Exception:
            continue
        step_name = prompt_packet.parent.name
        for image_path in iter_message_image_paths(packet.get("messages_preview")):
            key = str(image_path)
            if key in seen:
                continue
            seen.add(key)
            if image_path.exists():
                target = image_dir / step_name / image_path.name
                if copy_file(image_path, target):
                    copied += 1
    return copied


def collect_task_images(seed_dir: Path, episode_id: int, output_dir: Path) -> Dict[str, bool]:
    copied: Dict[str, bool] = {}
    rgb_root = seed_dir / "rgb_dir"
    image_out = output_dir / "03_runtime_saved_images"
    for rel in [
        Path("front") / str(episode_id),
        Path("query_views") / str(episode_id),
    ]:
        copied[str(rel)] = copy_tree(rgb_root / rel, image_out / rel)
    return copied


def collect_test_data(seed_dir: Path, output_dir: Path) -> bool:
    return copy_file(seed_dir / "test_data.csv", output_dir / "04_metrics" / "test_data.csv")


def safe_video_name(log_root: Path, video: Path) -> str:
    try:
        parts = video.relative_to(log_root).parts
    except ValueError:
        parts = video.parts[-8:]
    return "__".join(part.replace("/", "_").replace(" ", "_") for part in parts)


def video_belongs_to_task(video: Path, task: str) -> bool:
    if task in video.stem:
        return True
    for part in video.parts:
        if part == task or part.startswith(f"{task}_"):
            return True
    return False


def collect_videos(log_root: Path, output_root: Path, task: str, output_dir: Path) -> int:
    per_demo_dir = output_dir / "05_runtime_videos"
    all_video_dir = output_root / "runtime_videos_all"
    copied = 0
    for video in sorted(log_root.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue
        all_video_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_video_name(log_root, video)
        copy_file(video, all_video_dir / safe_name)
        if video_belongs_to_task(video, task):
            if copy_file(video, per_demo_dir / safe_name):
                copied += 1
    return copied


def task_seed_dir_from_manifest(manifest_path: Path) -> Path:
    # .../<task>/seed0/phase_memory_runtime/<task>/episode0/00_phase_memory_manifest.json
    return manifest_path.parent.parent.parent.parent


def write_demo_readme(output_dir: Path, manifest: Dict[str, Any], copied_prompt_images: int) -> None:
    task = manifest.get("task")
    episode = manifest.get("episode_id")
    lines = [
        f"# Phase-memory review packet: {task} episode {episode}",
        "",
        "Open these folders in order:",
        "",
        "1. `00_phase_packet_inputs/` - static inputs produced before simulation: query RGB, contact overlays, extracted geometry/target/contact descriptors, phase interpreter prompt/output, normalized anchored phases, compiled open-loop actions, and verifier output.",
        "2. `01_runtime_phase_memory/` - live controller outputs from the simulation. Each `step_*` folder contains `01_prompt_packet.json`, `02_vlm_output.json`, `03_controller_decision.json`, and usually `04_short_term_memory_after.json`.",
        "3. `02_runtime_prompt_images/` - copies of the exact RGB image files referenced by the runtime QwenVL prompt packets.",
        "4. `03_runtime_saved_images/` - front/top images saved by the agent for the episode.",
        "5. `04_metrics/test_data.csv` - task score file when RLBench wrote one.",
        "6. `05_runtime_videos/` - videos for this task when the recorder emitted task-named mp4s.",
        "",
        f"Copied runtime prompt images: {copied_prompt_images}",
        "",
        "The controller is phase-by-phase: retrieved demos are long-term memory, the current phase is the short-term objective, and every QwenVL call outputs one JSON decision/action.",
    ]
    write_text(output_dir / "README.md", "\n".join(lines))


def collect_one(index: int, manifest_path: Path, log_root: Path, output_root: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    task = str(manifest.get("task") or manifest_path.parent.parent.name)
    episode_id = int(manifest.get("episode_id") or 0)
    output_dir = output_root / f"{index:02d}_{task}_episode{episode_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_phase_packet = Path(str(manifest.get("source_phase_packet") or ""))
    copied_phase_packet = copy_tree(source_phase_packet, output_dir / "00_phase_packet_inputs")

    runtime_episode_dir = manifest_path.parent
    runtime_copy = output_dir / "01_runtime_phase_memory"
    copied_runtime = copy_tree(runtime_episode_dir, runtime_copy)
    copied_prompt_images = copy_referenced_prompt_images(runtime_copy, output_dir) if copied_runtime else 0

    seed_dir = task_seed_dir_from_manifest(manifest_path)
    copied_image_dirs = collect_task_images(seed_dir, episode_id, output_dir)
    copied_metrics = collect_test_data(seed_dir, output_dir)
    video_count = collect_videos(log_root, output_root, task, output_dir)

    step_count = len(list(runtime_copy.glob("step_*"))) if runtime_copy.exists() else 0
    row = {
        "task": task,
        "episode_id": episode_id,
        "step_count": step_count,
        "phase_count": manifest.get("phase_count"),
        "source_phase_packet": str(source_phase_packet),
        "copied_phase_packet": copied_phase_packet,
        "copied_runtime": copied_runtime,
        "copied_prompt_images": copied_prompt_images,
        "copied_metrics": copied_metrics,
        "video_count": video_count,
        "output_dir": str(output_dir),
    }
    row.update({f"copied_images_{key}": value for key, value in copied_image_dirs.items()})
    write_json(output_dir / "00_review_manifest.json", {**manifest, "review_collection": row})
    write_demo_readme(output_dir, manifest, copied_prompt_images)
    return row


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_task_filter(value: str) -> Optional[Set[str]]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    return set(tasks) if tasks else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    log_root = Path(args.log_root).resolve()
    run_name = args.name or datetime.utcnow().strftime("phase_memory_runtime_review_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    task_filter = parse_task_filter(args.tasks)
    manifests = find_runtime_manifests(log_root, task_filter)
    if args.limit > 0:
        manifests = manifests[: args.limit]

    rows = []
    for index, manifest_path in enumerate(manifests, start=1):
        print(f"Collecting {manifest_path}", flush=True)
        rows.append(collect_one(index, manifest_path, log_root, output_root))

    write_summary_csv(output_root / "phase_memory_runtime_review_summary.csv", rows)
    write_json(
        output_root / "phase_memory_runtime_review_summary.json",
        {
            "log_root": str(log_root),
            "output_root": str(output_root),
            "task_filter": sorted(task_filter) if task_filter else None,
            "demo_count": len(rows),
            "rows": rows,
        },
    )
    write_text(
        output_root / "README.md",
        "\n".join(
            [
                "# Phase-memory runtime review",
                "",
                "This folder bundles the static phase/contact/retrieval inputs and the live phase-memory controller inputs/outputs from the simulation.",
                "",
                "Each demo folder contains:",
                "- `00_phase_packet_inputs/` for phase interpreter, contact overlay, geometry descriptors, retrieval memory inputs, compiler, and verifier files.",
                "- `01_runtime_phase_memory/` for every runtime QwenVL prompt, raw response, parsed response, controller guard decision, and short-term memory state.",
                "- `02_runtime_prompt_images/` and `03_runtime_saved_images/` for the actual RGB images used by the prompt.",
                "- `04_metrics/` and `05_runtime_videos/` when those files are present.",
                "",
                f"Demo folders collected: {len(rows)}",
            ]
        ),
    )
    print(f"Wrote phase-memory runtime review to {output_root}", flush=True)


if __name__ == "__main__":
    main()
