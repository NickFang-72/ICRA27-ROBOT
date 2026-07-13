#!/usr/bin/env python3
"""Generate phase-interpreter packets for a multi-task benchmark run.

This is the benchmark-oriented version of generate_phase_interpreter_review.py:
it loads the VLM once, then writes query-only phase packets for every requested
task/episode pair.
"""

import argparse
import os
from datetime import datetime

import generate_phase_interpreter_review as phase_review


DEFAULT_UNSEEN_TASKS = [
    "put_toilet_roll_on_stand",
    "put_knife_on_chopping_board",
    "close_fridge",
    "close_microwave",
    "close_laptop_lid",
    "phone_on_base",
    "toilet_seat_down",
    "lamp_off",
    "lamp_on",
    "put_books_on_bookshelf",
    "put_umbrella_in_umbrella_stand",
    "open_grill",
    "put_rubbish_in_bin",
    "take_usb_out_of_computer",
    "take_lid_off_saucepan",
    "take_plate_off_colored_dish_rack",
    "basketball_in_hoop",
    "scoop_with_spatula",
    "straighten_rope",
    "turn_oven_on",
    "beat_the_buzz",
    "water_plants",
    "unplug_charger",
]


def parse_csv_list(text):
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_episode_ids(text):
    values = []
    for item in parse_csv_list(text):
        values.append(int(item))
    if not values:
        raise ValueError("--episode-ids must include at least one integer episode id")
    return values


def resolve_tasks(tasks_arg, count, episode_ids):
    if tasks_arg:
        tasks = parse_csv_list(tasks_arg)
    else:
        tasks = list(DEFAULT_UNSEEN_TASKS)

    if count and count > 0:
        tasks = tasks[:count]

    missing = []
    for task_name in tasks:
        if task_name not in phase_review.xicm.unseen_task_name_to_handler:
            missing.append((task_name, "handler"))
            continue
        for episode_id in episode_ids:
            episode_path = phase_review.task_episode_path(task_name, episode_id)
            if not os.path.isdir(episode_path):
                missing.append((task_name, f"episode{episode_id}"))
    if missing:
        details = ", ".join(f"{task}:{what}" for task, what in missing[:20])
        suffix = "" if len(missing) <= 20 else f" ... +{len(missing) - 20} more"
        raise FileNotFoundError(f"Missing task/episode inputs: {details}{suffix}")

    return tasks


def update_progress(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    phase_review.write_json(path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/data/yf23/projects/ICRA27-ROBOT/review")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--count", type=int, default=0, help="0 means all requested/default tasks.")
    parser.add_argument("--episode-ids", default="0,1,2,3,4")
    parser.add_argument("--progress-json", default="")
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "XICM_QWEN25_VL_7B_PATH",
            "/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct",
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episode_ids = parse_episode_ids(args.episode_ids)
    tasks = resolve_tasks(args.tasks, args.count, episode_ids)

    run_name = args.name or datetime.now().strftime("phase_anchor_benchmark_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    total = len(tasks) * len(episode_ids)
    phase_review.write_json(
        os.path.join(run_dir, "00_run_config.json"),
        {
            "run_name": run_name,
            "tasks": tasks,
            "episode_ids": episode_ids,
            "total_packets": total,
            "model_path": args.model_path,
            "dry_run": args.dry_run,
            "note": "Query-only phase benchmark packets. No retrieved seen demos are provided.",
        },
    )

    update_progress(
        args.progress_json,
        {
            "stage": "phase_interpreter_starting",
            "run_dir": run_dir,
            "completed_packets": 0,
            "total_packets": total,
            "tasks": tasks,
            "episode_ids": episode_ids,
        },
    )

    llm = None
    processor = None
    if not args.dry_run:
        print(f"Loading VLM: {args.model_path}", flush=True)
        llm, processor = phase_review.load_vlm(
            args.model_path,
            args.gpu_memory_utilization,
            args.max_model_len,
        )

    rows = []
    packet_index = 0
    for task_name in tasks:
        for episode_id in episode_ids:
            packet_index += 1
            task_dir = os.path.join(run_dir, f"{packet_index:03d}_{task_name}_episode{episode_id}")
            print(
                f"[{packet_index}/{total}] phase interpreting {task_name} episode{episode_id}",
                flush=True,
            )
            row = phase_review.run_one(
                task_name,
                episode_id,
                task_dir,
                llm,
                processor,
                args.max_tokens,
                args.dry_run,
            )
            rows.append(row)
            update_progress(
                args.progress_json,
                {
                    "stage": "phase_interpreter_running",
                    "run_dir": run_dir,
                    "completed_packets": packet_index,
                    "total_packets": total,
                    "current_task": task_name,
                    "current_episode_id": episode_id,
                    "last_parse_error": row.get("parse_error") or "",
                },
            )

    phase_review.write_csv(
        os.path.join(run_dir, "phase_interpreter_summary.csv"),
        rows,
        ["task", "episode_id", "instruction", "task_family", "phase_count", "parse_error", "output_dir"],
    )
    phase_review.write_json(os.path.join(run_dir, "phase_interpreter_summary.json"), rows)
    phase_review.write_text(
        os.path.join(run_dir, "README.md"),
        "\n".join(
            [
                f"# {run_name}",
                "",
                "Query-only phase benchmark review.",
                "",
                "Each task/episode folder contains extracted query inputs, the exact phase prompt, raw VLM output, parsed JSON, RGB images, and contact overlays.",
                "",
                f"Tasks: {len(tasks)}",
                f"Episodes per task: {len(episode_ids)}",
                f"Total packets: {total}",
            ]
        ),
    )
    update_progress(
        args.progress_json,
        {
            "stage": "phase_interpreter_complete",
            "run_dir": run_dir,
            "completed_packets": total,
            "total_packets": total,
            "summary_csv": os.path.join(run_dir, "phase_interpreter_summary.csv"),
        },
    )
    print(f"Wrote phase benchmark review to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
