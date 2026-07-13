#!/usr/bin/env python3
"""Run the query-only phase interpreter and write inspectable review packets."""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import form_icl_demonstrations_crosstask_ranking as xicm
import generate_rerank_review_trace as review_trace
import phase_interpreter


def write_text(path, text):
    with open(path, "w") as handle:
        handle.write(str(text).rstrip() + "\n")


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(review_trace.json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def messages_have_images(messages):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(isinstance(item, dict) and item.get("type") == "image" for item in content):
                return True
    return False


def generate_vllm_text(llm, processor, messages, max_tokens):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    llm_inputs = {"prompt": prompt}
    if messages_have_images(messages):
        image_inputs, video_inputs = process_vision_info(messages)
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs
        if multi_modal_data:
            llm_inputs["multi_modal_data"] = multi_modal_data

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        repetition_penalty=1.02,
        stop_token_ids=[],
    )
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def load_vlm(model_path, gpu_memory_utilization, max_model_len):
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": max(1, torch.cuda.device_count()),
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 2},
    }
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len
    if os.environ.get("XICM_VLLM_ENFORCE_EAGER", "0") == "1":
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return llm, processor


def task_episode_path(task_name, episode_id):
    return os.path.join(
        xicm.unseen_path,
        task_name,
        "all_variations",
        "episodes",
        f"episode{episode_id}",
    )


def sample_tasks(count, seed, episode_id, requested_tasks=None):
    if requested_tasks:
        tasks = [task.strip() for task in requested_tasks.split(",") if task.strip()]
    else:
        tasks = [
            name
            for name in sorted(os.listdir(xicm.unseen_path))
            if os.path.isdir(task_episode_path(name, episode_id))
            and name in xicm.unseen_task_name_to_handler
        ]
        tasks = random.Random(seed).sample(tasks, min(count, len(tasks)))
    return tasks[:count]


def build_query_packet(task_name, episode_id, output_dir):
    episode_path = task_episode_path(task_name, episode_id)
    instruction = review_trace.read_instruction(episode_path)
    handler = xicm.create_task_handler(task_name)

    mask_dict = xicm._get_mask_dict(episode_path, 0)
    mask_names_by_camera = xicm._get_mask_id_to_name_dict(episode_path, 0)
    mask_id_to_sim_name = review_trace.merge_mask_names(mask_names_by_camera)
    point_cloud_dict = xicm._get_point_cloud_dict(episode_path, 0)
    mask_id_to_real_name = {
        mask_id: handler.sim_name_to_real_name[name]
        for mask_id, name in mask_id_to_sim_name.items()
        if name in handler.sim_name_to_real_name
    }
    query_observation = xicm.form_obs(
        mask_dict,
        mask_id_to_real_name,
        point_cloud_dict,
        taskname=instruction,
        cross_task_eval=1,
    )
    query_geometry, query_contact = xicm._query_descriptors(
        task_name,
        instruction,
        mask_dict=mask_dict,
        mask_id_to_sim_name=mask_id_to_sim_name,
        point_cloud_dict=point_cloud_dict,
    )
    query_geometry = dict(query_geometry)
    query_geometry["task_key"] = task_name
    query_profile, goal_state = review_trace.query_goal_state(task_name, query_geometry, query_contact)

    query_rgb_sources = {}
    query_rgb_paths = {}
    for camera in xicm.CAMERAS:
        source_path = os.path.join(episode_path, f"{camera}_rgb", "0.png")
        if not os.path.exists(source_path):
            continue
        local_path = os.path.join(output_dir, f"query_{camera}_rgb_0.png")
        shutil.copy2(source_path, local_path)
        query_rgb_sources[camera] = source_path
        query_rgb_paths[camera] = local_path
    local_front = query_rgb_paths.get("front")
    local_overhead = query_rgb_paths.get("overhead")
    overlay_paths = review_trace.draw_contact_overlays(output_dir, query_contact)

    user_prompt = phase_interpreter.build_phase_interpreter_user_prompt(
        task_key=task_name,
        instruction=instruction,
        current_observation=query_observation,
        query_geometry=query_geometry,
        goal_state=goal_state,
        contact_hints=query_contact,
        interaction_profile=query_profile,
    )
    messages = phase_interpreter.build_phase_messages(user_prompt, local_front, local_overhead)

    extracted = {
        "task": task_name,
        "episode_id": episode_id,
        "instruction": instruction,
        "current_observation": query_observation,
        "geometry_g_j": query_geometry,
        "goal_state_h_j": goal_state,
        "contact_hints_c_j": query_contact,
        "interaction_profile_p_j": query_profile,
        "mask_id_to_sim_name": mask_id_to_sim_name,
        "mask_id_to_real_name": mask_id_to_real_name,
        "source_images": query_rgb_sources,
        "local_images": query_rgb_paths,
        "contact_overlay_paths": overlay_paths,
    }
    return extracted, user_prompt, messages


def write_query_files(output_dir, extracted, user_prompt, messages):
    write_json(os.path.join(output_dir, "01_extracted_query.json"), extracted)
    write_text(
        os.path.join(output_dir, "01_extracted_query.md"),
        "\n\n".join(
            [
                f"# Query extraction: {extracted['task']} episode{extracted['episode_id']}",
                f"Instruction: {extracted['instruction']}",
                "## Current observation",
                extracted["current_observation"],
                "## Query descriptors",
                xicm._format_compact_query_descriptor(
                    extracted["geometry_g_j"],
                    extracted["goal_state_h_j"],
                ),
                "## Query contact hints",
                xicm._format_compact_query_contact_hints(extracted["contact_hints_c_j"]),
                "## Interaction profile",
                xicm._format_profile_block(
                    "Precise interaction signature p_j",
                    extracted["interaction_profile_p_j"],
                ),
            ]
        ),
    )
    write_text(os.path.join(output_dir, "02_phase_interpreter_prompt.txt"), user_prompt)
    write_json(
        os.path.join(output_dir, "02_phase_interpreter_messages_preview.json"),
        {
            "system_prompt": phase_interpreter.PHASE_INTERPRETER_SYSTEM_PROMPT,
            "query_images": extracted["local_images"],
            "messages": messages,
        },
    )


def build_readme(output_dir, extracted, parsed, parse_error):
    if parsed:
        phase_lines = [
            f"- {index + 1}. {phase.get('phase', 'unknown')}: "
            f"{phase.get('purpose', '')} "
            f"(target_role={phase.get('target_role', 'unknown')}, gripper={phase.get('gripper', 'unknown')})"
            for index, phase in enumerate(parsed.get("phase_plan") or [])
        ]
    else:
        phase_lines = ["- Parse failed; inspect `03_phase_interpreter_raw_output.txt`."]
    lines = [
        f"# {extracted['task']} episode{extracted['episode_id']}",
        "",
        f"Instruction: {extracted['instruction']}",
        "",
        "Files:",
        "- `query_<camera>_rgb_0.png`: current query images for every available camera.",
        "- `contact_points_overlay_<camera>.png` and `contact_points_overlay_all_views.png`: role-labeled c_j points overlaid on RGB.",
        "- `01_extracted_query.md/json`: query-only g_j, h_j, c_j, p_j inputs.",
        "- `02_phase_interpreter_prompt.txt`: exact prompt sent to the phase interpreter.",
        "- `03_phase_interpreter_raw_output.txt`: raw VLM output.",
        "- `04_phase_interpreter_parsed.json`: parsed JSON phase plan, when valid.",
        "",
        "Phase summary:",
        *(phase_lines or ["- none"]),
    ]
    if parse_error:
        lines.extend(["", f"Parse error: {parse_error}"])
    write_text(os.path.join(output_dir, "README.md"), "\n".join(lines))


def run_one(task_name, episode_id, output_dir, llm, processor, max_tokens, dry_run):
    os.makedirs(output_dir, exist_ok=True)
    extracted, user_prompt, messages = build_query_packet(task_name, episode_id, output_dir)
    write_query_files(output_dir, extracted, user_prompt, messages)

    raw_output = ""
    parsed = None
    parse_error = ""
    if dry_run:
        raw_output = '{"task_family": "dry_run", "phase_plan": []}'
        parsed = phase_interpreter.parse_phase_interpreter_output(raw_output)
    else:
        raw_output = generate_vllm_text(llm, processor, messages, max_tokens=max_tokens)
        try:
            parsed = phase_interpreter.parse_phase_interpreter_output(raw_output)
        except Exception as exc:
            parse_error = str(exc)

    write_text(os.path.join(output_dir, "03_phase_interpreter_raw_output.txt"), raw_output)
    write_json(
        os.path.join(output_dir, "04_phase_interpreter_parsed.json"),
        parsed
        if parsed is not None
        else {
            "parse_error": parse_error,
            "raw_output_path": os.path.join(output_dir, "03_phase_interpreter_raw_output.txt"),
        },
    )
    build_readme(output_dir, extracted, parsed, parse_error)
    return {
        "task": task_name,
        "episode_id": episode_id,
        "instruction": extracted["instruction"],
        "task_family": parsed.get("task_family") if parsed else "parse_failed",
        "phase_count": len(parsed.get("phase_plan") or []) if parsed else 0,
        "parse_error": parse_error,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/data/yf23/projects/ICRA27-ROBOT/review")
    parser.add_argument("--name", default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--tasks", default="")
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

    run_name = args.name or datetime.now().strftime("phase_interpreter_query_only_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tasks = sample_tasks(args.count, args.seed, args.episode_id, args.tasks)
    write_json(
        os.path.join(run_dir, "00_run_config.json"),
        {
            "run_name": run_name,
            "count": args.count,
            "seed": args.seed,
            "episode_id": args.episode_id,
            "tasks": tasks,
            "model_path": args.model_path,
            "dry_run": args.dry_run,
            "note": "Query-only phase interpreter. No retrieved seen demos are provided.",
        },
    )

    llm = None
    processor = None
    if not args.dry_run:
        print(f"Loading VLM: {args.model_path}", flush=True)
        llm, processor = load_vlm(
            args.model_path,
            args.gpu_memory_utilization,
            args.max_model_len,
        )

    rows = []
    for index, task_name in enumerate(tasks, start=1):
        task_dir = os.path.join(run_dir, f"{index:02d}_{task_name}_episode{args.episode_id}")
        print(f"[{index}/{len(tasks)}] phase interpreting {task_name} episode{args.episode_id}", flush=True)
        rows.append(
            run_one(
                task_name,
                args.episode_id,
                task_dir,
                llm,
                processor,
                args.max_tokens,
                args.dry_run,
            )
        )

    write_csv(
        os.path.join(run_dir, "phase_interpreter_summary.csv"),
        rows,
        ["task", "episode_id", "instruction", "task_family", "phase_count", "parse_error", "output_dir"],
    )
    write_json(os.path.join(run_dir, "phase_interpreter_summary.json"), rows)
    write_text(
        os.path.join(run_dir, "README.md"),
        "\n".join(
            [
                f"# {run_name}",
                "",
                "Query-only phase interpreter review.",
                "",
                "Each task folder contains extracted query inputs, the exact phase prompt, raw VLM output, parsed JSON, RGB images, and contact overlays.",
                "",
                "Summary:",
                *[
                    f"- {row['task']}: family={row['task_family']}, phases={row['phase_count']}, parse_error={row['parse_error'] or 'none'}"
                    for row in rows
                ],
            ]
        ),
    )
    print(f"Wrote phase interpreter review to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
