#!/usr/bin/env python3
"""Collect quick QwenVL 5-episode component ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


UNSEEN_TASKS = [
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

LEVEL_1_TASKS = set(UNSEEN_TASKS[:13])
LEVEL_2_TASKS = set(UNSEEN_TASKS[13:])
FINAL_SCORE_RE = re.compile(r"Finished\s+(.+?)\s+\|\s+Final Score:\s*([-+]?\d+(?:\.\d+)?)")
MODEL_NAME = "Qwen2.5.VL.7B.instruct"

METHODS = {
    "qwenvl_geo_target_pose_5eps_seed0": {
        "label": "QwenVL + geometry retrieval + target-pose prompt (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": False,
    },
    "qwenvl_contact_points_5eps_seed0": {
        "label": "QwenVL + contact points prompt only (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.aff",
        "weights": {"alpha_dynamic": 1.0, "beta_geometry": 0.0, "gamma_contact": 0.0},
        "uses_geometry_retrieval": False,
        "uses_target_pose_prompt": False,
        "uses_contact_prompt": True,
    },
    "qwenvl_everything_5eps_seed0": {
        "label": "QwenVL + geometry/target-pose + contact points (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo.aff",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": True,
    },
}

PLAN_METHODS = {
    "qwenvl_plan_geo_target_pose_5eps_seed0": {
        "label": "QwenVL + plan-guided geometry/target-pose retrieval (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo_plan",
        "weights": {
            "alpha_dynamic": 0.72,
            "beta_geometry": 0.03,
            "gamma_contact": 0.0,
            "delta_profile": 0.25,
            "penalty_weight": 0.55,
            "plan_weight": 0.45,
        },
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": False,
        "uses_plan_guided_retrieval": True,
        "target_pose_in_retrieval": True,
    },
    "qwenvl_plan_geo_target_pose_contact_5eps_seed0": {
        "label": "QwenVL + plan-guided geometry/target-pose retrieval + contact points (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo_plan.aff",
        "weights": {
            "alpha_dynamic": 0.72,
            "beta_geometry": 0.03,
            "gamma_contact": 0.0,
            "delta_profile": 0.25,
            "penalty_weight": 0.55,
            "plan_weight": 0.45,
        },
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": True,
        "uses_plan_guided_retrieval": True,
        "target_pose_in_retrieval": True,
    },
}

CLOSED_LOOP_METHODS = {
    "qwenvl_closed_loop_plan_geo_target_pose_5eps_seed0": {
        "label": "QwenVL + closed-loop plan-guided geometry/target-pose retrieval (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo_plan.closed_loop",
        "weights": {
            "alpha_dynamic": 0.72,
            "beta_geometry": 0.03,
            "gamma_contact": 0.0,
            "delta_profile": 0.25,
            "penalty_weight": 0.55,
            "plan_weight": 0.45,
        },
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": False,
        "uses_plan_guided_retrieval": True,
        "target_pose_in_retrieval": True,
        "uses_closed_loop": True,
        "closed_loop_max_replans": 4,
    },
    "qwenvl_closed_loop_plan_geo_target_pose_contact_5eps_seed0": {
        "label": "QwenVL + closed-loop plan-guided geometry/target-pose retrieval + contact points (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo_plan.aff.closed_loop",
        "weights": {
            "alpha_dynamic": 0.72,
            "beta_geometry": 0.03,
            "gamma_contact": 0.0,
            "delta_profile": 0.25,
            "penalty_weight": 0.55,
            "plan_weight": 0.45,
        },
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": True,
        "uses_plan_guided_retrieval": True,
        "target_pose_in_retrieval": True,
        "uses_closed_loop": True,
        "closed_loop_max_replans": 4,
    },
}

CLOSED_LOOP_NO_PLAN_METHODS = {
    "qwenvl_closed_loop_geo_target_pose_5eps_seed0": {
        "label": "QwenVL + closed-loop geometry/target-pose retrieval (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo.closed_loop",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": False,
        "uses_plan_guided_retrieval": False,
        "target_pose_in_retrieval": False,
        "uses_closed_loop": True,
        "closed_loop_max_replans": 4,
    },
    "qwenvl_closed_loop_geo_target_pose_contact_5eps_seed0": {
        "label": "QwenVL + closed-loop geometry/target-pose retrieval + contact points (5 eps, seed 0)",
        "ranking_method": "lang_vis.out.geo.aff.closed_loop",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": True,
        "uses_plan_guided_retrieval": False,
        "target_pose_in_retrieval": False,
        "uses_closed_loop": True,
        "closed_loop_max_replans": 4,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output = root / "test_files/geometry_affordance_probe/ablation_results/qwenvl_component_5eps_seed0_2026-06-27"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=default_output / "cair_logs")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--demo-num-per-icl", type=int, default=18)
    parser.add_argument(
        "--method-set",
        choices=[
            "default",
            "plan",
            "closed_loop",
            "closed_loop_no_plan",
            "default,plan",
            "plan,closed_loop",
        ],
        default="default",
        help="Which QwenVL method definitions to collect.",
    )
    parser.add_argument(
        "--baseline-task-means-csv",
        type=Path,
        default=root
        / "test_files/geometry_affordance_probe/ablation_results/qwen_vs_qwenvl_front_top_2026-06-26/qwen_vs_qwenvl_front_top_task_means.csv",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def selected_methods(method_set: str) -> dict[str, dict]:
    methods = {}
    if method_set in {"default", "default,plan"}:
        methods.update(METHODS)
    if method_set in {"plan", "default,plan", "plan,closed_loop"}:
        methods.update(PLAN_METHODS)
    if method_set in {"closed_loop", "plan,closed_loop"}:
        methods.update(CLOSED_LOOP_METHODS)
    if method_set == "closed_loop_no_plan":
        methods.update(CLOSED_LOOP_NO_PLAN_METHODS)
    return methods


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}"


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def add_summary(scores: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    output = dict(scores)
    output["Level 1 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_1_TASKS)
    output["Level 2 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_2_TASKS)
    output["Average"] = mean_optional(output[task] for task in UNSEEN_TASKS)
    return output


def load_baseline_rows(path: Path) -> list[tuple[str, str, dict[str, Optional[float]]]]:
    if not path.exists():
        return []
    wanted = [
        (
            "qwen_text_baseline",
            "Qwen2.5 7B text-only baseline (5 eps, seed 0)",
            "qwen_text_baseline_5eps_seed0",
        ),
        (
            "qwenvl_front_top_baseline",
            "QwenVL front+top baseline (5 eps, seed 0)",
            "qwenvl_front_top_baseline_5eps_seed0",
        ),
    ]
    by_source = {
        source_run: {task: None for task in UNSEEN_TASKS}
        for source_run, _label, _run in wanted
    }
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("run_id", "")
            if run_id not in by_source:
                continue
            task = row.get("task", "")
            if task in by_source[run_id] and row.get("mean_final_score", "") != "":
                by_source[run_id][task] = float(row["mean_final_score"])
    rows = []
    for source_run, label, run in wanted:
        scores = by_source[source_run]
        if any(value is not None for value in scores.values()):
            rows.append((label, run, scores))
    return rows


def parse_test_data_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    matches = list(FINAL_SCORE_RE.finditer(path.read_text(errors="replace")))
    return float(matches[-1].group(2)) if matches else None


def load_seed_scores(logs_root: Path, method: str, seed: int) -> dict[str, Optional[float]]:
    scores: dict[str, Optional[float]] = {}
    for task in UNSEEN_TASKS:
        scores[task] = parse_test_data_score(logs_root / method / task / f"seed{seed}" / "test_data.csv")
    return scores


def method_name(ranking_method: str, demo_num_per_icl: int) -> str:
    return f"XICM_Cross.ZS_Ranking.{ranking_method}_{MODEL_NAME}_icl.{demo_num_per_icl}_test"


def average_complete_seed_scores(
    seed_scores: dict[int, dict[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    averaged: dict[str, Optional[float]] = {}
    for task in UNSEEN_TASKS:
        values = [scores.get(task) for scores in seed_scores.values()]
        averaged[task] = mean(values) if values and all(value is not None for value in values) else None
    return averaged


def load_method_results(
    logs_root: Path,
    seeds: list[int],
    demo_num_per_icl: int,
    methods: dict[str, dict],
) -> tuple[list[tuple[str, str, dict[str, Optional[float]]]], dict[str, dict[int, dict[str, Optional[float]]]]]:
    rows = []
    seed_scores_by_run = {}
    for run, spec in methods.items():
        method = method_name(spec["ranking_method"], demo_num_per_icl)
        seed_scores = {seed: load_seed_scores(logs_root, method, seed) for seed in seeds}
        rows.append((spec["label"], run, average_complete_seed_scores(seed_scores)))
        seed_scores_by_run[run] = seed_scores
    return rows, seed_scores_by_run


def write_paper_style(
    output_dir: Path,
    rows: list[tuple[str, str, dict[str, Optional[float]]]],
) -> Path:
    path = output_dir / "qwenvl_5ep_component_ablation_paper_style_scores.csv"
    columns = ["method", "run", *UNSEEN_TASKS, "Level 1 Avg", "Level 2 Avg", "Average"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for label, run, scores in rows:
            expanded = add_summary(scores)
            row = {"method": label, "run": run}
            row.update({task: fmt(expanded.get(task)) for task in columns[2:]})
            writer.writerow(row)
    return path


def write_task_scores(
    output_dir: Path,
    baseline_rows: list[tuple[str, str, dict[str, Optional[float]]]],
    method_rows: list[tuple[str, str, dict[str, Optional[float]]]],
    seed_scores_by_run: dict[str, dict[int, dict[str, Optional[float]]]],
    seeds: list[int],
    methods: dict[str, dict],
) -> Path:
    path = output_dir / "qwenvl_5ep_component_ablation_task_scores.csv"
    baseline_by_run = {run: scores for _label, run, scores in baseline_rows}
    qwen_scores = baseline_by_run.get("qwen_text_baseline_5eps_seed0", {task: None for task in UNSEEN_TASKS})
    qwenvl_scores = baseline_by_run.get("qwenvl_front_top_baseline_5eps_seed0", {task: None for task in UNSEEN_TASKS})
    method_by_run = {run: scores for _label, run, scores in method_rows}
    columns = ["task", "qwen_text_baseline", "qwenvl_front_top_baseline"]
    for run in methods:
        columns.extend([f"{run}_seed{seed}" for seed in seeds])
        columns.extend([
            run,
            f"delta_{run}_vs_qwen_text_baseline",
            f"delta_{run}_vs_qwenvl_baseline",
        ])
    columns.append("status")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in UNSEEN_TASKS:
            qwen_base = qwen_scores.get(task)
            qwenvl_base = qwenvl_scores.get(task)
            row = {
                "task": task,
                "qwen_text_baseline": fmt(qwen_base),
                "qwenvl_front_top_baseline": fmt(qwenvl_base),
            }
            statuses = []
            for run in methods:
                value = method_by_run[run].get(task)
                for seed in seeds:
                    row[f"{run}_seed{seed}"] = fmt(seed_scores_by_run[run][seed].get(task))
                row[run] = fmt(value)
                row[f"delta_{run}_vs_qwen_text_baseline"] = fmt(value - qwen_base) if value is not None and qwen_base is not None else ""
                row[f"delta_{run}_vs_qwenvl_baseline"] = fmt(value - qwenvl_base) if value is not None and qwenvl_base is not None else ""
                if value is not None and qwenvl_base is not None:
                    if value > qwenvl_base:
                        statuses.append(f"{run}:better")
                    elif value < qwenvl_base:
                        statuses.append(f"{run}:worse")
                    else:
                        statuses.append(f"{run}:same")
                elif value is None:
                    statuses.append(f"{run}:pending")
            row["status"] = ";".join(statuses)
            writer.writerow(row)
    return path


def write_metadata(
    output_dir: Path,
    seeds: list[int],
    episodes: int,
    demo_num_per_icl: int,
    rows: list[tuple[str, str, dict[str, Optional[float]]]],
    missing: list[str],
    methods: dict[str, dict],
    method_set: str,
) -> Path:
    path = output_dir / "qwenvl_5ep_component_ablation_metadata.json"
    payload = {
        "condition": "qwenvl_component_ablation_5eps_seed0" if method_set == "default" else f"qwenvl_{method_set.replace(',', '_')}_5eps_seed0",
        "seeds": seeds,
        "episodes": episodes,
        "demo_num_per_icl": demo_num_per_icl,
        "qwen_vl_query_images": ["front_rgb_initial", "overhead_rgb_initial"],
        "contact_points_in_retrieval": False,
        "target_pose_in_retrieval": any(spec.get("target_pose_in_retrieval") for spec in methods.values()),
        "methods": methods,
        "row_averages": {
            run: add_summary(scores).get("Average")
            for _label, run, scores in rows
        },
        "missing_strict_finals": missing,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    methods = selected_methods(args.method_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_baseline_rows(args.baseline_task_means_csv)
    method_rows, seed_scores_by_run = load_method_results(args.logs_root, seeds, args.demo_num_per_icl, methods)
    rows = baseline_rows + method_rows
    missing = []
    for run, seed_scores in seed_scores_by_run.items():
        for seed, scores in seed_scores.items():
            for task, score in scores.items():
                if score is None:
                    missing.append(f"{run}:seed{seed}:{task}")
    write_paper_style(args.output_dir, rows)
    write_task_scores(args.output_dir, baseline_rows, method_rows, seed_scores_by_run, seeds, methods)
    write_metadata(args.output_dir, seeds, args.episodes, args.demo_num_per_icl, rows, missing, methods, args.method_set)
    if args.require_complete and missing:
        raise SystemExit(f"Missing {len(missing)} strict final scores")
    for _label, run, scores in rows:
        avg = add_summary(scores).get("Average")
        print(f"{run}: Average={fmt(avg) if avg is not None else '(pending)'}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
