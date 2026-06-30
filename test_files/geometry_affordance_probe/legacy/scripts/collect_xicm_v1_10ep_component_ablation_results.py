#!/usr/bin/env python3
"""Collect 10-episode X-ICM v1 component ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Optional


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

BASELINE_COMPARISON_CSV = (
    "test_files/xicm_baseline_results/vanilla_rerun_3seed_2026-06-22/"
    "vanilla_baseline_rerun_3seed_paper_style_scores.csv"
)

METHODS = {
    "v1_geo_target_pose_10eps_3seed": {
        "label": "X-ICM 7B v1 + geometry retrieval + target-pose prompt (10 eps x 3 seeds)",
        "method": "XICM_Cross.ZS_Ranking.lang_vis.out.geo_Qwen2.5.7B.instruct_icl.18_test",
        "ranking_method": "lang_vis.out.geo",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": False,
    },
    "v1_contact_points_10eps_3seed": {
        "label": "X-ICM 7B v1 + contact points prompt only (10 eps x 3 seeds)",
        "method": "XICM_Cross.ZS_Ranking.lang_vis.out.aff_Qwen2.5.7B.instruct_icl.18_test",
        "ranking_method": "lang_vis.out.aff",
        "weights": {"alpha_dynamic": 1.0, "beta_geometry": 0.0, "gamma_contact": 0.0},
        "uses_geometry_retrieval": False,
        "uses_target_pose_prompt": False,
        "uses_contact_prompt": True,
    },
    "v1_everything_10eps_3seed": {
        "label": "X-ICM 7B v1 + geometry/target-pose + contact points (10 eps x 3 seeds)",
        "method": "XICM_Cross.ZS_Ranking.lang_vis.out.geo.aff_Qwen2.5.7B.instruct_icl.18_test",
        "ranking_method": "lang_vis.out.geo.aff",
        "weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "uses_geometry_retrieval": True,
        "uses_target_pose_prompt": True,
        "uses_contact_prompt": True,
    },
}

FINAL_SCORE_RE = re.compile(r"Final Score:\s*([-+]?\d+(?:\.\d+)?)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output = (
        root
        / "test_files/geometry_affordance_probe/ablation_results/"
        / "v1_component_10eps_3seed_2026-06-25"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-comparison-csv",
        type=Path,
        default=root / BASELINE_COMPARISON_CSV,
        help="Paper-style CSV containing paper and clean vanilla baseline rows.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=default_output / "cair_logs",
        help="Local copy of CAIR method logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated result files.",
    )
    parser.add_argument("--seeds", default="0,50,99", help="Comma-separated seeds.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless all method/seed/task strict final scores exist.",
    )
    return parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return mean(known)


def add_summary(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    output = dict(scores)
    output["Level 1 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_1_TASKS)
    output["Level 2 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_2_TASKS)
    output["Average"] = mean_optional(output[task] for task in UNSEEN_TASKS)
    return output


def load_baseline_rows(path: Path) -> list[tuple[str, str, Dict[str, Optional[float]]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    wanted_runs = {"paper_xicm_7b", "vanilla_baseline_3seed_rerun"}
    rows: list[tuple[str, str, Dict[str, Optional[float]]]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run = row.get("run", "")
            if run not in wanted_runs:
                continue
            scores = {
                task: float(row[task]) if row.get(task, "") != "" else None
                for task in UNSEEN_TASKS
            }
            rows.append((row.get("method", run), run, scores))
    missing = wanted_runs - {run for _label, run, _scores in rows}
    if missing:
        raise ValueError(f"Missing rows in {path}: {sorted(missing)}")
    rows.sort(key=lambda item: 0 if item[1] == "paper_xicm_7b" else 1)
    return rows


def parse_test_data_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        match = FINAL_SCORE_RE.search(line)
        if match:
            return float(match.group(1))
    return None


def load_seed_scores(logs_root: Path, method: str, seed: int) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    for task in UNSEEN_TASKS:
        score_path = logs_root / method / task / f"seed{seed}" / "test_data.csv"
        scores[task] = parse_test_data_score(score_path)
    return scores


def average_complete_seed_scores(seed_scores: Dict[int, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    averaged: Dict[str, Optional[float]] = {}
    for task in UNSEEN_TASKS:
        values = [scores.get(task) for scores in seed_scores.values()]
        averaged[task] = mean(values) if values and all(value is not None for value in values) else None
    return averaged


def load_method_results(
    logs_root: Path,
    seeds: list[int],
) -> tuple[list[tuple[str, str, Dict[str, Optional[float]]]], Dict[str, Dict[int, Dict[str, Optional[float]]]]]:
    rows: list[tuple[str, str, Dict[str, Optional[float]]]] = []
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for run, spec in METHODS.items():
        seed_scores = {seed: load_seed_scores(logs_root, spec["method"], seed) for seed in seeds}
        averaged = average_complete_seed_scores(seed_scores)
        rows.append((spec["label"], run, averaged))
        seed_scores_by_run[run] = seed_scores
    return rows, seed_scores_by_run


def write_paper_style(output_dir: Path, rows: list[tuple[str, str, Dict[str, Optional[float]]]]) -> Path:
    path = output_dir / "xicm_v1_10ep_component_ablation_paper_style_scores.csv"
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


def write_detailed(
    output_dir: Path,
    baseline_rows: list[tuple[str, str, Dict[str, Optional[float]]]],
    method_rows: list[tuple[str, str, Dict[str, Optional[float]]]],
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    seeds: list[int],
) -> Path:
    path = output_dir / "xicm_v1_10ep_component_ablation_task_scores.csv"
    baseline_by_run = {run: scores for _label, run, scores in baseline_rows}
    method_by_run = {run: scores for _label, run, scores in method_rows}

    columns = [
        "task",
        "paper_xicm_7b",
        "vanilla_baseline_3seed_rerun",
    ]
    for run in METHODS:
        columns.extend([f"{run}_seed{seed}" for seed in seeds])
        columns.extend([run, f"delta_{run}_vs_vanilla"])
    columns.append("status")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in UNSEEN_TASKS:
            baseline = baseline_by_run["vanilla_baseline_3seed_rerun"].get(task)
            row = {
                "task": task,
                "paper_xicm_7b": fmt(baseline_by_run["paper_xicm_7b"].get(task)),
                "vanilla_baseline_3seed_rerun": fmt(baseline),
            }
            statuses = []
            for run in METHODS:
                value = method_by_run[run].get(task)
                for seed in seeds:
                    row[f"{run}_seed{seed}"] = fmt(seed_scores_by_run[run][seed].get(task))
                row[run] = fmt(value)
                row[f"delta_{run}_vs_vanilla"] = fmt(
                    None if value is None or baseline is None else value - baseline
                )
                statuses.append(f"{run}:{'complete' if value is not None else 'pending'}")
            row["status"] = ";".join(statuses)
            writer.writerow(row)
    return path


def completion_summary(
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    seeds: list[int],
) -> dict:
    summary = {}
    for run, seed_scores in seed_scores_by_run.items():
        per_seed = {}
        count = 0
        for seed in seeds:
            seed_count = sum(value is not None for value in seed_scores[seed].values())
            per_seed[str(seed)] = seed_count
            count += seed_count
        summary[run] = {
            "strict_final_seed_task_count": count,
            "total_seed_task_count": len(seeds) * len(UNSEEN_TASKS),
            "per_seed": per_seed,
        }
    return summary


def write_metadata(
    output_dir: Path,
    paper_style_csv: Path,
    detailed_csv: Path,
    baseline_csv: Path,
    logs_root: Path,
    seeds: list[int],
    episodes: int,
    completion: dict,
) -> Path:
    path = output_dir / "xicm_v1_10ep_component_ablation_metadata.json"
    payload = {
        "description": "X-ICM v1 component ablation, 23 tasks x 10 episodes x 3 seeds per row.",
        "seeds": seeds,
        "episodes_per_task_per_seed": episodes,
        "tasks": UNSEEN_TASKS,
        "methods": METHODS,
        "baseline_reference_csv": str(baseline_csv),
        "baseline_reference_note": "Baseline rows are the existing 25-episode 3-seed reference, included for comparison only.",
        "logs_root": str(logs_root),
        "paper_style_csv": str(paper_style_csv),
        "detailed_task_csv": str(detailed_csv),
        "completion": completion,
        "contact_points_in_retrieval": False,
        "target_pose_in_retrieval": False,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_markdown(
    output_dir: Path,
    rows: list[tuple[str, str, Dict[str, Optional[float]]]],
    completion: dict,
    paper_style_csv: Path,
    detailed_csv: Path,
    metadata_json: Path,
) -> Path:
    path = output_dir / "xicm_v1_10ep_component_ablation_summary.md"
    lines = [
        "# X-ICM v1 10-Episode Component Ablation",
        "",
        "This run isolates geometry/target-pose prompting, contact-point prompting, and both together.",
        "Target-pose is prompt-only. Contact points are prompt-only. Contact gamma is 0.0 for every row.",
        "",
        "Baseline rows are the existing 25-episode 3-seed references, so deltas should be treated as directional until a matched 10-episode baseline is run.",
        "",
        "## Outputs",
        "",
        f"- Paper-style scores: `{paper_style_csv}`",
        f"- Per-task/seed details: `{detailed_csv}`",
        f"- Metadata: `{metadata_json}`",
        "",
        "## Aggregate Scores",
        "",
        "| Run | Level 1 Avg | Level 2 Avg | Average | Strict finals |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, run, scores in rows:
        expanded = add_summary(scores)
        comp = completion.get(run)
        if comp:
            finals = f"{comp['strict_final_seed_task_count']}/{comp['total_seed_task_count']}"
        else:
            finals = "reference"
        lines.append(
            f"| {label} | {fmt(expanded.get('Level 1 Avg'))} | "
            f"{fmt(expanded.get('Level 2 Avg'))} | {fmt(expanded.get('Average'))} | {finals} |"
        )
    lines.append("")
    lines.append("## Method Notes")
    lines.append("")
    for run, spec in METHODS.items():
        weights = spec["weights"]
        lines.extend(
            [
                f"- `{run}`: `{spec['ranking_method']}`, "
                f"alpha={weights['alpha_dynamic']}, beta={weights['beta_geometry']}, gamma={weights['gamma_contact']}; "
                f"geometry retrieval={spec['uses_geometry_retrieval']}, "
                f"target-pose prompt={spec['uses_target_pose_prompt']}, "
                f"contact prompt={spec['uses_contact_prompt']}.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def require_complete_or_raise(completion: dict) -> None:
    missing = {
        run: item["total_seed_task_count"] - item["strict_final_seed_task_count"]
        for run, item in completion.items()
        if item["strict_final_seed_task_count"] != item["total_seed_task_count"]
    }
    if missing:
        raise SystemExit(f"Incomplete strict final scores: {missing}")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_baseline_rows(args.baseline_comparison_csv)
    method_rows, seed_scores_by_run = load_method_results(args.logs_root, seeds)
    all_rows = baseline_rows + method_rows
    completion = completion_summary(seed_scores_by_run, seeds)

    paper_style_csv = write_paper_style(args.output_dir, all_rows)
    detailed_csv = write_detailed(args.output_dir, baseline_rows, method_rows, seed_scores_by_run, seeds)
    metadata_json = write_metadata(
        args.output_dir,
        paper_style_csv,
        detailed_csv,
        args.baseline_comparison_csv,
        args.logs_root,
        seeds,
        args.episodes,
        completion,
    )
    summary_md = write_markdown(args.output_dir, all_rows, completion, paper_style_csv, detailed_csv, metadata_json)

    print(f"Wrote {paper_style_csv}")
    print(f"Wrote {detailed_csv}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {summary_md}")

    if args.require_complete:
        require_complete_or_raise(completion)


if __name__ == "__main__":
    main()
