#!/usr/bin/env python3
"""Collect a vanilla X-ICM baseline rerun against the paper 7B scores."""

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

PAPER_STYLE_CSV = (
    "test_files/geometry_affordance_probe/ablation_results/"
    "xicm_geometry_affordance_ablation_paper_style_scores.csv"
)
FROZEN_BASELINE_CSV = "test_files/xicm_baseline_results/seed0_original_prompt_scores.csv"
METHOD = "XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test"
FINAL_SCORE_RE = re.compile(r"Final Score:\s*([-+]?\d+(?:\.\d+)?)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output = root / "test_files/xicm_baseline_results/vanilla_rerun_3seed_2026-06-22"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-style-csv",
        type=Path,
        default=root / PAPER_STYLE_CSV,
        help="Existing paper-style CSV containing paper_xicm_7b.",
    )
    parser.add_argument(
        "--frozen-baseline-csv",
        type=Path,
        default=root / FROZEN_BASELINE_CSV,
        help="Frozen local vanilla seed0 baseline CSV from the first rerun.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=default_output / "cair_logs",
        help="Local copy of the new vanilla rerun CAIR logs.",
    )
    parser.add_argument(
        "--method",
        default=METHOD,
        help="Method directory name for the vanilla rerun.",
    )
    parser.add_argument(
        "--seeds",
        default="0,50,99",
        help="Comma-separated seeds to average for the new rerun.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated comparison files.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless the new rerun has all 23 strict final scores.",
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


def load_paper_7b_scores(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") == "paper_xicm_7b":
                return {task: float(row[task]) if row.get(task) else None for task in UNSEEN_TASKS}
    raise ValueError(f"paper_xicm_7b row not found in {path}")


def load_task_score_csv(path: Path) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {task: None for task in UNSEEN_TASKS}
    if not path.exists():
        return scores
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row.get("task", "").strip()
            if task in scores and row.get("score", "") != "":
                scores[task] = float(row["score"])
    return scores


def parse_test_data_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        match = FINAL_SCORE_RE.search(line)
        if match:
            return float(match.group(1))
    return None


def load_rerun_seed_scores(logs_root: Path, method: str, seed: int) -> Dict[str, Optional[float]]:
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


def add_summary(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    output = dict(scores)
    output["Level 1 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_1_TASKS)
    output["Level 2 Avg"] = mean_optional(output[task] for task in UNSEEN_TASKS if task in LEVEL_2_TASKS)
    output["Average"] = mean_optional(output[task] for task in UNSEEN_TASKS)
    return output


def write_paper_style(output_dir: Path, rows: list[tuple[str, str, Dict[str, Optional[float]]]]) -> Path:
    path = output_dir / "vanilla_baseline_rerun_3seed_paper_style_scores.csv"
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
    rows: list[tuple[str, str, Dict[str, Optional[float]]]],
    seed_scores: Dict[int, Dict[str, Optional[float]]],
) -> Path:
    path = output_dir / "vanilla_baseline_rerun_3seed_task_scores.csv"
    by_run = {run: scores for _, run, scores in rows}
    columns = [
        "task",
        "paper_xicm_7b",
        *[f"vanilla_seed{seed}" for seed in seed_scores],
        "vanilla_3seed_mean",
        "delta_new_vs_paper",
        "frozen_xicm_7b_seed0_rerun",
        "delta_new_vs_frozen",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in UNSEEN_TASKS:
            paper = by_run["paper_xicm_7b"].get(task)
            frozen = by_run["frozen_xicm_7b_rerun"].get(task)
            new = by_run["vanilla_baseline_3seed_rerun"].get(task)
            seed_values = {f"vanilla_seed{seed}": fmt(scores.get(task)) for seed, scores in seed_scores.items()}
            writer.writerow(
                {
                    "task": task,
                    "paper_xicm_7b": fmt(paper),
                    **seed_values,
                    "vanilla_3seed_mean": fmt(new),
                    "delta_new_vs_paper": fmt(None if new is None or paper is None else new - paper),
                    "frozen_xicm_7b_seed0_rerun": fmt(frozen),
                    "delta_new_vs_frozen": fmt(None if new is None or frozen is None else new - frozen),
                    "status": "complete_all_3_seeds" if new is not None else "pending_seed_results",
                }
            )
    return path


def write_markdown(output_dir: Path, rows: list[tuple[str, str, Dict[str, Optional[float]]]]) -> Path:
    path = output_dir / "vanilla_baseline_rerun_3seed_table.md"
    columns = ["Method", *UNSEEN_TASKS, "Level 1 Avg", "Level 2 Avg", "Average"]
    lines = [
        "# Vanilla X-ICM Baseline Rerun",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in columns[1:]]]) + " |",
    ]
    for label, _run, scores in rows:
        expanded = add_summary(scores)
        lines.append("| " + " | ".join([label, *[fmt(expanded.get(task)) for task in columns[1:]]]) + " |")
    lines.append("")
    lines.append("Scores are strict final task means across seeds 0, 50, and 99. Blank cells mean at least one seed for that task has not finished yet.")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(
    output_dir: Path,
    paper_style: Path,
    detailed: Path,
    markdown: Path,
    complete_count: int,
    total_seed_task_count: int,
    complete_task_mean_count: int,
    method: str,
    seeds: list[int],
) -> Path:
    path = output_dir / "vanilla_baseline_rerun_3seed_metadata.json"
    payload = {
        "method": method,
        "seeds": seeds,
        "strict_final_seed_task_count": complete_count,
        "total_seed_task_count": total_seed_task_count,
        "complete_task_mean_count": complete_task_mean_count,
        "total_tasks": len(UNSEEN_TASKS),
        "paper_style_csv": str(paper_style),
        "detailed_csv": str(detailed),
        "markdown_table": str(markdown),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper = load_paper_7b_scores(args.paper_style_csv)
    frozen = load_task_score_csv(args.frozen_baseline_csv)
    seed_scores = {seed: load_rerun_seed_scores(args.logs_root, args.method, seed) for seed in seeds}
    rerun = average_complete_seed_scores(seed_scores)

    rows = [
        ("X-ICM 7B (paper)", "paper_xicm_7b", paper),
        ("X-ICM 7B vanilla 3-seed rerun", "vanilla_baseline_3seed_rerun", rerun),
        ("X-ICM 7B frozen rerun", "frozen_xicm_7b_rerun", frozen),
    ]

    paper_style = write_paper_style(args.output_dir, rows)
    detailed = write_detailed(args.output_dir, rows, seed_scores)
    markdown = write_markdown(args.output_dir, rows)

    complete_count = sum(
        1
        for scores in seed_scores.values()
        for task in UNSEEN_TASKS
        if scores.get(task) is not None
    )
    total_seed_task_count = len(seeds) * len(UNSEEN_TASKS)
    complete_task_mean_count = sum(1 for task in UNSEEN_TASKS if rerun.get(task) is not None)
    metadata = write_metadata(
        args.output_dir,
        paper_style,
        detailed,
        markdown,
        complete_count,
        total_seed_task_count,
        complete_task_mean_count,
        args.method,
        seeds,
    )

    print(f"Wrote {paper_style}")
    print(f"Wrote {detailed}")
    print(f"Wrote {markdown}")
    print(f"Wrote {metadata}")
    print(f"New vanilla rerun strict seed-task final scores: {complete_count}/{total_seed_task_count}")

    if args.require_complete and complete_count != total_seed_task_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
