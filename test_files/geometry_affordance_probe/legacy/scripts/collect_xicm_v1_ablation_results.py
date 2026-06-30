#!/usr/bin/env python3
"""Collect clean v1 X-ICM geometry/contact ablations against the baseline rerun."""

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

V1_METHODS = {
    "v1_geometry_retrieval_3seed": {
        "label": "X-ICM 7B v1 + geometry retrieval",
        "method": "XICM_Cross.ZS_Ranking.lang_vis.out.geo_Qwen2.5.7B.instruct_icl.18_test",
    },
    "v1_geometry_retrieval_contact_prompt_3seed": {
        "label": "X-ICM 7B v1 + geometry retrieval + contact prompt",
        "method": "XICM_Cross.ZS_Ranking.lang_vis.out.geo.aff_Qwen2.5.7B.instruct_icl.18_test",
    },
}

FINAL_SCORE_RE = re.compile(r"Final Score:\s*([-+]?\d+(?:\.\d+)?)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output = root / "test_files/geometry_affordance_probe/ablation_results/v1_3seed_2026-06-24"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-comparison-csv",
        type=Path,
        default=root / BASELINE_COMPARISON_CSV,
        help="Paper-style CSV containing the paper 7B row and clean 3-seed vanilla baseline row.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=default_output / "cair_logs",
        help="Local copy of CAIR logs for the v1 ablations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated v1 comparison files.",
    )
    parser.add_argument(
        "--seeds",
        default="0,50,99",
        help="Comma-separated seeds to average.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every configured method has all strict seed-task final scores.",
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


def load_v1_method_results(logs_root: Path, seeds: list[int]) -> tuple[
    list[tuple[str, str, Dict[str, Optional[float]]]],
    Dict[str, Dict[int, Dict[str, Optional[float]]]],
]:
    rows: list[tuple[str, str, Dict[str, Optional[float]]]] = []
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for run, spec in V1_METHODS.items():
        seed_scores = {
            seed: load_seed_scores(logs_root, spec["method"], seed)
            for seed in seeds
        }
        averaged = average_complete_seed_scores(seed_scores)
        rows.append((spec["label"], run, averaged))
        seed_scores_by_run[run] = seed_scores
    return rows, seed_scores_by_run


def write_paper_style(output_dir: Path, rows: list[tuple[str, str, Dict[str, Optional[float]]]]) -> Path:
    path = output_dir / "xicm_v1_ablation_3seed_paper_style_scores.csv"
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
    v1_rows: list[tuple[str, str, Dict[str, Optional[float]]]],
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    seeds: list[int],
) -> Path:
    path = output_dir / "xicm_v1_ablation_3seed_task_scores.csv"
    baseline_by_run = {run: scores for _label, run, scores in baseline_rows}
    v1_by_run = {run: scores for _label, run, scores in v1_rows}
    columns = [
        "task",
        "paper_xicm_7b",
        "vanilla_baseline_3seed_rerun",
        *[f"geo_seed{seed}" for seed in seeds],
        "v1_geometry_retrieval_3seed",
        "delta_geo_vs_baseline",
        *[f"geo_contact_seed{seed}" for seed in seeds],
        "v1_geometry_retrieval_contact_prompt_3seed",
        "delta_geo_contact_vs_baseline",
        "delta_geo_contact_vs_geo",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in UNSEEN_TASKS:
            paper = baseline_by_run["paper_xicm_7b"].get(task)
            baseline = baseline_by_run["vanilla_baseline_3seed_rerun"].get(task)
            geo = v1_by_run["v1_geometry_retrieval_3seed"].get(task)
            geo_contact = v1_by_run["v1_geometry_retrieval_contact_prompt_3seed"].get(task)
            geo_seed_values = {
                f"geo_seed{seed}": fmt(seed_scores_by_run["v1_geometry_retrieval_3seed"][seed].get(task))
                for seed in seeds
            }
            contact_seed_values = {
                f"geo_contact_seed{seed}": fmt(seed_scores_by_run["v1_geometry_retrieval_contact_prompt_3seed"][seed].get(task))
                for seed in seeds
            }
            status = []
            status.append("geo_complete" if geo is not None else "geo_pending")
            status.append("geo_contact_complete" if geo_contact is not None else "geo_contact_pending")
            writer.writerow(
                {
                    "task": task,
                    "paper_xicm_7b": fmt(paper),
                    "vanilla_baseline_3seed_rerun": fmt(baseline),
                    **geo_seed_values,
                    "v1_geometry_retrieval_3seed": fmt(geo),
                    "delta_geo_vs_baseline": fmt(None if geo is None or baseline is None else geo - baseline),
                    **contact_seed_values,
                    "v1_geometry_retrieval_contact_prompt_3seed": fmt(geo_contact),
                    "delta_geo_contact_vs_baseline": fmt(None if geo_contact is None or baseline is None else geo_contact - baseline),
                    "delta_geo_contact_vs_geo": fmt(None if geo_contact is None or geo is None else geo_contact - geo),
                    "status": ";".join(status),
                }
            )
    return path


def write_markdown(output_dir: Path, rows: list[tuple[str, str, Dict[str, Optional[float]]]]) -> Path:
    path = output_dir / "xicm_v1_ablation_3seed_table.md"
    columns = ["Method", *UNSEEN_TASKS, "Level 1 Avg", "Level 2 Avg", "Average"]
    lines = [
        "# X-ICM v1 3-Seed Ablations",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in columns[1:]]]) + " |",
    ]
    for label, _run, scores in rows:
        expanded = add_summary(scores)
        lines.append("| " + " | ".join([label, *[fmt(expanded.get(task)) for task in columns[1:]]]) + " |")
    lines.extend(
        [
            "",
            "A v1 task score is the mean of seeds 0, 50, and 99, each with 25 evaluation episodes. Blank cells mean at least one seed for that task has not finished yet.",
            "The contact-prompt row uses the same geometry retrieval score as the geometry-only row; contact hints are prompt-side evidence only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(
    output_dir: Path,
    paper_style: Path,
    detailed: Path,
    markdown: Path,
    seed_scores_by_run: Dict[str, Dict[int, Dict[str, Optional[float]]]],
    seeds: list[int],
) -> Path:
    path = output_dir / "xicm_v1_ablation_3seed_metadata.json"
    per_run_counts = {}
    for run, seed_scores in seed_scores_by_run.items():
        complete_seed_tasks = sum(
            1
            for scores in seed_scores.values()
            for task in UNSEEN_TASKS
            if scores.get(task) is not None
        )
        complete_task_means = sum(
            1
            for task in UNSEEN_TASKS
            if all(seed_scores[seed].get(task) is not None for seed in seeds)
        )
        per_run_counts[run] = {
            "method": V1_METHODS[run]["method"],
            "strict_final_seed_task_count": complete_seed_tasks,
            "total_seed_task_count": len(seeds) * len(UNSEEN_TASKS),
            "complete_task_mean_count": complete_task_means,
            "total_tasks": len(UNSEEN_TASKS),
        }
    payload = {
        "seeds": seeds,
        "episodes_per_task_per_seed": 25,
        "demo_num_per_icl": 18,
        "retrieval_weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "contact_points_in_retrieval": False,
        "per_run_counts": per_run_counts,
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

    baseline_rows = load_baseline_rows(args.baseline_comparison_csv)
    v1_rows, seed_scores_by_run = load_v1_method_results(args.logs_root, seeds)
    rows = [*baseline_rows, *v1_rows]

    paper_style = write_paper_style(args.output_dir, rows)
    detailed = write_detailed(args.output_dir, baseline_rows, v1_rows, seed_scores_by_run, seeds)
    markdown = write_markdown(args.output_dir, rows)
    metadata = write_metadata(args.output_dir, paper_style, detailed, markdown, seed_scores_by_run, seeds)

    print(f"Wrote {paper_style}")
    print(f"Wrote {detailed}")
    print(f"Wrote {markdown}")
    print(f"Wrote {metadata}")

    total_expected = len(V1_METHODS) * len(seeds) * len(UNSEEN_TASKS)
    complete = sum(
        1
        for seed_scores in seed_scores_by_run.values()
        for scores in seed_scores.values()
        for task in UNSEEN_TASKS
        if scores.get(task) is not None
    )
    print(f"v1 strict seed-task final scores: {complete}/{total_expected}")
    if args.require_complete and complete != total_expected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
