#!/usr/bin/env python3
"""Collect X-ICM baseline and geometry/affordance ablation results.

The evaluator writes one test_data.csv per task. This script converts those
folders plus the frozen baseline CSV into a single wide comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


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

RUNS = {
    "original_xicm": None,
    "geometry": "XICM_Cross.ZS_Ranking.lang_vis.out.geo_Qwen2.5.7B.instruct_icl.18_test",
    "affordance": "XICM_Cross.ZS_Ranking.lang_vis.out.aff_Qwen2.5.7B.instruct_icl.18_test",
    "geometry_affordance": "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_Qwen2.5.7B.instruct_icl.18_test",
}

FINAL_SCORE_RE = re.compile(r"Final Score:\s*([0-9]+(?:\.[0-9]+)?)")
EPISODE_SCORE_RE = re.compile(r"Score:\s*([0-9]+(?:\.[0-9]+)?)\s*\|")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.ArgumentParser(description=__doc__).parse_args()


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=root / "test_files/xicm_baseline_results/seed0_original_prompt_scores.csv",
        help="Frozen original X-ICM baseline task-score CSV.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=root / "test_files/geometry_affordance_probe/ablation_results/cair_logs",
        help="Local copy of the CAIR X-ICM logs directory containing ablation method folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "test_files/geometry_affordance_probe/ablation_results",
        help="Directory where wide CSV/Markdown tables are written.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Evaluation episodes per task. Used to report success counts.",
    )
    parser.add_argument(
        "--allow-partial-episode-mean",
        action="store_true",
        help="For debugging only: infer a score from episode lines when Final Score is missing.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every run has all 23 numeric final task scores.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_baseline_scores(path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = row.get("task", "").strip()
            if not task:
                continue
            scores[task] = float(row["score"])
    return scores


def parse_test_data_score(path: Path, allow_partial_episode_mean: bool = False) -> Optional[float]:
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        match = FINAL_SCORE_RE.search(line)
        if match:
            return float(match.group(1))

    if not allow_partial_episode_mean:
        return None

    episode_scores: List[float] = []
    for line in lines:
        match = EPISODE_SCORE_RE.search(line)
        if match:
            episode_scores.append(float(match.group(1)))
    if episode_scores:
        return mean(episode_scores)
    return None


def load_ablation_scores(
    logs_root: Path,
    method: str,
    seed: int,
    allow_partial_episode_mean: bool = False,
) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    for task in UNSEEN_TASKS:
        score_path = logs_root / method / task / f"seed{seed}" / "test_data.csv"
        scores[task] = parse_test_data_score(score_path, allow_partial_episode_mean)
    return scores


def fmt_score(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_accuracy(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value / 100.0:.3f}"


def success_count(value: Optional[float], episodes: int) -> str:
    if value is None:
        return ""
    return str(round((value / 100.0) * episodes))


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return mean(known)


def best_run(scores: Dict[str, Optional[float]]) -> str:
    known = [(name, value) for name, value in scores.items() if value is not None]
    if not known:
        return ""
    best_name, _ = max(known, key=lambda item: item[1])
    return best_name


def make_rows(all_scores: Dict[str, Dict[str, Optional[float]]], episodes: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for task in UNSEEN_TASKS:
        task_scores = {run: all_scores[run].get(task) for run in RUNS}
        row: Dict[str, str] = {"task": task, "row_type": "task"}
        for run in RUNS:
            score = task_scores[run]
            row[f"{run}_score"] = fmt_score(score)
            row[f"{run}_accuracy"] = fmt_accuracy(score)
            row[f"{run}_successes_of_{episodes}"] = success_count(score, episodes)
        row["best_run"] = best_run(task_scores)
        row["best_score"] = fmt_score(task_scores.get(row["best_run"])) if row["best_run"] else ""
        base = task_scores.get("original_xicm")
        for run in ["geometry", "affordance", "geometry_affordance"]:
            score = task_scores[run]
            row[f"{run}_delta_vs_original"] = fmt_score(None if score is None or base is None else score - base)
        rows.append(row)

    summary_groups = {
        "MEAN_ALL": UNSEEN_TASKS,
        "MEAN_LEVEL_1": UNSEEN_TASKS[:13],
        "MEAN_LEVEL_2": UNSEEN_TASKS[13:],
    }
    for label, tasks in summary_groups.items():
        row = {"task": label, "row_type": "summary"}
        summary_scores = {
            run: mean_optional(all_scores[run].get(task) for task in tasks)
            for run in RUNS
        }
        for run in RUNS:
            score = summary_scores[run]
            row[f"{run}_score"] = fmt_score(score)
            row[f"{run}_accuracy"] = fmt_accuracy(score)
            row[f"{run}_successes_of_{episodes}"] = ""
        row["best_run"] = best_run(summary_scores)
        row["best_score"] = fmt_score(summary_scores.get(row["best_run"])) if row["best_run"] else ""
        base = summary_scores.get("original_xicm")
        for run in ["geometry", "affordance", "geometry_affordance"]:
            score = summary_scores[run]
            row[f"{run}_delta_vs_original"] = fmt_score(None if score is None or base is None else score - base)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Dict[str, str]], fieldnames: List[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    return "\n".join([header, divider] + body)


def write_markdown(path: Path, rows: List[Dict[str, str]], all_scores: Dict[str, Dict[str, Optional[float]]]) -> None:
    summary_rows = [row for row in rows if row["row_type"] == "summary"]
    task_rows = [row for row in rows if row["row_type"] == "task"]
    summary_fields = [
        "task",
        "original_xicm_score",
        "geometry_score",
        "affordance_score",
        "geometry_affordance_score",
        "best_run",
        "best_score",
        "geometry_delta_vs_original",
        "affordance_delta_vs_original",
        "geometry_affordance_delta_vs_original",
    ]
    task_fields = [
        "task",
        "original_xicm_score",
        "geometry_score",
        "affordance_score",
        "geometry_affordance_score",
        "best_run",
        "geometry_delta_vs_original",
        "affordance_delta_vs_original",
        "geometry_affordance_delta_vs_original",
    ]
    completed = {
        run: sum(1 for task in UNSEEN_TASKS if all_scores[run].get(task) is not None)
        for run in RUNS
    }
    metadata = "\n".join(f"- {run}: {count}/23 task scores" for run, count in completed.items())
    text = (
        "# X-ICM Geometry/Affordance Ablation Results\n\n"
        "Scores are numeric final success percentages from 25 evaluation episodes per task. "
        "Accuracy is score / 100 in the CSV.\n\n"
        "## Completion\n\n"
        f"{metadata}\n\n"
        "## Summary\n\n"
        f"{markdown_table(summary_rows, summary_fields)}\n\n"
        "## Task Results\n\n"
        f"{markdown_table(task_rows, task_fields)}\n"
    )
    path.write_text(text)


def write_metadata(path: Path, args: argparse.Namespace, all_scores: Dict[str, Dict[str, Optional[float]]]) -> None:
    data = {
        "baseline_csv": str(args.baseline_csv),
        "logs_root": str(args.logs_root),
        "seed": args.seed,
        "episodes": args.episodes,
        "allow_partial_episode_mean": args.allow_partial_episode_mean,
        "runs": RUNS,
        "completed_task_scores": {
            run: sum(1 for task in UNSEEN_TASKS if all_scores[run].get(task) is not None)
            for run in RUNS
        },
        "missing_final_scores": {
            run: [task for task in UNSEEN_TASKS if all_scores[run].get(task) is None]
            for run in RUNS
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_scores: Dict[str, Dict[str, Optional[float]]] = {}
    all_scores["original_xicm"] = load_baseline_scores(args.baseline_csv)
    for run, method in RUNS.items():
        if method is None:
            continue
        all_scores[run] = load_ablation_scores(
            args.logs_root,
            method,
            args.seed,
            allow_partial_episode_mean=args.allow_partial_episode_mean,
        )

    rows = make_rows(all_scores, args.episodes)
    csv_path = args.output_dir / "xicm_geometry_affordance_ablation_wide_table.csv"
    md_path = args.output_dir / "xicm_geometry_affordance_ablation_wide_table.md"
    meta_path = args.output_dir / "xicm_geometry_affordance_ablation_metadata.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, all_scores)
    write_metadata(meta_path, args, all_scores)
    print(csv_path)
    print(md_path)
    print(meta_path)
    if args.require_complete:
        incomplete = {
            run: [task for task in UNSEEN_TASKS if all_scores[run].get(task) is None]
            for run in RUNS
        }
        if any(incomplete.values()):
            raise SystemExit(
                "Missing numeric final scores: "
                + json.dumps({k: v for k, v in incomplete.items() if v}, indent=2)
            )


if __name__ == "__main__":
    main()
