#!/usr/bin/env python3
"""Collect the v1 goal-state descriptor smoke benchmark against baseline rows."""

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

BASELINE_COMPARISON_CSV = (
    "test_files/xicm_baseline_results/vanilla_rerun_3seed_2026-06-22/"
    "vanilla_baseline_rerun_3seed_paper_style_scores.csv"
)
BASELINE_LOG_ROOT = (
    "test_files/xicm_baseline_results/vanilla_rerun_3seed_2026-06-22/cair_logs/"
    "XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test"
)
SMOKE_METHOD = "XICM_Cross.ZS_Ranking.lang_vis.out.geo_Qwen2.5.7B.instruct_icl.18_test"

FINAL_SCORE_RE = re.compile(r"Final Score:\s*([-+]?\d+(?:\.\d+)?)")
EPISODE_SCORE_RE = re.compile(r"Episode\s+(\d+)\s+\|.*?Score:\s*([-+]?\d+(?:\.\d+)?)\s+\|")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output = root / "test_files/geometry_affordance_probe/ablation_results/v1_pose_smoke_5eps_2026-06-25"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=default_output / "cair_logs")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--baseline-comparison-csv", type=Path, default=root / BASELINE_COMPARISON_CSV)
    parser.add_argument("--baseline-log-root", type=Path, default=root / BASELINE_LOG_ROOT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


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
    wanted_runs = {"paper_xicm_7b", "vanilla_baseline_3seed_rerun"}
    rows: list[tuple[str, str, dict[str, Optional[float]]]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run = row.get("run", "")
            if run not in wanted_runs:
                continue
            scores = {
                task: float(row[task]) if row.get(task, "") else None
                for task in UNSEEN_TASKS
            }
            rows.append((row.get("method", run), run, scores))
    rows.sort(key=lambda item: 0 if item[1] == "paper_xicm_7b" else 1)
    return rows


def parse_final_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        match = FINAL_SCORE_RE.search(line)
        if match:
            return float(match.group(1))
    return None


def parse_first_n_episode_score(path: Path, episodes: int) -> Optional[float]:
    if not path.exists():
        return None
    by_episode: dict[int, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = EPISODE_SCORE_RE.search(line)
        if match:
            by_episode[int(match.group(1))] = float(match.group(2))
    values = [by_episode.get(index) for index in range(episodes)]
    if any(value is None for value in values):
        return None
    return mean(value for value in values if value is not None)


def load_smoke_scores(logs_root: Path, seed: int) -> dict[str, Optional[float]]:
    return {
        task: parse_final_score(logs_root / SMOKE_METHOD / task / f"seed{seed}" / "test_data.csv")
        for task in UNSEEN_TASKS
    }


def load_baseline_first_n_scores(log_root: Path, seed: int, episodes: int) -> dict[str, Optional[float]]:
    return {
        task: parse_first_n_episode_score(log_root / task / f"seed{seed}" / "test_data.csv", episodes)
        for task in UNSEEN_TASKS
    }


def success_count(score: Optional[float], episodes: int) -> str:
    if score is None:
        return ""
    return str(round((score / 100.0) * episodes))


def write_paper_style(
    output_dir: Path,
    rows: list[tuple[str, str, dict[str, Optional[float]]]],
) -> Path:
    path = output_dir / "xicm_v1_pose_smoke_5eps_scores.csv"
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
    paper_rows: list[tuple[str, str, dict[str, Optional[float]]]],
    baseline_smoke: dict[str, Optional[float]],
    smoke_scores: dict[str, Optional[float]],
    episodes: int,
) -> Path:
    path = output_dir / "xicm_v1_pose_smoke_5eps_task_scores.csv"
    baseline_by_run = {run: scores for _label, run, scores in paper_rows}
    columns = [
        "task",
        "paper_xicm_7b_75rollout_score",
        "vanilla_baseline_3seed_75rollout_score",
        f"vanilla_baseline_seed0_first{episodes}_score",
        f"vanilla_baseline_seed0_first{episodes}_successes",
        f"v1_goal_state_geo_seed0_{episodes}eps_score",
        f"v1_goal_state_geo_seed0_{episodes}eps_successes",
        "delta_v1_vs_baseline_first5",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in UNSEEN_TASKS:
            baseline = baseline_smoke.get(task)
            smoke = smoke_scores.get(task)
            writer.writerow(
                {
                    "task": task,
                    "paper_xicm_7b_75rollout_score": fmt(baseline_by_run["paper_xicm_7b"].get(task)),
                    "vanilla_baseline_3seed_75rollout_score": fmt(baseline_by_run["vanilla_baseline_3seed_rerun"].get(task)),
                    f"vanilla_baseline_seed0_first{episodes}_score": fmt(baseline),
                    f"vanilla_baseline_seed0_first{episodes}_successes": success_count(baseline, episodes),
                    f"v1_goal_state_geo_seed0_{episodes}eps_score": fmt(smoke),
                    f"v1_goal_state_geo_seed0_{episodes}eps_successes": success_count(smoke, episodes),
                    "delta_v1_vs_baseline_first5": fmt(None if baseline is None or smoke is None else smoke - baseline),
                    "status": "complete" if smoke is not None else "pending",
                }
            )
    return path


def write_metadata(
    output_dir: Path,
    paper_style: Path,
    detailed: Path,
    smoke_scores: dict[str, Optional[float]],
    episodes: int,
    seed: int,
) -> Path:
    path = output_dir / "xicm_v1_pose_smoke_5eps_metadata.json"
    complete = sum(1 for task in UNSEEN_TASKS if smoke_scores.get(task) is not None)
    payload = {
        "run": "v1_goal_state_geo_smoke",
        "seed": seed,
        "episodes_per_task": episodes,
        "total_expected_seed_task_final_scores": len(UNSEEN_TASKS),
        "complete_seed_task_final_scores": complete,
        "ranking_method": "lang_vis.out.geo",
        "method": SMOKE_METHOD,
        "demo_num_per_icl": 18,
        "retrieval_weights": {"alpha_dynamic": 0.70, "beta_geometry": 0.30, "gamma_contact": 0.0},
        "contact_points_in_prompt": False,
        "contact_points_in_retrieval": False,
        "goal_state_contact_pose_descriptor_in_prompt": True,
        "paper_style_csv": str(paper_style),
        "task_scores_csv": str(detailed),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_rows = load_baseline_rows(args.baseline_comparison_csv)
    baseline_smoke = load_baseline_first_n_scores(args.baseline_log_root, args.seed, args.episodes)
    smoke_scores = load_smoke_scores(args.logs_root, args.seed)
    rows = [
        *paper_rows,
        (f"X-ICM 7B vanilla seed0 first-{args.episodes} smoke baseline", f"vanilla_baseline_seed0_first{args.episodes}", baseline_smoke),
        (f"X-ICM 7B v1 + geometry + goal-state descriptor seed0 {args.episodes}-episode smoke", "v1_goal_state_geo_seed0_5eps", smoke_scores),
    ]
    paper_style = write_paper_style(args.output_dir, rows)
    detailed = write_task_scores(args.output_dir, paper_rows, baseline_smoke, smoke_scores, args.episodes)
    metadata = write_metadata(args.output_dir, paper_style, detailed, smoke_scores, args.episodes, args.seed)

    complete = sum(1 for task in UNSEEN_TASKS if smoke_scores.get(task) is not None)
    print(f"Wrote {paper_style}")
    print(f"Wrote {detailed}")
    print(f"Wrote {metadata}")
    print(f"v1 pose smoke strict seed-task final scores: {complete}/{len(UNSEEN_TASKS)}")
    if args.require_complete and complete != len(UNSEEN_TASKS):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
