#!/usr/bin/env python3
"""Tune alpha/beta/gamma for geometry-affordance retrieval on seen validation.

The validation file must contain seen-task rows only. This tuner optimizes a
retrieval proxy such as finding same-task or explicitly marked positive seen
demos in the top-k list. It must not be run on the 23 held-out AGNOSTOS unseen
tasks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from score_xicm_geometry_affordance_retrieval import (
    SEEN_TASKS,
    canonical_demo_id,
    infer_task,
    load_descriptor_cache,
    load_dynamic_score_rows,
    rank_candidates,
)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    row_path = Path(path)
    if row_path.suffix == ".jsonl":
        return [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    payload = json.loads(row_path.read_text())
    if isinstance(payload, list):
        return payload
    for key in ("validation_queries", "queries", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Cannot find validation rows in {row_path}")


def parse_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("weight grid cannot be empty")
    return values


def query_geometry(row: dict[str, Any]) -> dict[str, Any]:
    return (
        row.get("query_geometry_g_j")
        or row.get("geometry_g_j")
        or row.get("geometry")
        or row.get("geometry_g_i")
        or {}
    )


def query_affordance(row: dict[str, Any]) -> dict[str, Any]:
    return (
        row.get("query_affordance_a_j")
        or row.get("affordance_a_j")
        or row.get("affordance")
        or row.get("affordance_a_i")
        or {}
    )


def load_dynamic_rows_for_query(row: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    if isinstance(row.get("dynamic_scores"), list):
        return [dict(item) for item in row["dynamic_scores"]]
    score_file = row.get("dynamic_scores_file")
    if not score_file:
        raise ValueError(f"Validation row {row.get('query_id', '<unknown>')} is missing dynamic_scores")
    score_path = Path(score_file)
    if not score_path.is_absolute():
        score_path = base_dir / score_path
    return load_dynamic_score_rows(score_path)


def positive_sets(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    positive_ids = {str(item) for item in row.get("positive_demo_ids", [])}
    positive_tasks = {str(item) for item in row.get("positive_tasks", [])}
    task = infer_task(row)
    if task and not positive_tasks and not positive_ids:
        positive_tasks.add(task)
    return positive_ids, positive_tasks


def first_positive_rank(ranked: list[dict[str, Any]], positive_ids: set[str], positive_tasks: set[str]) -> int | None:
    for item in ranked:
        demo_id = item.get("demo_id")
        task = item.get("task")
        if (demo_id and demo_id in positive_ids) or (task and task in positive_tasks):
            return int(item["rank"])
    return None


def validate_seen_rows(rows: list[dict[str, Any]]) -> None:
    bad = []
    for row in rows:
        task = infer_task(row)
        positives = row.get("positive_tasks", [])
        if task not in SEEN_TASKS:
            bad.append(task)
        for positive_task in positives:
            if positive_task not in SEEN_TASKS:
                bad.append(positive_task)
    if bad:
        raise ValueError(
            "Seen-validation tuning received non-seen or unknown tasks; "
            f"refusing to tune on {sorted({str(item) for item in bad})}"
        )


def evaluate_weights(
    rows: list[dict[str, Any]],
    descriptor_cache: dict[str, dict[str, Any]],
    base_dir: Path,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
    exclude_self: bool,
) -> dict[str, Any]:
    per_query = []
    reciprocal_ranks = []
    recalls = []

    for row in rows:
        dynamic_rows = load_dynamic_rows_for_query(row, base_dir)
        excluded = set(row.get("exclude_demo_ids", []))
        if exclude_self:
            query_demo_id = row.get("query_demo_id") or canonical_demo_id(row)
            if query_demo_id:
                excluded.add(str(query_demo_id))

        ranked = rank_candidates(
            dynamic_rows,
            descriptor_cache,
            query_geometry(row),
            query_affordance(row),
            alpha,
            beta,
            gamma,
            top_k,
            seen_only_action="error",
            exclude_demo_ids=excluded,
        )
        positive_ids, positive_tasks = positive_sets(row)
        rank = first_positive_rank(ranked, positive_ids, positive_tasks)
        reciprocal = 1.0 / rank if rank is not None and rank <= top_k else 0.0
        recall = 1.0 if rank is not None and rank <= top_k else 0.0
        reciprocal_ranks.append(reciprocal)
        recalls.append(recall)
        per_query.append(
            {
                "query_id": row.get("query_id"),
                "task": infer_task(row),
                "first_positive_rank": rank,
                "top_demo": ranked[0]["demo_id"] if ranked else None,
                "top_task": ranked[0]["task"] if ranked else None,
            }
        )

    query_count = len(rows)
    return {
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "query_count": query_count,
        "mrr_at_k": sum(reciprocal_ranks) / query_count if query_count else 0.0,
        "recall_at_k": sum(recalls) / query_count if query_count else 0.0,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-cache", required=True)
    parser.add_argument("--validation-queries", required=True)
    parser.add_argument("--grid", default="0,0.25,0.5,0.75,1.0", help="Comma-separated values for each weight")
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--exclude-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_rows(args.validation_queries)
    validate_seen_rows(rows)
    descriptor_cache = load_descriptor_cache(args.descriptor_cache)
    base_dir = Path(args.validation_queries).resolve().parent
    grid = parse_grid(args.grid)

    results = []
    for alpha, beta, gamma in itertools.product(grid, repeat=3):
        if alpha == beta == gamma == 0:
            continue
        results.append(
            evaluate_weights(
                rows,
                descriptor_cache,
                base_dir,
                alpha,
                beta,
                gamma,
                args.top_k,
                args.exclude_self,
            )
        )

    best = max(results, key=lambda item: (item["mrr_at_k"], item["recall_at_k"], item["weights"]["alpha"]))
    payload = {
        "selection_rule": "Maximize MRR@K on seen-task validation rows, tie-break by Recall@K then alpha.",
        "leakage_guard": "Rows are rejected unless the query task and positive tasks are in the AGNOSTOS seen-task set.",
        "top_k": args.top_k,
        "grid": grid,
        "best": best,
        "all_results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": args.out, "best": best["weights"], "mrr_at_k": best["mrr_at_k"]}, indent=2))


if __name__ == "__main__":
    main()
