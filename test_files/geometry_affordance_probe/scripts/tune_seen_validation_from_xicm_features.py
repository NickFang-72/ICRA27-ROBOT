#!/usr/bin/env python3
"""Tune geometry/affordance retrieval weights from X-ICM seen-demo features.

This is the real seen-task validation tuner for the geometry/affordance
ablation. It does not inspect AGNOSTOS held-out unseen tasks.

Default split:
- validation queries: episodes >= 180 for each of the 18 seen tasks
- candidate demos: episodes < 180 for each of the 18 seen tasks

The validation proxy is same-seen-task retrieval: for a held-out seen query,
rank the earlier seen demos and reward same-task demos in the top-k list.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEEN_TASKS = {
    "turn_tap",
    "close_jar",
    "light_bulb_in",
    "slide_block_to_color_target",
    "sweep_to_dustpan_of_size",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_money_in_safe",
    "place_shape_in_shape_sorter",
    "put_item_in_drawer",
    "insert_onto_square_peg",
    "open_drawer",
    "reach_and_drag",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "meat_off_grill",
    "stack_cups",
}

GEOMETRY_FIELDS = [
    "manipulated_object",
    "key_features",
    "primary_shape",
    "part_geometry",
    "size",
    "aspect_ratio",
    "orientation",
    "opening_geometry",
    "axis_geometry",
    "symmetry",
    "clearance_geometry",
    "task_relevant_geometric_cues",
]

AFFORDANCE_FIELDS = [
    "grasp_affordance",
    "contact_affordance",
    "motion_affordance",
    "support_affordance",
    "containment_affordance",
    "articulation_affordance",
    "required_contact_region",
    "precision_requirement",
    "force_requirement",
    "failure_sensitive_property",
]


def task_episode_from_path(path: str) -> tuple[str, int]:
    parts = Path(path).parts
    if len(parts) < 4:
        raise ValueError(f"Cannot infer task/episode from {path}")
    task = parts[-4]
    match = re.search(r"episode(\d+)$", parts[-1])
    if not match:
        raise ValueError(f"Cannot infer episode id from {path}")
    return task, int(match.group(1))


def load_review_bundle(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            episode_id = row.get("episode_id")
            if task in SEEN_TASKS and episode_id is not None:
                rows[(task, int(episode_id))] = row
    return rows


def normalize_token(text: str) -> list[str]:
    compact = str(text).strip().lower().replace(" ", "_")
    parts = [part for part in re.split(r"[^a-z0-9]+", compact) if part]
    if compact and compact != "unknown":
        parts.append(compact)
    return parts


def flatten_descriptor_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_descriptor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (int, float)):
                continue
            yield from flatten_descriptor_values(item)
    elif isinstance(value, str):
        yield from normalize_token(value)
    elif not isinstance(value, (int, float, bool)):
        yield from normalize_token(str(value))


def descriptor_tokens(descriptor: dict[str, Any], fields: Iterable[str]) -> set[str]:
    tokens = set()
    for field in fields:
        tokens.update(flatten_descriptor_values(descriptor.get(field)))
    return {token for token in tokens if token not in {"none", "unknown", "null"}}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out.append((x, y))
    return out


def point_similarity(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return 0.0
    distances = []
    for ax, ay in a:
        distances.append(min(math.dist((ax, ay), (bx, by)) for bx, by in b))
    mean_distance = sum(distances) / len(distances)
    return max(0.0, 1.0 - mean_distance / math.sqrt(2.0))


def minmax_by_query(values: np.ndarray) -> np.ndarray:
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    span = hi - lo
    return np.divide(values - lo, span, out=np.zeros_like(values, dtype=np.float32), where=span != 0)


def build_component_matrices(
    feature_rows: list[dict[str, Any]],
    review_rows: dict[tuple[str, int], dict[str, Any]],
    memory_features: np.ndarray,
    validation_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    indexed = []
    for idx, row in enumerate(feature_rows):
        key = (row["task"], row["episode_id"])
        review = review_rows.get(key)
        if review is None:
            continue
        item = {**row, "feature_index": idx, "review": review}
        indexed.append(item)

    queries = [row for row in indexed if row["episode_id"] >= validation_start]
    candidates = [row for row in indexed if row["episode_id"] < validation_start]
    if not queries or not candidates:
        raise ValueError("Validation split produced no queries or candidates")

    query_features = memory_features[[row["feature_index"] for row in queries]]
    candidate_features = memory_features[[row["feature_index"] for row in candidates]]
    s_dyn = minmax_by_query(query_features @ candidate_features.T)

    candidate_geo = []
    candidate_aff = []
    candidate_points = []
    for row in candidates:
        review = row["review"]
        geo = review.get("geometry_g_i") or {}
        aff = review.get("affordance_a_i") or {}
        candidate_geo.append(descriptor_tokens(geo, GEOMETRY_FIELDS))
        candidate_aff.append(descriptor_tokens(aff, AFFORDANCE_FIELDS))
        candidate_points.append(points(aff.get("preferred_contact_points")))

    s_geo = np.zeros((len(queries), len(candidates)), dtype=np.float32)
    s_aff = np.zeros((len(queries), len(candidates)), dtype=np.float32)

    for qi, query in enumerate(queries):
        q_review = query["review"]
        q_geo = descriptor_tokens(q_review.get("geometry_g_i") or {}, GEOMETRY_FIELDS)
        q_aff_doc = q_review.get("affordance_a_i") or {}
        q_aff = descriptor_tokens(q_aff_doc, AFFORDANCE_FIELDS)
        q_points = points(q_aff_doc.get("preferred_contact_points"))
        for ci in range(len(candidates)):
            s_geo[qi, ci] = jaccard(candidate_geo[ci], q_geo)
            label_score = jaccard(candidate_aff[ci], q_aff)
            if q_points and candidate_points[ci]:
                s_aff[qi, ci] = 0.8 * label_score + 0.2 * point_similarity(candidate_points[ci], q_points)
            else:
                s_aff[qi, ci] = label_score

    return queries, candidates, s_dyn, s_geo, s_aff


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, abs_tol=1e-8):
        raise ValueError("--grid-step must divide 1.0 exactly, e.g. 0.1 or 0.05")
    out = []
    for ai in range(units + 1):
        for bi in range(units + 1 - ai):
            gi = units - ai - bi
            out.append((ai * step, bi * step, gi * step))
    return out


def topk_metrics(
    scores: np.ndarray,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    reciprocal_ranks = []
    recalls = []
    same_task_counts = []
    first_ranks = []
    candidate_tasks = np.array([row["task"] for row in candidates])

    for qi, query in enumerate(queries):
        order = np.argsort(scores[qi])[::-1]
        same = candidate_tasks[order] == query["task"]
        positive_positions = np.flatnonzero(same)
        first_rank = int(positive_positions[0] + 1) if len(positive_positions) else None
        first_ranks.append(first_rank)
        reciprocal_ranks.append(1.0 / first_rank if first_rank and first_rank <= top_k else 0.0)
        recalls.append(1.0 if first_rank and first_rank <= top_k else 0.0)
        same_task_counts.append(int(same[:top_k].sum()))

    return {
        "mrr_at_k": float(np.mean(reciprocal_ranks)),
        "recall_at_k": float(np.mean(recalls)),
        "mean_same_task_in_top_k": float(np.mean(same_task_counts)),
        "median_first_positive_rank": float(np.median([rank for rank in first_ranks if rank is not None])),
    }


def summarize_best_examples(
    scores: np.ndarray,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    out = []
    for qi, query in enumerate(queries[:limit]):
        order = np.argsort(scores[qi])[::-1][:top_k]
        out.append(
            {
                "query_demo_id": f"{query['task']}_episode{query['episode_id']}",
                "query_instruction": query["review"].get("language_description"),
                "top_k": [
                    {
                        "demo_id": f"{candidates[ci]['task']}_episode{candidates[ci]['episode_id']}",
                        "task": candidates[ci]["task"],
                        "score": float(scores[qi, ci]),
                        "instruction": candidates[ci]["review"].get("language_description"),
                    }
                    for ci in order
                ],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-bundle", required=True)
    parser.add_argument("--features-pkl", required=True)
    parser.add_argument("--validation-start", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    review_rows = load_review_bundle(args.review_bundle)
    with Path(args.features_pkl).open("rb") as f:
        features = pickle.load(f)
    memory_features = np.concatenate(
        [features["all_prompt_feats"], features["all_output_image_feats"]],
        axis=1,
    ).astype(np.float32)

    feature_rows = []
    for path in features["all_demo_paths"]:
        task, episode_id = task_episode_from_path(path)
        if task in SEEN_TASKS:
            feature_rows.append({"path": path, "task": task, "episode_id": episode_id})

    queries, candidates, s_dyn, s_geo, s_aff = build_component_matrices(
        feature_rows,
        review_rows,
        memory_features,
        args.validation_start,
    )

    results = []
    for alpha, beta, gamma in simplex_grid(args.grid_step):
        scores = alpha * s_dyn + beta * s_geo + gamma * s_aff
        metrics = topk_metrics(scores, queries, candidates, args.top_k)
        results.append(
            {
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                **metrics,
            }
        )

    best = max(
        results,
        key=lambda row: (
            row["mrr_at_k"],
            row["recall_at_k"],
            row["mean_same_task_in_top_k"],
            row["weights"]["alpha"],
        ),
    )
    best_weights = best["weights"]
    best_scores = (
        best_weights["alpha"] * s_dyn
        + best_weights["beta"] * s_geo
        + best_weights["gamma"] * s_aff
    )
    baseline_rows = {}
    for name, weights in {
        "dynamic_only": (1.0, 0.0, 0.0),
        "geometry_only": (0.0, 1.0, 0.0),
        "affordance_only": (0.0, 0.0, 1.0),
        "equal": (1 / 3, 1 / 3, 1 / 3),
    }.items():
        scores = weights[0] * s_dyn + weights[1] * s_geo + weights[2] * s_aff
        baseline_rows[name] = {"weights": {"alpha": weights[0], "beta": weights[1], "gamma": weights[2]}, **topk_metrics(scores, queries, candidates, args.top_k)}

    by_task = {}
    for task in sorted({query["task"] for query in queries}):
        query_ids = [idx for idx, query in enumerate(queries) if query["task"] == task]
        by_task[task] = topk_metrics(best_scores[query_ids], [queries[idx] for idx in query_ids], candidates, args.top_k)

    payload = {
        "method": "seen_task_validation_same_task_retrieval_proxy",
        "leakage_guard": "Uses only the 18 AGNOSTOS seen tasks. Held-out unseen tasks are not loaded or scored.",
        "split": {
            "validation_query_rule": f"episode_id >= {args.validation_start}",
            "candidate_rule": f"episode_id < {args.validation_start}",
            "num_queries": len(queries),
            "num_candidates": len(candidates),
            "top_k": args.top_k,
        },
        "score_formula": "score(D_i,Q_j)=alpha*S_dyn(D_i,Q_j)+beta*S_geo(g_i,g_j)+gamma*S_aff(a_i,a_j)",
        "component_notes": {
            "S_dyn": "Original X-ICM prompt/output diffusion feature dot product, min-max normalized per query.",
            "S_geo": "Jaccard similarity over normalized geometry descriptor tokens.",
            "S_aff": "Jaccard similarity over normalized affordance descriptor tokens, with contact point proximity when both sides have points.",
        },
        "grid_step": args.grid_step,
        "best": best,
        "baselines": baseline_rows,
        "by_task_best": by_task,
        "top_results": sorted(results, key=lambda row: (row["mrr_at_k"], row["mean_same_task_in_top_k"]), reverse=True)[:20],
        "examples_best": summarize_best_examples(best_scores, queries, candidates, args.top_k),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(out), "best": best, "split": payload["split"]}, indent=2))


if __name__ == "__main__":
    main()
