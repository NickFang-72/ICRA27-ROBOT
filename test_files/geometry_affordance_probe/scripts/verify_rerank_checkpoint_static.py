#!/usr/bin/env python3
"""Static checks for the active double-retrieval rerank checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


RETRIEVAL_RERANKING_ALIASES = {
    "retrieval_reranking",
    "retrieval-reranking",
    "rerank_top50",
    "rerank-top50",
}

CHECKS = [
    (
        "active README names the double-retrieval checkpoint",
        "test_files/geometry_affordance_probe/README.md",
        [
            "current double-retrieval rerank checkpoint",
            "lang_vis.out.geo.aff_v3.rerank_top50",
            "Broad retrieval with the original X-ICM dynamic diffusion similarity",
            "Fine rerank that shortlist with geometry, target-pose/profile compatibility",
            "phase/action-chain retrieval are not part of this checkpoint",
        ],
    ),
    (
        "gitignore keeps generated and parked phase files out of git add",
        ".gitignore",
        [
            "*.mp4",
            "*.inspect.ndjson",
            "X-ICM/scripts/generate_phase_review_trace.py",
            "current push surface is the frozen",
        ],
    ),
    (
        "freeze note marks rerank as active",
        "test_files/geometry_affordance_probe/retrieval_reranking_model_freeze.md",
        [
            "active retrieval reranking checkpoint",
            "Broad retrieval with X-ICM dynamic diffusion similarity",
            "Fine rerank inside that shortlist",
            "Query contact points `c_j` are prompt hints only",
            "must stay on separate ranking methods containing `.phase` or `action_chain`",
        ],
    ),
    (
        "ranking code has broad and fine rerank stages",
        "X-ICM/form_icl_demonstrations_crosstask_ranking.py",
        [
            "RETRIEVAL_RERANKING_ALIASES",
            "def _is_retrieval_reranking_metric",
            "def _rerank_candidate_count",
            "def _dynamic_shortlist_candidates",
            "candidates = _dynamic_shortlist_candidates(",
            "score += delta * s_profile - penalty_weight * penalty",
            "ranked = _attention_bias_for_ranked_items(ranked)",
        ],
    ),
    (
        "final prompt keeps seen contact hints internal",
        "X-ICM/form_icl_demonstrations_crosstask_ranking.py",
        [
            "Role-labeled oracle contact points c_j, when present, are final-action hints only; they were not used to retrieve the demonstrations.",
            "Retrieved seen demonstrations intentionally omit c_i contact hints.",
            "_format_compact_query_contact_hints(query_affordance)",
        ],
    ),
    (
        "rerank review trace defaults to explicit active metric",
        "X-ICM/scripts/generate_rerank_review_trace.py",
        [
            'parser.add_argument("--ranking-metric", default="lang_vis.out.geo.aff_v3.rerank_top50")',
            "02_broad_dynamic_top50.csv",
            "03_fine_rerank_scored_pool_top50.csv",
            "05_final_prompt.txt",
        ],
    ),
    (
        "CAIR rerank runner defaults to explicit active metric",
        "test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh",
        [
            'RANKING_METHOD="${RANKING_METHOD:-lang_vis.out.geo.aff_v3.rerank_top50}"',
            'RERANK_CANDIDATES="${RERANK_CANDIDATES:-50}"',
            'export XICM_GA_RERANK_CANDIDATES="$RERANK_CANDIDATES"',
            "bash scripts/eval_XICM.sh",
        ],
    ),
    (
        "one-episode video smoke runner uses active rerank method",
        "test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh",
        [
            'RANKING_METHOD="${RANKING_METHOD:-lang_vis.out.geo.aff_v3.rerank_top50}"',
            'K="${K:-8}"',
            'ENABLE_VIDEO="${ENABLE_VIDEO:-1}"',
            'RECORD_EVERY_N="${RECORD_EVERY_N:-1}"',
        ],
    ),
    (
        "script index exposes only current rerank scripts",
        "test_files/geometry_affordance_probe/SCRIPT_INDEX.md",
        [
            "Current Double-Retrieval Files",
            "scripts/verify_rerank_checkpoint_static.py",
            "run_rerank_top50_all23_1ep_video_on_cair.sh",
            "Ignored Local Work",
            "parked phase/action-chain branch is intentionally ignored",
        ],
    ),
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def normalized_metric_tokens(ranking_metric: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(ranking_metric).strip().lower())
        if token
    }


def is_retrieval_reranking_metric(ranking_metric: str) -> bool:
    normalized = str(ranking_metric).strip().lower().replace("-", "_")
    tokens = normalized_metric_tokens(normalized)
    return any(alias.replace("-", "_") in normalized for alias in RETRIEVAL_RERANKING_ALIASES) or {
        "retrieval",
        "reranking",
    }.issubset(tokens)


def is_phase_metric(ranking_metric: str) -> bool:
    normalized = str(ranking_metric).strip().lower().replace("-", "_")
    tokens = normalized_metric_tokens(normalized)
    return "phase" in tokens or "phases" in tokens or "action_chain" in normalized or "actionchain" in normalized


def metric_failures() -> list[str]:
    cases = [
        ("lang_vis.out.geo.aff_v3.retrieval_reranking", True, False),
        ("lang_vis.out.geo.aff_v3.rerank_top50", True, False),
        ("lang_vis.out.geo.aff_v3.phase", False, True),
        ("lang_vis.out.geo.aff_v3.action_chain", False, True),
        ("lang_vis.out.geo.aff.closed_loop", False, False),
    ]
    failures = []
    for metric, expected_rerank, expected_phase in cases:
        actual_rerank = is_retrieval_reranking_metric(metric)
        actual_phase = is_phase_metric(metric)
        if actual_rerank != expected_rerank or actual_phase != expected_phase:
            failures.append(
                f"{metric}: rerank={actual_rerank}, phase={actual_phase}; "
                f"expected rerank={expected_rerank}, phase={expected_phase}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    failures: list[str] = []
    for label, rel_path, needles in CHECKS:
        path = repo_root / rel_path
        if not path.exists():
            failures.append(f"{label}: missing file {rel_path}")
            continue
        text = path.read_text(errors="replace")
        missing = [needle for needle in needles if needle not in text]
        if missing:
            failures.append(f"{label}: missing {missing!r} in {rel_path}")
    failures.extend(metric_failures())

    if failures:
        print("Static rerank-checkpoint verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Static rerank-checkpoint verification passed.")
    print(f"Checked {len(CHECKS)} code/documentation invariants and metric semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
