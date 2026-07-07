# Retrieval Reranking Model Freeze

This file freezes the active retrieval reranking checkpoint. Phase/action-chain
work is parked as a future idea; use this path for current experiments.

## Stable Name

Use either ranking method name:

```text
lang_vis.out.geo.aff_v3.retrieval_reranking
lang_vis.out.geo.aff_v3.rerank_top50
```

Both names route through the same code path as the tested reranker:

1. Broad retrieval with X-ICM dynamic diffusion similarity.
2. Keep the top `XICM_GA_RERANK_CANDIDATES`, default `50`.
3. Fine rerank inside that shortlist using geometry, target-pose/profile, and
   conflict penalties.
4. Query contact points `c_j` are prompt hints only; seen contact points `c_i`
   stay internal and are not copied into the final LLM prompt.

## Core Files

```text
X-ICM/form_icl_demonstrations_crosstask_ranking.py
X-ICM/crosstask_icl_agent.py
X-ICM/scripts/generate_rerank_review_trace.py
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh
```

## Latest Local Results

```text
results/rerank_top50_k4_k8_5eps_20260701/rerank_top50_vs_baselines_k4_k8_wide_best_marked.csv
results/rerank_top50_k4_k8_5eps_20260701/rerank_top50_vs_baselines_k4_k8_overall.csv
```

Summary:

```csv
k,condition,mean_final_score,successes_out_of_115
4,rerank_top50_geo_contact_qwenvl,16.521739,19
8,rerank_top50_geo_contact_qwenvl,22.608696,26
```

## CAIR Environment

```bash
export XICM_GA_RERANK_CANDIDATES=50
export XICM_GA_REVIEW_BUNDLE=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl
```

This model is now the active retrieval-reranking path. Future phase experiments
must stay on separate ranking methods containing `.phase` or `action_chain` so
they cannot silently replace the broad-then-fine checkpoint.

## Git Surface

The current push surface is the double-retrieval code, rerank review generator,
rerank CAIR helpers, verifier, and these docs. Generated results, local review
packets, runtime videos, presentation exports, and parked phase/action-chain
scripts are ignored by `.gitignore`.
