# Geometry/Affordance Script Index

This is the cleaned map for the current double-retrieval rerank checkpoint. It
intentionally excludes parked phase/action-chain scripts and generated review
artifacts from the Git surface.

## Current Double-Retrieval Files

| File | Purpose |
|---|---|
| `X-ICM/form_icl_demonstrations_crosstask_ranking.py` | Implements broad dynamic retrieval, top-50 shortlist selection, fine geometry/profile rerank, and final prompt construction. |
| `X-ICM/crosstask_icl_agent.py` | Builds QwenVL messages with current query images and parses the returned 7D key actions. |
| `X-ICM/scripts/generate_rerank_review_trace.py` | Writes inspectable query extraction, broad retrieval, fine rerank, selected demos, final prompt, and contact-overlay artifacts. |
| `scripts/verify_rerank_checkpoint_static.py` | Fast local verifier for the frozen rerank checkpoint, aliases, docs, and runner defaults. |
| `retrieval_reranking_model_freeze.md` | Human-readable freeze note for the active model. |

## Current CAIR Helpers

| Script | Purpose |
|---|---|
| `cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh` | Runs the active rerank checkpoint for k4 and k8 quick comparison. |
| `cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh` | Watches the k4/k8 run from the local machine. |
| `cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh` | Runs one episode per unseen task with video recording enabled. |
| `cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh` | Pulls the video smoke review folder, MP4s, logs, and score CSV to local. |

## Remaining Local Helper

Only `scripts/verify_rerank_checkpoint_static.py` remains in the probe helper
folder. Older descriptor-cache builders, prompt renderers, Robopoint projection
checks, closed-loop ablation collectors, and tuning utilities were removed from
the active tree because the current method gets its descriptors and final prompt
through the X-ICM rerank code path above.

## Ignored Local Work

The parked phase/action-chain branch is intentionally ignored by `.gitignore`.
Generated folders such as `results/`, `outputs/`, `review/`, `batch_*`, logs,
runtime videos, and presentation inspection sidecars are also local-only.

Before pushing, `git status --short --untracked-files=all` should show only
source/docs/scripts for the current rerank checkpoint, not result folders or
runtime artifacts.
