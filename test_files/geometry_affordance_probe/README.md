# Geometry/Affordance Probe

This folder is now organized around the current double-retrieval rerank checkpoint.
Generated result folders, review packets, CAIR logs, videos, local cache
batches, and parked phase/action-chain files are ignored by Git.

## Active Checkpoint

Use either stable ranking name:

```text
lang_vis.out.geo.aff_v3.retrieval_reranking
lang_vis.out.geo.aff_v3.rerank_top50
```

The active pipeline is:

1. Broad retrieval with the original X-ICM dynamic diffusion similarity.
2. Keep the top `XICM_GA_RERANK_CANDIDATES` candidates, default `50`.
3. Fine rerank that shortlist with geometry, target-pose/profile compatibility,
   and conflict penalties.
4. Build the final QwenVL prompt from selected seen demos, current front plus
   overhead query images, compact descriptors, and query-only role-labeled
   contact hints `c_j`.

Seen-demo contact hints `c_i` stay internal. Plan-guided retrieval, closed-loop
ablations, and phase/action-chain retrieval are not part of this checkpoint.

## Files To Commit

The current double-retrieval push surface is:

```text
X-ICM/form_icl_demonstrations_crosstask_ranking.py
X-ICM/crosstask_icl_agent.py
X-ICM/scripts/generate_rerank_review_trace.py
test_files/geometry_affordance_probe/retrieval_reranking_model_freeze.md
test_files/geometry_affordance_probe/SCRIPT_INDEX.md
test_files/geometry_affordance_probe/README.md
test_files/geometry_affordance_probe/scripts/verify_rerank_checkpoint_static.py
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh
```

## Main Commands

Generate a five-task review packet on CAIR:

```bash
cd /data/yf23/projects/ICRA27-ROBOT/X-ICM
export XICM_GA_RERANK_CANDIDATES=50
export XICM_GA_REVIEW_BUNDLE=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl
python3 scripts/generate_rerank_review_trace.py \
  --name rerank_top50_k5_pipeline_review \
  --count 5 \
  --top-k 5 \
  --rerank-candidates 50 \
  --ranking-metric lang_vis.out.geo.aff_v3.rerank_top50
```

Run the frozen k4/k8 quick comparison on CAIR:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh
```

Watch that run from local:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
```

Run the one-episode all-task video smoke:

```bash
K=8 EPISODES=1 ENABLE_VIDEO=1 RECORD_EVERY_N=1 \
bash test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh
```

Pull the latest all-task video smoke to the local review folder:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh
```

Run the fast local static verifier:

```bash
python3 test_files/geometry_affordance_probe/scripts/verify_rerank_checkpoint_static.py
```

## Local-Only Outputs

Do not commit these paths:

```text
results/
outputs/
test_files/geometry_affordance_probe/ablation_results/
test_files/geometry_affordance_probe/batch_*/
test_files/geometry_affordance_probe/review/
test_files/geometry_affordance_probe/figures/
test_files/geometry_affordance_probe/live_view/
```

Do not commit runtime media (`*.mp4`, `*.avi`, `*.mov`, `*.webm`) or generated
PowerPoint inspection sidecars (`*.inspect.ndjson`). These stay local or on
CAIR and can be regenerated or pulled when needed.
