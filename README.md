# Geometry- and Affordance-Augmented X-ICM

This repository is Nicholas's working ICRA 2027 robot manipulation project built
on top of **AGNOSTOS** and **X-ICM**.

The current checkpoint is the **double-retrieval rerank** version:

1. Run the original X-ICM dynamic diffusion retrieval broadly.
2. Keep the top `XICM_GA_RERANK_CANDIDATES` candidates, default `50`.
3. Rerank that shortlist with compact geometry, target-pose/profile
   compatibility, and conflict penalties.
4. Prompt QwenVL with the selected seen demonstrations, current query images,
   compact query descriptors, and query-only role-labeled contact hints.
5. Save review packets, result CSVs, and videos as local/CAIR artifacts, not as
   Git-tracked source.

Phase/action-chain retrieval and the older closed-loop ablation scripts are
parked as future ideas. They are intentionally not part of the current push
surface.

## Active Files

The code and helper scripts that define the current checkpoint are:

```text
X-ICM/form_icl_demonstrations_crosstask_ranking.py
X-ICM/crosstask_icl_agent.py
X-ICM/scripts/generate_rerank_review_trace.py
test_files/geometry_affordance_probe/retrieval_reranking_model_freeze.md
test_files/geometry_affordance_probe/README.md
test_files/geometry_affordance_probe/SCRIPT_INDEX.md
test_files/geometry_affordance_probe/scripts/verify_rerank_checkpoint_static.py
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh
test_files/geometry_affordance_probe/cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh
```

The detailed script map lives in:

```text
test_files/geometry_affordance_probe/SCRIPT_INDEX.md
```

## Main Commands

Run the k4/k8 quick comparison on CAIR:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_k4_k8_5ep_on_cair.sh
```

Watch that comparison from local:

```bash
ONCE=1 bash test_files/geometry_affordance_probe/cair_setup_scripts/watch_rerank_top50_k4_k8_5ep_from_local.sh
```

Run one episode per unseen task with video recording:

```bash
K=8 EPISODES=1 ENABLE_VIDEO=1 RECORD_EVERY_N=1 \
bash test_files/geometry_affordance_probe/cair_setup_scripts/run_rerank_top50_all23_1ep_video_on_cair.sh
```

Pull the all-task video smoke artifacts to the local review folder:

```bash
bash test_files/geometry_affordance_probe/cair_setup_scripts/pull_rerank_top50_all23_1ep_video_from_cair.sh
```

Run the local static verifier:

```bash
python3 test_files/geometry_affordance_probe/scripts/verify_rerank_checkpoint_static.py
```

## GitHub Push Policy

Git should contain source code, launch scripts, and documentation only.
Generated outputs stay local or on CAIR and are ignored by `.gitignore`.

Local-only paths include:

```text
results/
outputs/
test_files/geometry_affordance_probe/ablation_results/
test_files/geometry_affordance_probe/batch_*/
test_files/geometry_affordance_probe/review/
test_files/geometry_affordance_probe/figures/
test_files/geometry_affordance_probe/live_view/
test_files/xicm_baseline_results/
```

Runtime media (`*.mp4`, `*.avi`, `*.mov`, `*.webm`) and generated inspection
sidecars (`*.inspect.ndjson`) should also remain untracked.

## References

- **AGNOSTOS / X-ICM paper**: [Exploring the Limits of Vision-Language-Action Manipulation in Cross-task Generalization](https://arxiv.org/pdf/2505.15660)
- **AGNOSTOS project page**: [jiaming-zhou.github.io/AGNOSTOS](https://jiaming-zhou.github.io/AGNOSTOS/)
- **X-ICM GitHub repository**: [jiaming-zhou/X-ICM](https://github.com/jiaming-zhou/X-ICM)
- **AGNOSTOS dataset**: [Hugging Face dataset](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)
- **X-ICM model**: [Hugging Face model](https://huggingface.co/Jiaming2472/X-ICM)
