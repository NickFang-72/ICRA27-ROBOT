# Test Files

This folder contains experiment code, launch scripts, lightweight fixtures, and
local-only generated artifacts for the AGNOSTOS/X-ICM geometry-affordance probe.

## Current Folder

- `geometry_affordance_probe/`: active closed-loop no-plan geometry/contact
  ablation code, CAIR launchers, fixtures, and local-only result pulls.

Start here when looking for scripts:

- `geometry_affordance_probe/SCRIPT_INDEX.md`

## Git Tracking Policy

Track code, launch scripts, small fixtures, and documentation. Do not track
generated experiment outputs.

Ignored local artifacts include:

- `geometry_affordance_probe/batch_*/`
- `geometry_affordance_probe/ablation_results/`
- `geometry_affordance_probe/review/`
- `geometry_affordance_probe/figures/`
- `xicm_baseline_results/`

Those folders can contain rendered observations, Qwen/RoboPoint review JSON,
CAIR logs, result CSVs, figures, and benchmark summaries. They stay on this
machine or on CAIR and should be regenerated or pulled when needed.

## Active Runs

The current comparison line is:

- baseline Qwen text-only versus QwenVL front+overhead
- closed-loop no-plan geometry retrieval
- closed-loop no-plan geometry retrieval plus contact-point prompt hints

Closed-loop no-plan k-sweep results are pulled into:

```text
geometry_affordance_probe/ablation_results/closed_loop_no_plan_k4_k6_k8_k10_comparison_2026-06-30.csv
```

Older v1-v4 launchers and one-off review/collector scripts were moved to:

```text
geometry_affordance_probe/legacy/
```

