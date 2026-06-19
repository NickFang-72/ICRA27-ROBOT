#!/usr/bin/env bash
set -Eeuo pipefail

# Pull completed CAIR ablation result folders into the local workspace.

REMOTE_HOST="${REMOTE_HOST:-cair}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM/logs}"
LOCAL_ROOT="${LOCAL_ROOT:-/Users/nicholas/Documents/ICRA27 ROBOT/test_files/geometry_affordance_probe/ablation_results/cair_logs}"

methods=(
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_Qwen2.5.7B.instruct_icl.18_test"
    "XICM_Cross.ZS_Ranking.lang_vis.out.aff_Qwen2.5.7B.instruct_icl.18_test"
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_Qwen2.5.7B.instruct_icl.18_test"
)

mkdir -p "$LOCAL_ROOT"

for method in "${methods[@]}"; do
    echo "Pulling $method"
    rsync -a "$REMOTE_HOST:$REMOTE_LOG_ROOT/$method/" "$LOCAL_ROOT/$method/"
done

echo "Local ablation logs are under $LOCAL_ROOT"
