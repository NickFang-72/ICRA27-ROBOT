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
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.6_test"
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.8_test"
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v2_Qwen2.5.7B.instruct_icl.10_test"
    "XICM_Cross.ZS_Ranking.lang_vis.out.geo_aff_v3_Qwen2.5.7B.instruct_icl.6_test"
)

mkdir -p "$LOCAL_ROOT"

pulled_count=0
skipped_count=0

for method in "${methods[@]}"; do
    remote_method_dir="$REMOTE_LOG_ROOT/$method"
    local_method_dir="$LOCAL_ROOT/$method"

    if ssh "$REMOTE_HOST" "test -d '$remote_method_dir'"; then
        echo "Pulling $method"
        mkdir -p "$local_method_dir"
        rsync -a "$REMOTE_HOST:$remote_method_dir/" "$local_method_dir/"
        pulled_count=$((pulled_count + 1))
    else
        echo "Skipping missing $method"
        skipped_count=$((skipped_count + 1))
    fi
done

echo "Local ablation logs are under $LOCAL_ROOT"
echo "Pulled $pulled_count method folder(s), skipped $skipped_count missing folder(s)."
