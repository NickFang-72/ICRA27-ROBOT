#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
XICM_ROOT="${XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
PROGRESS_JSON="$RUN_ROOT/progress.json"

if [[ -f "$PROGRESS_JSON" ]]; then
    echo "progress.json:"
    cat "$PROGRESS_JSON"
else
    echo "progress.json is missing at $PROGRESS_JSON"
fi

echo
echo "matching processes:"
pgrep -af "run_geometry_affordance_ablations_on_cair|scripts/eval_XICM.sh|python main.py" || true

echo
echo "completed task CSV counts:"
for ranking in lang_vis.out.geo lang_vis.out.aff lang_vis.out.geo_aff; do
    method="XICM_Cross.ZS_Ranking.${ranking}_Qwen2.5.7B.instruct_icl.18_test"
    method_dir="$XICM_ROOT/logs/$method"
    if [[ -d "$method_dir" ]]; then
        file_count=$(find "$method_dir" -path "*/seed0/test_data.csv" -type f 2>/dev/null | wc -l | tr -d ' ')
        final_count=0
        while IFS= read -r score_file; do
            if grep -Eq "Finished .*Final Score: [0-9]+(\\.[0-9]+)?" "$score_file"; then
                final_count=$((final_count + 1))
            fi
        done < <(find "$method_dir" -path "*/seed0/test_data.csv" -type f 2>/dev/null)
    else
        file_count=0
        final_count=0
    fi
    printf "%-24s %s/23 final, %s/23 files\n" "$ranking" "${final_count:-0}" "${file_count:-0}"
done

echo
echo "recent ablation log tail:"
latest_log=""
if [[ -f "$PROGRESS_JSON" ]]; then
    latest_log=$(python - "$PROGRESS_JSON" <<'PY' 2>/dev/null || true
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle).get("log_path", ""))
PY
)
fi
if [[ -z "${latest_log:-}" || ! -f "$latest_log" ]]; then
    latest_log=$(find "$RUN_ROOT/logs" -type f -name "*.log" -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1 || true)
fi
if [[ -n "${latest_log:-}" ]]; then
    echo "$latest_log"
    echo
    echo "latest episode lines:"
    grep -E "Evaluating .* Episode|Finished .* Final Score" "$latest_log" | tail -20 || true
    echo
    echo "raw tail:"
    tail -40 "$latest_log"
else
    echo "No ablation log found yet."
fi
