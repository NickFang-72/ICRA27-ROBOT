#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${RUN_ROOT:-/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_ablations}"
XICM_ROOT="${XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
PROGRESS_JSON="$RUN_ROOT/progress.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
for ranking in lang_vis.out.geo lang_vis.out.aff lang_vis.out.geo_aff lang_vis.out.geo_aff_v2; do
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
    latest_log=$("$PYTHON_BIN" - "$PROGRESS_JSON" <<'PY' 2>/dev/null || true
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
    echo "current task summary:"
    "$PYTHON_BIN" - "$latest_log" <<'PY' 2>/dev/null || true
import re
import sys

log_path = sys.argv[1]
eval_re = re.compile(
    r"Evaluating (?P<task>[^|]+?) \| Episode (?P<episode>\d+) \| Step: (?P<step>\d+) "
    r"\| Score: (?P<score>[0-9]+(?:\.[0-9]+)?) \| Lang Goal: (?P<lang_goal>.*)$"
)
finish_re = re.compile(
    r"Finished (?P<task>[^|]+?) \| Final Score: (?P<score>[0-9]+(?:\.[0-9]+)?)"
)

last_eval = None
last_finish = None
finished = []
with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        eval_match = eval_re.search(line.strip())
        if eval_match:
            last_eval = eval_match.groupdict()
        finish_match = finish_re.search(line.strip())
        if finish_match:
            last_finish = finish_match.groupdict()
            if not finished or finished[-1] != last_finish:
                finished.append(last_finish)

if last_eval:
    episode = int(last_eval["episode"])
    print(
        f"- current_task: {last_eval['task'].strip()} "
        f"(episode {min(episode + 1, 25)}/25, last_step {last_eval['step']}, "
        f"last_score {last_eval['score']})"
    )
    print(f"- lang_goal: {last_eval['lang_goal']}")
elif last_finish:
    print(f"- last_event: finished {last_finish['task'].strip()} with score {last_finish['score']}")
else:
    print("- No evaluation line found yet.")

if finished:
    print("- recent_finished:")
    for item in finished[-5:]:
        print(f"  {item['task'].strip()}: {item['score']}")
PY
    echo
    echo "latest episode lines:"
    grep -E "Evaluating .* Episode|Finished .* Final Score" "$latest_log" | tail -20 || true
    echo
    echo "raw tail:"
    tail -40 "$latest_log"
else
    echo "No ablation log found yet."
fi
