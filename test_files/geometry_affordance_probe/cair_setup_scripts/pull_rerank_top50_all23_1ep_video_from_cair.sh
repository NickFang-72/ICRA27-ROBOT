#!/usr/bin/env bash
set -Eeuo pipefail

# Pull the latest rerank top-50 all-task video smoke from CAIR into the local
# review folder. Pass REMOTE_COMPONENT_ROOT to pull a specific run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=1
)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

if [[ -z "$REMOTE_COMPONENT_ROOT" ]]; then
  REMOTE_COMPONENT_ROOT="$(
    ssh "${SSH_OPTS[@]}" "$CAIR_HOST" \
      "ls -td '$REMOTE_PROJECT_ROOT'/experiments/rerank_top50_all23_1ep_video_* 2>/dev/null | head -1"
  )"
fi

if [[ -z "$REMOTE_COMPONENT_ROOT" ]]; then
  echo "No remote rerank_top50_all23_1ep_video_* run found." >&2
  exit 1
fi

RUN_NAME="$(basename "$REMOTE_COMPONENT_ROOT")"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-$REPO_ROOT/test_files/geometry_affordance_probe/review/$RUN_NAME}"
REMOTE_PROGRESS_JSON="${REMOTE_PROGRESS_JSON:-$REMOTE_COMPONENT_ROOT/progress_rerank_top50_all23_1ep_video.json}"

mkdir -p "$LOCAL_OUTPUT_ROOT"
rsync -e "$RSYNC_SSH" -a "$CAIR_HOST:$REMOTE_PROGRESS_JSON" "$LOCAL_OUTPUT_ROOT/progress_rerank_top50_all23_1ep_video.json"

read -r METHOD_NAME RUN_XICM_ROOT REVIEW_ROOT < <(
  python3 - "$LOCAL_OUTPUT_ROOT/progress_rerank_top50_all23_1ep_video.json" <<'PY'
import json
import sys
from pathlib import Path

progress = json.loads(Path(sys.argv[1]).read_text())
print(progress.get("method_name", ""), progress.get("run_xicm_root", ""), progress.get("review_root", ""))
PY
)

if [[ -z "$REVIEW_ROOT" ]]; then
  REVIEW_ROOT="$REMOTE_COMPONENT_ROOT/review"
fi

mkdir -p "$LOCAL_OUTPUT_ROOT/runner_logs" "$LOCAL_OUTPUT_ROOT/method_logs"

rsync -e "$RSYNC_SSH" -a "$CAIR_HOST:$REMOTE_COMPONENT_ROOT/runner_logs/" "$LOCAL_OUTPUT_ROOT/runner_logs/" || true
rsync -e "$RSYNC_SSH" -a "$CAIR_HOST:$REVIEW_ROOT/" "$LOCAL_OUTPUT_ROOT/" || true

if [[ -n "$METHOD_NAME" && -n "$RUN_XICM_ROOT" ]]; then
  mkdir -p "$LOCAL_OUTPUT_ROOT/method_logs/$METHOD_NAME"
  rsync -e "$RSYNC_SSH" -a \
    --include '*/' \
    --include 'test_data.csv' \
    --exclude '*' \
    "$CAIR_HOST:$RUN_XICM_ROOT/logs/$METHOD_NAME/" \
    "$LOCAL_OUTPUT_ROOT/method_logs/$METHOD_NAME/" || true
fi

python3 - "$LOCAL_OUTPUT_ROOT" <<'PY'
from pathlib import Path
import csv
import json
import re
import sys

root = Path(sys.argv[1])
progress_path = root / "progress_rerank_top50_all23_1ep_video.json"
progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
videos = sorted((root / "runtime_videos").glob("*.mp4"))
tasks = progress.get("tasks", [])
if isinstance(tasks, str):
    tasks = [item.strip() for item in tasks.split(",") if item.strip()]
method_name = progress.get("method_name", "")
seed = str(progress.get("seed", 0))
method_root = root / "method_logs" / method_name
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*([^\r\n]+)")
step_score_re = re.compile(r"Score:\s*([-+]?\d+(?:\.\d+)?)")
rows = []
for task in tasks:
    csv_path = method_root / task / f"seed{seed}" / "test_data.csv"
    score = ""
    if csv_path.exists():
        text = csv_path.read_text(errors="replace")
        match = finish_re.search(text)
        if match:
            final = match.group(1).strip()
            if final.lower() == "unknown":
                step_scores = step_score_re.findall(text)
                score = step_scores[-1] if step_scores else "unknown"
            else:
                score = final
    task_videos = sorted((root / "runtime_videos").glob(f"{task}_seed{seed}_*.mp4"))
    rows.append({
        "task": task,
        "seed": seed,
        "score": score,
        "video_file": task_videos[-1].name if task_videos else "",
        "test_data_csv": str(csv_path) if csv_path.exists() else "",
    })
if rows:
    with (root / "all23_scores_and_videos.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", "score", "video_file", "test_data_csv"])
        writer.writeheader()
        writer.writerows(rows)
    progress["finished_task_count"] = sum(1 for row in rows if row["test_data_csv"])
    progress["video_count"] = len(videos)
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
print()
print(f"Local video review folder: {root}")
print(f"Progress status: {progress.get('status', 'unknown')}")
print(f"Finished tasks: {progress.get('finished_task_count', 0)}/{progress.get('total_task_count', len(progress.get('tasks', [])))}")
print(f"Videos pulled: {len(videos)}")
print(f"Scores CSV: {root / 'all23_scores_and_videos.csv'}")
if videos:
    print("First videos:")
    for path in videos[:5]:
        print(f"  {path}")
PY
