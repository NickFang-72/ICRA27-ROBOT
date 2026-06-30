#!/usr/bin/env bash
set -Eeuo pipefail

# Generic CAIR launcher for Qwen/QwenVL X-ICM ablation matrices.
#
# Wrapper scripts set METHOD_CONFIG_TEXT and the run/cache paths for a specific
# experiment. When run directly, the defaults still mirror the older v1 component
# ablation, so prefer calling a wrapper such as:
#
#   run_xicm_qwenvl_closed_loop_no_plan_5ep_ablation_on_cair.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_COMPONENT_ROOT="${REMOTE_COMPONENT_ROOT:-$REMOTE_PROJECT_ROOT/experiments/v1_component_10eps_3seed_20260625}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_COMPONENT_ROOT/X-ICM_v1_component}"
REMOTE_RUNNER_LOG_ROOT="${REMOTE_RUNNER_LOG_ROOT:-$REMOTE_COMPONENT_ROOT/runner_logs}"
REMOTE_CLEAN_GEOMETRY_CACHE="${REMOTE_CLEAN_GEOMETRY_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_v1_primitive_full_cache_20260623/review_bundle.jsonl}"
REMOTE_OLD_CONTACT_CACHE="${REMOTE_OLD_CONTACT_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"
REMOTE_CONTACT_CACHE_DIR="${REMOTE_CONTACT_CACHE_DIR:-$REMOTE_COMPONENT_ROOT/merged_v1_geometry_with_contact_hints}"
REMOTE_CONTACT_CACHE="${REMOTE_CONTACT_CACHE:-$REMOTE_CONTACT_CACHE_DIR/review_bundle.jsonl}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0,50,99}"
EPISODES="${EPISODES:-10}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
GPU_ID="${GPU_ID:-0}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"
RESET_RUN_TREE="${RESET_RUN_TREE:-0}"
FORCE_REBUILD_CONTACT_CACHE="${FORCE_REBUILD_CONTACT_CACHE:-0}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-15}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
METHOD_CONFIG_TEXT="${METHOD_CONFIG_TEXT:-}"
XICM_GA_DELTA="${XICM_GA_DELTA:-0.25}"
XICM_GA_PENALTY="${XICM_GA_PENALTY:-0.55}"
XICM_GA_PLAN_WEIGHT="${XICM_GA_PLAN_WEIGHT:-0.45}"
XICM_GA_PLAN_BLOCK_CAP="${XICM_GA_PLAN_BLOCK_CAP:-0.15}"
XICM_GA_PLAN_WEAK_CAP="${XICM_GA_PLAN_WEAK_CAP:-0.55}"
XICM_GA_PLAN_UNKNOWN_CAP="${XICM_GA_PLAN_UNKNOWN_CAP:-0.45}"
XICM_GA_MAX_PER_TASK="${XICM_GA_MAX_PER_TASK:-2}"
XICM_GA_MAX_PER_FAMILY="${XICM_GA_MAX_PER_FAMILY:-3}"
XICM_CLOSED_LOOP_MAX_REPLANS="${XICM_CLOSED_LOOP_MAX_REPLANS:-4}"
XICM_TASKS_OVERRIDE="${XICM_TASKS_OVERRIDE:-}"

METHOD_CONFIG_TEXT_B64=""
if [[ -n "$METHOD_CONFIG_TEXT" ]]; then
  METHOD_CONFIG_TEXT_B64="$(printf "%s" "$METHOD_CONFIG_TEXT" | base64 | tr -d '\n')"
fi

LOCAL_FORM="$REPO_ROOT/X-ICM/form_icl_demonstrations_crosstask_ranking.py"
LOCAL_AGENT="$REPO_ROOT/X-ICM/crosstask_icl_agent.py"
LOCAL_MAIN="$REPO_ROOT/X-ICM/main.py"
LOCAL_EVAL_SCRIPT="$REPO_ROOT/X-ICM/scripts/eval_XICM.sh"
XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"

ssh "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' RESET_RUN_TREE='$RESET_RUN_TREE' bash -s" <<'REMOTE_SETUP'
set -Eeuo pipefail

mkdir -p "$REMOTE_COMPONENT_ROOT" "$REMOTE_RUNNER_LOG_ROOT"
if [[ "$RESET_RUN_TREE" == "1" && -d "$REMOTE_RUN_XICM" ]]; then
  rm -rf "$REMOTE_RUN_XICM"
fi
if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  echo "Creating isolated v1 component X-ICM tree at $REMOTE_RUN_XICM"
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"
fi
mkdir -p "$REMOTE_RUN_XICM/logs"
REMOTE_SETUP

rsync -a \
  "$LOCAL_MAIN" \
  "$LOCAL_FORM" \
  "$LOCAL_AGENT" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/"
rsync -a \
  "$LOCAL_EVAL_SCRIPT" \
  "$CAIR_HOST:$REMOTE_RUN_XICM/scripts/eval_XICM.sh"

ssh "$CAIR_HOST" \
  "REMOTE_COMPONENT_ROOT='$REMOTE_COMPONENT_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_RUNNER_LOG_ROOT='$REMOTE_RUNNER_LOG_ROOT' REMOTE_CLEAN_GEOMETRY_CACHE='$REMOTE_CLEAN_GEOMETRY_CACHE' REMOTE_OLD_CONTACT_CACHE='$REMOTE_OLD_CONTACT_CACHE' REMOTE_CONTACT_CACHE_DIR='$REMOTE_CONTACT_CACHE_DIR' REMOTE_CONTACT_CACHE='$REMOTE_CONTACT_CACHE' CONDA_ENV='$CONDA_ENV' SEEDS='$SEEDS' EPISODES='$EPISODES' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' GPU_ID='$GPU_ID' TOTAL_TASKS='$TOTAL_TASKS' FORCE_REBUILD_CONTACT_CACHE='$FORCE_REBUILD_CONTACT_CACHE' WAIT_FOR_GPU='$WAIT_FOR_GPU' MIN_FREE_GPU_MEMORY_MB='$MIN_FREE_GPU_MEMORY_MB' MAX_GPU_UTIL_PERCENT='$MAX_GPU_UTIL_PERCENT' GPU_WAIT_INTERVAL_SECONDS='$GPU_WAIT_INTERVAL_SECONDS' XICM_QWEN25_VL_7B_PATH='$XICM_QWEN25_VL_7B_PATH' XICM_VLLM_GPU_MEMORY_UTILIZATION='$XICM_VLLM_GPU_MEMORY_UTILIZATION' XICM_VLLM_MAX_MODEL_LEN='$XICM_VLLM_MAX_MODEL_LEN' XICM_VL_MAX_IMAGES='$XICM_VL_MAX_IMAGES' METHOD_CONFIG_TEXT_B64='$METHOD_CONFIG_TEXT_B64' XICM_GA_DELTA='$XICM_GA_DELTA' XICM_GA_PENALTY='$XICM_GA_PENALTY' XICM_GA_PLAN_WEIGHT='$XICM_GA_PLAN_WEIGHT' XICM_GA_PLAN_BLOCK_CAP='$XICM_GA_PLAN_BLOCK_CAP' XICM_GA_PLAN_WEAK_CAP='$XICM_GA_PLAN_WEAK_CAP' XICM_GA_PLAN_UNKNOWN_CAP='$XICM_GA_PLAN_UNKNOWN_CAP' XICM_GA_MAX_PER_TASK='$XICM_GA_MAX_PER_TASK' XICM_GA_MAX_PER_FAMILY='$XICM_GA_MAX_PER_FAMILY' XICM_CLOSED_LOOP_MAX_REPLANS='$XICM_CLOSED_LOOP_MAX_REPLANS' XICM_TASKS_OVERRIDE='$XICM_TASKS_OVERRIDE' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

METHOD_CONFIG_FILE="$REMOTE_COMPONENT_ROOT/v1_10ep_component_methods.tsv"
PROGRESS_JSON="$REMOTE_COMPONENT_ROOT/progress_v1_10ep_component.json"
RUN_SCRIPT="$REMOTE_COMPONENT_ROOT/run_v1_10ep_component_ablation.sh"
PID_FILE="$REMOTE_COMPONENT_ROOT/v1_10ep_component_ablation.pid"

if [[ -n "${METHOD_CONFIG_TEXT_B64:-}" ]]; then
  printf "%s" "$METHOD_CONFIG_TEXT_B64" | base64 -d > "$METHOD_CONFIG_FILE"
  printf "\n" >> "$METHOD_CONFIG_FILE"
else
  cat > "$METHOD_CONFIG_FILE" <<'CONFIG'
v1_geo_target_pose_10eps_3seed|lang_vis.out.geo|v1_geo_target_pose|clean|0.70|0.30|0.0
v1_contact_points_10eps_3seed|lang_vis.out.aff|v1_contact_points|contact|1.00|0.00|0.0
v1_everything_10eps_3seed|lang_vis.out.geo.aff|v1_everything|contact|0.70|0.30|0.0
CONFIG
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
TOTAL_SEED_TASKS=$(( ${#SEED_LIST[@]} * TOTAL_TASKS ))
TOTAL_METHODS="$(wc -l < "$METHOD_CONFIG_FILE" | tr -d '[:space:]')"
TOTAL_ALL_SEED_TASKS=$(( TOTAL_METHODS * TOTAL_SEED_TASKS ))

method_name_for_ranking() {
  local ranking="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$ranking" "$MODEL_NAME" "$DEMO_NUM_PER_ICL"
}

count_method_completed() {
  local method="$1"
  METHOD="$method" REMOTE_RUN_XICM="$REMOTE_RUN_XICM" SEEDS="$SEEDS" python3 - <<'PY'
from pathlib import Path
import os
import re

method_dir = Path(os.environ["REMOTE_RUN_XICM"]) / "logs" / os.environ["METHOD"]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
count = 0
if method_dir.exists():
    for seed in seeds:
        for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
            if finish_re.search(path.read_text(errors="replace")):
                count += 1
print(count)
PY
}

write_progress() {
  local status="$1"
  local active_run_id="$2"
  local active_ranking="$3"
  local log_path="$4"
  local message="$5"
  STATUS="$status" ACTIVE_RUN_ID="$active_run_id" ACTIVE_RANKING="$active_ranking" ACTIVE_LOG="$log_path" MESSAGE="$message" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" METHOD_CONFIG_FILE="$METHOD_CONFIG_FILE" \
  MODEL_NAME="$MODEL_NAME" DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" SEEDS="$SEEDS" EPISODES="$EPISODES" GPU_ID="$GPU_ID" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" MIN_FREE_GPU_MEMORY_MB="$MIN_FREE_GPU_MEMORY_MB" MAX_GPU_UTIL_PERCENT="$MAX_GPU_UTIL_PERCENT" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" TOTAL_ALL_SEED_TASKS="$TOTAL_ALL_SEED_TASKS" PROGRESS_JSON="$PROGRESS_JSON" \
  XICM_GA_DELTA="$XICM_GA_DELTA" XICM_GA_PENALTY="$XICM_GA_PENALTY" XICM_GA_PLAN_WEIGHT="$XICM_GA_PLAN_WEIGHT" \
  XICM_GA_PLAN_BLOCK_CAP="$XICM_GA_PLAN_BLOCK_CAP" XICM_GA_PLAN_WEAK_CAP="$XICM_GA_PLAN_WEAK_CAP" \
  XICM_GA_PLAN_UNKNOWN_CAP="$XICM_GA_PLAN_UNKNOWN_CAP" XICM_GA_MAX_PER_TASK="$XICM_GA_MAX_PER_TASK" \
  XICM_GA_MAX_PER_FAMILY="$XICM_GA_MAX_PER_FAMILY" XICM_CLOSED_LOOP_MAX_REPLANS="$XICM_CLOSED_LOOP_MAX_REPLANS" \
  XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

config_path = Path(os.environ["METHOD_CONFIG_FILE"])
run_root = Path(os.environ["REMOTE_RUN_XICM"])
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
per_method = {}
total_complete = 0
for line in config_path.read_text().splitlines():
    if not line.strip():
        continue
    run_id, ranking, condition, cache_kind, alpha, beta, gamma = line.split("|")[:7]
    uses_plan = "geo_plan" in ranking or ".plan" in ranking
    uses_closed_loop = "closed_loop" in ranking.replace("-", "_") or ".cl" in ranking
    method = f"XICM_Cross.ZS_Ranking.{ranking}_{os.environ['MODEL_NAME']}_icl.{os.environ['DEMO_NUM_PER_ICL']}_test"
    method_dir = run_root / "logs" / method
    count = 0
    per_seed = {}
    for seed in seeds:
        seed_count = 0
        if method_dir.exists():
            for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
                if finish_re.search(path.read_text(errors="replace")):
                    count += 1
                    seed_count += 1
        per_seed[f"seed{seed}"] = seed_count
    total_complete += count
    per_method[run_id] = {
        "ranking_method": ranking,
        "method": method,
        "condition": condition,
        "cache_kind": cache_kind,
        "strict_final_seed_task_count": count,
        "total_seed_task_count": int(os.environ["TOTAL_SEED_TASKS"]),
        "per_seed": per_seed,
        "retrieval_weights": {
            "alpha_dynamic": float(alpha),
            "beta_geometry": float(beta),
            "gamma_contact": float(gamma),
            "delta_profile": float(os.environ["XICM_GA_DELTA"]),
            "penalty_weight": float(os.environ["XICM_GA_PENALTY"]),
            "plan_weight": float(os.environ["XICM_GA_PLAN_WEIGHT"]) if uses_plan else 0.0,
            "plan_block_cap": float(os.environ["XICM_GA_PLAN_BLOCK_CAP"]),
            "plan_weak_cap": float(os.environ["XICM_GA_PLAN_WEAK_CAP"]),
            "plan_unknown_cap": float(os.environ["XICM_GA_PLAN_UNKNOWN_CAP"]),
        },
        "uses_geometry_retrieval": float(beta) > 0,
        "uses_target_pose_prompt": ".geo" in ranking,
        "uses_contact_prompt": ".aff" in ranking and "geo_aff" not in ranking,
        "uses_plan_guided_retrieval": uses_plan,
        "uses_closed_loop": uses_closed_loop,
        "closed_loop_max_replans": int(os.environ["XICM_CLOSED_LOOP_MAX_REPLANS"]) if uses_closed_loop else 0,
        "contact_points_in_retrieval": False,
        "target_pose_in_retrieval": uses_plan,
        "retrieval_diversity_caps": {
            "max_per_task": int(os.environ["XICM_GA_MAX_PER_TASK"]),
            "max_per_family": int(os.environ["XICM_GA_MAX_PER_FAMILY"]),
        },
    }
payload = {
    "status": os.environ["STATUS"],
    "condition": "xicm_v1_component_ablation_10eps_3seed",
    "active_run_id": os.environ["ACTIVE_RUN_ID"],
    "active_ranking_method": os.environ["ACTIVE_RANKING"],
    "active_log_path": os.environ["ACTIVE_LOG"],
    "seeds": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "demo_num_per_icl": int(os.environ["DEMO_NUM_PER_ICL"]),
    "tasks_override": os.environ.get("XICM_TASKS_OVERRIDE", ""),
    "gpu_id": os.environ["GPU_ID"],
    "gpu_wait": {
        "enabled": os.environ["WAIT_FOR_GPU"] in {"1", "true", "yes"},
        "min_free_memory_mb": int(os.environ["MIN_FREE_GPU_MEMORY_MB"]),
        "max_util_percent": int(os.environ["MAX_GPU_UTIL_PERCENT"]),
    },
    "per_method": per_method,
    "completed_seed_task_csvs": total_complete,
    "total_seed_task_csvs": int(os.environ["TOTAL_ALL_SEED_TASKS"]),
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "component_root": os.environ["REMOTE_COMPONENT_ROOT"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if [[ "$FORCE_REBUILD_CONTACT_CACHE" == "1" || ! -f "$REMOTE_CONTACT_CACHE" ]]; then
  mkdir -p "$REMOTE_CONTACT_CACHE_DIR"
  REMOTE_CLEAN_GEOMETRY_CACHE="$REMOTE_CLEAN_GEOMETRY_CACHE" \
  REMOTE_OLD_CONTACT_CACHE="$REMOTE_OLD_CONTACT_CACHE" \
  REMOTE_CONTACT_CACHE="$REMOTE_CONTACT_CACHE" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re

clean_path = Path(os.environ["REMOTE_CLEAN_GEOMETRY_CACHE"])
old_path = Path(os.environ["REMOTE_OLD_CONTACT_CACHE"])
out_path = Path(os.environ["REMOTE_CONTACT_CACHE"])

def label(value, default="unknown"):
    if isinstance(value, str) and value.strip():
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or default
    if isinstance(value, (list, tuple)):
        for item in value:
            got = label(item, "")
            if got:
                return got
    return default

def contact_mode(aff):
    text = " ".join(str(aff.get(key, "")) for key in ["grasp_affordance", "contact_affordance", "motion_affordance"]).lower()
    if "press" in text or "button" in text:
        return "press_point"
    if any(token in text for token in ["push", "surface", "sweep", "slide", "drag"]):
        return "single_contact"
    if any(token in text for token in ["grasp", "pinch", "pull", "twist", "lift", "place", "insert"]):
        return "grasp_pair"
    return "region_hint"

def norm_part(value):
    part = label(value)
    for key in ["handle", "knob", "rim", "edge", "slot", "hole", "button", "opening", "surface", "body", "plug", "spout", "socket", "lid"]:
        if key in part:
            return "button_top" if key == "button" else key
    return part

def points(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x, y = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            out.append([x, y])
    return out

old_rows = {}
with old_path.open() as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        old_rows[(row["task"], int(row["episode_id"]))] = row

count = 0
with clean_path.open() as src, out_path.open("w") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        old = old_rows.get((row["task"], int(row["episode_id"])), {})
        aff = old.get("affordance_a_i") or {}
        region = aff.get("required_contact_region") or row.get("geometry_g_i", {}).get("contact_region") or "unknown"
        row["contact_hints_i"] = {
            "contact_mode": contact_mode(aff),
            "source_view": "front_rgb_initial",
            "target_object": label(row.get("geometry_g_i", {}).get("manipulated_object") or row["task"]),
            "target_part": norm_part(region),
            "points_2d_normalized": points(aff.get("preferred_contact_points") or []),
            "contact_region_text": label(region),
            "candidate_contact_coordinates": [],
            "use_as": "seen_demo_contact_hint_only_not_retrieval",
            "source_note": "merged clean v1 geometry with legacy seen-demo RoboPoint-style contact hints",
        }
        dst.write(json.dumps(row) + "\n")
        count += 1
print(f"Wrote {count} merged contact-hint rows to {out_path}")
PY
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && ps -p "$old_pid" -o args= | grep -q "$RUN_SCRIPT"; then
    write_progress "already_running" "" "" "" "The v1 10-episode component ablation runner is already active."
    echo "v1 component ablation already running as PID $old_pid"
    exit 0
  fi
fi

if ps -eo pid=,args= | grep -F "$REMOTE_COMPONENT_ROOT" | grep -E 'run_v1_10ep_component_ablation|eval_XICM.sh|main.py' | grep -v grep >/dev/null 2>&1; then
  write_progress "already_running" "" "" "" "A v1 10-episode component ablation process already appears to be running."
  echo "v1 component ablation already appears to be running."
  exit 0
fi

cat > "$RUN_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail

count_method_completed() {
  local method="$1"
  METHOD="$method" REMOTE_RUN_XICM="$REMOTE_RUN_XICM" SEEDS="$SEEDS" python3 - <<'PY'
from pathlib import Path
import os
import re

method_dir = Path(os.environ["REMOTE_RUN_XICM"]) / "logs" / os.environ["METHOD"]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
count = 0
if method_dir.exists():
    for seed in seeds:
        for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
            if finish_re.search(path.read_text(errors="replace")):
                count += 1
print(count)
PY
}

write_progress() {
  local status="$1"
  local active_run_id="$2"
  local active_ranking="$3"
  local log_path="$4"
  local message="$5"
  STATUS="$status" ACTIVE_RUN_ID="$active_run_id" ACTIVE_RANKING="$active_ranking" ACTIVE_LOG="$log_path" MESSAGE="$message" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" METHOD_CONFIG_FILE="$METHOD_CONFIG_FILE" \
  MODEL_NAME="$MODEL_NAME" DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" SEEDS="$SEEDS" EPISODES="$EPISODES" GPU_ID="$GPU_ID" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" MIN_FREE_GPU_MEMORY_MB="$MIN_FREE_GPU_MEMORY_MB" MAX_GPU_UTIL_PERCENT="$MAX_GPU_UTIL_PERCENT" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" TOTAL_ALL_SEED_TASKS="$TOTAL_ALL_SEED_TASKS" PROGRESS_JSON="$PROGRESS_JSON" \
  XICM_GA_DELTA="$XICM_GA_DELTA" XICM_GA_PENALTY="$XICM_GA_PENALTY" XICM_GA_PLAN_WEIGHT="$XICM_GA_PLAN_WEIGHT" \
  XICM_GA_PLAN_BLOCK_CAP="$XICM_GA_PLAN_BLOCK_CAP" XICM_GA_PLAN_WEAK_CAP="$XICM_GA_PLAN_WEAK_CAP" \
  XICM_GA_PLAN_UNKNOWN_CAP="$XICM_GA_PLAN_UNKNOWN_CAP" XICM_GA_MAX_PER_TASK="$XICM_GA_MAX_PER_TASK" \
  XICM_GA_MAX_PER_FAMILY="$XICM_GA_MAX_PER_FAMILY" XICM_CLOSED_LOOP_MAX_REPLANS="$XICM_CLOSED_LOOP_MAX_REPLANS" \
  XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

config_path = Path(os.environ["METHOD_CONFIG_FILE"])
run_root = Path(os.environ["REMOTE_RUN_XICM"])
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
per_method = {}
total_complete = 0
for line in config_path.read_text().splitlines():
    if not line.strip():
        continue
    run_id, ranking, condition, cache_kind, alpha, beta, gamma = line.split("|")[:7]
    uses_plan = "geo_plan" in ranking or ".plan" in ranking
    uses_closed_loop = "closed_loop" in ranking.replace("-", "_") or ".cl" in ranking
    method = f"XICM_Cross.ZS_Ranking.{ranking}_{os.environ['MODEL_NAME']}_icl.{os.environ['DEMO_NUM_PER_ICL']}_test"
    method_dir = run_root / "logs" / method
    count = 0
    per_seed = {}
    for seed in seeds:
        seed_count = 0
        if method_dir.exists():
            for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
                if finish_re.search(path.read_text(errors="replace")):
                    count += 1
                    seed_count += 1
        per_seed[f"seed{seed}"] = seed_count
    total_complete += count
    per_method[run_id] = {
        "ranking_method": ranking,
        "method": method,
        "condition": condition,
        "cache_kind": cache_kind,
        "strict_final_seed_task_count": count,
        "total_seed_task_count": int(os.environ["TOTAL_SEED_TASKS"]),
        "per_seed": per_seed,
        "retrieval_weights": {
            "alpha_dynamic": float(alpha),
            "beta_geometry": float(beta),
            "gamma_contact": float(gamma),
            "delta_profile": float(os.environ["XICM_GA_DELTA"]),
            "penalty_weight": float(os.environ["XICM_GA_PENALTY"]),
            "plan_weight": float(os.environ["XICM_GA_PLAN_WEIGHT"]) if uses_plan else 0.0,
            "plan_block_cap": float(os.environ["XICM_GA_PLAN_BLOCK_CAP"]),
            "plan_weak_cap": float(os.environ["XICM_GA_PLAN_WEAK_CAP"]),
            "plan_unknown_cap": float(os.environ["XICM_GA_PLAN_UNKNOWN_CAP"]),
        },
        "uses_geometry_retrieval": float(beta) > 0,
        "uses_target_pose_prompt": ".geo" in ranking,
        "uses_contact_prompt": ".aff" in ranking and "geo_aff" not in ranking,
        "uses_plan_guided_retrieval": uses_plan,
        "uses_closed_loop": uses_closed_loop,
        "closed_loop_max_replans": int(os.environ["XICM_CLOSED_LOOP_MAX_REPLANS"]) if uses_closed_loop else 0,
        "contact_points_in_retrieval": False,
        "target_pose_in_retrieval": uses_plan,
        "retrieval_diversity_caps": {
            "max_per_task": int(os.environ["XICM_GA_MAX_PER_TASK"]),
            "max_per_family": int(os.environ["XICM_GA_MAX_PER_FAMILY"]),
        },
    }
payload = {
    "status": os.environ["STATUS"],
    "condition": "xicm_v1_component_ablation_10eps_3seed",
    "active_run_id": os.environ["ACTIVE_RUN_ID"],
    "active_ranking_method": os.environ["ACTIVE_RANKING"],
    "active_log_path": os.environ["ACTIVE_LOG"],
    "seeds": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "demo_num_per_icl": int(os.environ["DEMO_NUM_PER_ICL"]),
    "tasks_override": os.environ.get("XICM_TASKS_OVERRIDE", ""),
    "gpu_id": os.environ["GPU_ID"],
    "gpu_wait": {
        "enabled": os.environ["WAIT_FOR_GPU"] in {"1", "true", "yes"},
        "min_free_memory_mb": int(os.environ["MIN_FREE_GPU_MEMORY_MB"]),
        "max_util_percent": int(os.environ["MAX_GPU_UTIL_PERCENT"]),
    },
    "per_method": per_method,
    "completed_seed_task_csvs": total_complete,
    "total_seed_task_csvs": int(os.environ["TOTAL_ALL_SEED_TASKS"]),
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "component_root": os.environ["REMOTE_COMPONENT_ROOT"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

wait_for_gpu_capacity() {
  local run_id="$1"
  local ranking="$2"
  local log_path="$3"
  if [[ "$WAIT_FOR_GPU" != "1" && "$WAIT_FOR_GPU" != "true" && "$WAIT_FOR_GPU" != "yes" ]]; then
    return 0
  fi
  while true; do
    local stats used total util free
    stats="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    IFS=',' read -r used total util <<< "$stats"
    free=$(( total - used ))
    if (( free >= MIN_FREE_GPU_MEMORY_MB && util <= MAX_GPU_UTIL_PERCENT )); then
      {
        echo "gpu_wait_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util"
      } >> "$log_path"
      return 0
    fi
    {
      echo "gpu_wait_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_id=$GPU_ID free_memory_mb=$free used_memory_mb=$used total_memory_mb=$total util_percent=$util threshold_free_mb=$MIN_FREE_GPU_MEMORY_MB threshold_util_percent=$MAX_GPU_UTIL_PERCENT"
    } >> "$log_path"
    write_progress "waiting_for_gpu" "$run_id" "$ranking" "$log_path" "Waiting for GPU $GPU_ID capacity before starting $run_id."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

source /data/yf23/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REMOTE_RUN_XICM"

export COPPELIASIM_ROOT="$REMOTE_RUN_XICM/CoppeliaSim"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export HF_HOME="${HF_HOME:-/data/yf23/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yf23/huggingface/transformers}"
export HF_HUB_DISABLE_XET=1
export XICM_QWEN_7B_PATH="${XICM_QWEN_7B_PATH:-/data/yf23/models/Qwen2.5-7B-Instruct}"
export XICM_QWEN25_VL_7B_PATH="${XICM_QWEN25_VL_7B_PATH:-/data/yf23/checkpoints/ICRA27-ROBOT/Qwen2.5-VL-7B-Instruct}"
export XICM_VLLM_GPU_MEMORY_UTILIZATION="${XICM_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export XICM_VLLM_MAX_MODEL_LEN="${XICM_VLLM_MAX_MODEL_LEN:-24576}"
export XICM_VL_MAX_IMAGES="${XICM_VL_MAX_IMAGES:-2}"
export XICM_SD2_BASE_PATH="${XICM_SD2_BASE_PATH:-/data/yf23/models/Manojb-stable-diffusion-2-base}"
export MULTIPROCESSING_START_METHOD=spawn

while IFS='|' read -r run_id ranking condition cache_kind alpha beta gamma; do
  [[ -z "${run_id:-}" ]] && continue
  method="XICM_Cross.ZS_Ranking.${ranking}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
  if [[ "$cache_kind" == "contact" ]]; then
    export XICM_GA_REVIEW_BUNDLE="$REMOTE_CONTACT_CACHE"
  else
    export XICM_GA_REVIEW_BUNDLE="$REMOTE_CLEAN_GEOMETRY_CACHE"
  fi
  export XICM_GA_ALPHA="$alpha"
  export XICM_GA_BETA="$beta"
  export XICM_GA_GAMMA="$gamma"
  export XICM_GA_AUDIT_JSONL="$REMOTE_COMPONENT_ROOT/retrieval_audit_${condition}.jsonl"

  completed="$(count_method_completed "$method")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "skipped_completed" "$run_id" "$ranking" "" "$run_id already has all strict seed-task final scores."
    continue
  fi
  if [[ "$completed" -eq 0 ]]; then
    : > "$XICM_GA_AUDIT_JSONL"
  fi

  log_path="$REMOTE_RUNNER_LOG_ROOT/${condition}_seed${SEEDS}_$(date -u +%Y%m%d_%H%M%S).log"
  write_progress "running" "$run_id" "$ranking" "$log_path" "Started $run_id."
  wait_for_gpu_capacity "$run_id" "$ranking" "$log_path"
  (
    echo "run_id=$run_id"
    echo "condition=$condition"
    echo "ranking=$ranking"
    echo "method=$method"
    echo "seed=$SEEDS"
    echo "episodes=$EPISODES"
    echo "demo_num_per_icl=$DEMO_NUM_PER_ICL"
    echo "gpu_id=$GPU_ID"
    echo "review_bundle=$XICM_GA_REVIEW_BUNDLE"
    echo "retrieval_weights alpha=$XICM_GA_ALPHA beta=$XICM_GA_BETA gamma=$XICM_GA_GAMMA"
    echo "profile_weights delta=$XICM_GA_DELTA penalty=$XICM_GA_PENALTY"
    echo "plan_guided_weight=$XICM_GA_PLAN_WEIGHT block_cap=$XICM_GA_PLAN_BLOCK_CAP weak_cap=$XICM_GA_PLAN_WEAK_CAP unknown_cap=$XICM_GA_PLAN_UNKNOWN_CAP"
    echo "retrieval_diversity max_per_task=$XICM_GA_MAX_PER_TASK max_per_family=$XICM_GA_MAX_PER_FAMILY"
    echo "closed_loop_max_replans=$XICM_CLOSED_LOOP_MAX_REPLANS"
    echo "tasks_override=${XICM_TASKS_OVERRIDE:-}"
    echo "retrieval_audit_jsonl=$XICM_GA_AUDIT_JSONL"
    if [[ "$ranking" == *"geo_plan"* || "$ranking" == *".plan"* ]]; then
      echo "target_pose_in_retrieval=true"
      echo "uses_plan_guided_retrieval=true"
    else
      echo "target_pose_prompt_only=true"
      echo "uses_plan_guided_retrieval=false"
    fi
    echo "contact_points_in_retrieval=false"
    if [[ "$ranking" == *"closed_loop"* || "$ranking" == *".cl"* ]]; then
      echo "uses_closed_loop=true"
    else
      echo "uses_closed_loop=false"
    fi
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    bash scripts/eval_XICM.sh "$SEEDS" "$EPISODES" "$MODEL_NAME" "$DEMO_NUM_PER_ICL" "$GPU_ID" "$ranking" "true"
    status=$?
    set -e
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=$status"
    exit "$status"
  ) > "$log_path" 2>&1

  completed="$(count_method_completed "$method")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "method_completed" "$run_id" "$ranking" "$log_path" "$run_id finished all strict seed-task final scores."
  else
    write_progress "failed_or_partial" "$run_id" "$ranking" "$log_path" "$run_id exited before all strict seed-task final scores were present."
    exit 1
  fi
done < "$METHOD_CONFIG_FILE"

write_progress "completed" "" "" "" "All configured v1 10-episode component ablations finished."
RUNNER
chmod +x "$RUN_SCRIPT"

completed_all=0
while IFS='|' read -r _run_id ranking _condition _cache_kind _alpha _beta _gamma; do
  method="$(method_name_for_ranking "$ranking")"
  completed_all=$((completed_all + $(count_method_completed "$method")))
done < "$METHOD_CONFIG_FILE"

if [[ "$completed_all" -ge "$TOTAL_ALL_SEED_TASKS" ]]; then
  write_progress "skipped_completed" "" "" "" "All configured v1 10-episode component ablations already have strict seed-task final scores."
  echo "v1 10-episode component ablations already complete: $completed_all/$TOTAL_ALL_SEED_TASKS"
  exit 0
fi

write_progress "running" "" "" "" "Started v1 10-episode component ablation runner."
nohup env \
  REMOTE_COMPONENT_ROOT="$REMOTE_COMPONENT_ROOT" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  REMOTE_RUNNER_LOG_ROOT="$REMOTE_RUNNER_LOG_ROOT" \
  REMOTE_CLEAN_GEOMETRY_CACHE="$REMOTE_CLEAN_GEOMETRY_CACHE" \
  REMOTE_CONTACT_CACHE="$REMOTE_CONTACT_CACHE" \
  CONDA_ENV="$CONDA_ENV" \
  SEEDS="$SEEDS" \
  EPISODES="$EPISODES" \
  MODEL_NAME="$MODEL_NAME" \
  DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" \
  GPU_ID="$GPU_ID" \
  WAIT_FOR_GPU="$WAIT_FOR_GPU" \
  MIN_FREE_GPU_MEMORY_MB="$MIN_FREE_GPU_MEMORY_MB" \
  MAX_GPU_UTIL_PERCENT="$MAX_GPU_UTIL_PERCENT" \
  GPU_WAIT_INTERVAL_SECONDS="$GPU_WAIT_INTERVAL_SECONDS" \
  TOTAL_TASKS="$TOTAL_TASKS" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" \
  TOTAL_ALL_SEED_TASKS="$TOTAL_ALL_SEED_TASKS" \
  XICM_QWEN25_VL_7B_PATH="$XICM_QWEN25_VL_7B_PATH" \
  XICM_VLLM_GPU_MEMORY_UTILIZATION="$XICM_VLLM_GPU_MEMORY_UTILIZATION" \
  XICM_VLLM_MAX_MODEL_LEN="$XICM_VLLM_MAX_MODEL_LEN" \
  XICM_VL_MAX_IMAGES="$XICM_VL_MAX_IMAGES" \
  XICM_GA_DELTA="$XICM_GA_DELTA" \
  XICM_GA_PENALTY="$XICM_GA_PENALTY" \
  XICM_GA_PLAN_WEIGHT="$XICM_GA_PLAN_WEIGHT" \
  XICM_GA_PLAN_BLOCK_CAP="$XICM_GA_PLAN_BLOCK_CAP" \
  XICM_GA_PLAN_WEAK_CAP="$XICM_GA_PLAN_WEAK_CAP" \
  XICM_GA_PLAN_UNKNOWN_CAP="$XICM_GA_PLAN_UNKNOWN_CAP" \
  XICM_GA_MAX_PER_TASK="$XICM_GA_MAX_PER_TASK" \
  XICM_GA_MAX_PER_FAMILY="$XICM_GA_MAX_PER_FAMILY" \
  XICM_CLOSED_LOOP_MAX_REPLANS="$XICM_CLOSED_LOOP_MAX_REPLANS" \
  XICM_TASKS_OVERRIDE="$XICM_TASKS_OVERRIDE" \
  METHOD_CONFIG_FILE="$METHOD_CONFIG_FILE" \
  PROGRESS_JSON="$PROGRESS_JSON" \
  "$RUN_SCRIPT" \
  > "$REMOTE_RUNNER_LOG_ROOT/v1_10ep_component_launcher_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"
echo "Launched v1 10-episode component ablation runner PID $pid"
echo "Progress: $PROGRESS_JSON"
echo "Run root: $REMOTE_RUN_XICM"
echo "Runner logs: $REMOTE_RUNNER_LOG_ROOT"
echo "Method logs: $REMOTE_RUN_XICM/logs"
REMOTE_SCRIPT
