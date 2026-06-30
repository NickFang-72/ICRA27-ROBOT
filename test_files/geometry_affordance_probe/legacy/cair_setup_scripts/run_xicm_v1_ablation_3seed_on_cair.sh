#!/usr/bin/env bash
set -Eeuo pipefail

# Launch clean v1 X-ICM ablations on CAIR:
#   1) baseline + geometry retrieval
#   2) baseline + geometry retrieval + prompt-side contact hints
#
# Both rows use seeds 0,50,99, 25 episodes per task, k=18 ICL demos, and
# retrieval score alpha*S_dyn + beta*S_geo. Contact hints are not used in
# retrieval; they are prompt-only evidence for the second row.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CAIR_HOST="${CAIR_HOST:-cair}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/data/yf23/projects/ICRA27-ROBOT}"
REMOTE_SOURCE_XICM="${REMOTE_SOURCE_XICM:-$REMOTE_PROJECT_ROOT/X-ICM}"
REMOTE_V1_ROOT="${REMOTE_V1_ROOT:-$REMOTE_PROJECT_ROOT/experiments/v1_ablation_3seed_20260624}"
REMOTE_RUN_XICM="${REMOTE_RUN_XICM:-$REMOTE_V1_ROOT/X-ICM_v1}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-$REMOTE_V1_ROOT/logs}"
REMOTE_CLEAN_GEOMETRY_CACHE="${REMOTE_CLEAN_GEOMETRY_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_v1_primitive_full_cache_20260623/review_bundle.jsonl}"
REMOTE_OLD_CONTACT_CACHE="${REMOTE_OLD_CONTACT_CACHE:-$REMOTE_PROJECT_ROOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl}"
REMOTE_CONTACT_CACHE_DIR="${REMOTE_CONTACT_CACHE_DIR:-$REMOTE_V1_ROOT/merged_v1_geometry_with_contact_hints}"
REMOTE_CONTACT_CACHE="${REMOTE_CONTACT_CACHE:-$REMOTE_CONTACT_CACHE_DIR/review_bundle.jsonl}"
CONDA_ENV="${CONDA_ENV:-/data/yf23/conda/envs/zero-shot}"
SEEDS="${SEEDS:-0,50,99}"
EPISODES="${EPISODES:-25}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5.7B.instruct}"
DEMO_NUM_PER_ICL="${DEMO_NUM_PER_ICL:-18}"
GPU_ID="${GPU_ID:-0}"
TOTAL_TASKS="${TOTAL_TASKS:-23}"
ALPHA="${XICM_GA_ALPHA:-0.70}"
BETA="${XICM_GA_BETA:-0.30}"
GAMMA="${XICM_GA_GAMMA:-0.0}"
RANKING_METHODS="${RANKING_METHODS:-lang_vis.out.geo,lang_vis.out.geo.aff}"
RESET_RUN_TREE="${RESET_RUN_TREE:-0}"
FORCE_REBUILD_CONTACT_CACHE="${FORCE_REBUILD_CONTACT_CACHE:-0}"

LOCAL_FORM="$REPO_ROOT/X-ICM/form_icl_demonstrations_crosstask_ranking.py"
LOCAL_AGENT="$REPO_ROOT/X-ICM/crosstask_icl_agent.py"

ssh "$CAIR_HOST" \
  "REMOTE_SOURCE_XICM='$REMOTE_SOURCE_XICM' REMOTE_V1_ROOT='$REMOTE_V1_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' RESET_RUN_TREE='$RESET_RUN_TREE' bash -s" <<'REMOTE_SETUP'
set -Eeuo pipefail

mkdir -p "$REMOTE_V1_ROOT" "$REMOTE_LOG_ROOT"
if [[ "$RESET_RUN_TREE" == "1" && -d "$REMOTE_RUN_XICM" ]]; then
  rm -rf "$REMOTE_RUN_XICM"
fi
if [[ ! -d "$REMOTE_RUN_XICM" ]]; then
  echo "Creating isolated v1 X-ICM tree at $REMOTE_RUN_XICM"
  mkdir -p "$REMOTE_RUN_XICM"
  rsync -a --delete \
    --exclude 'logs' \
    --exclude 'outputs' \
    "$REMOTE_SOURCE_XICM/" "$REMOTE_RUN_XICM/"
fi
mkdir -p "$REMOTE_RUN_XICM/logs"
REMOTE_SETUP

rsync -a "$LOCAL_FORM" "$CAIR_HOST:$REMOTE_RUN_XICM/form_icl_demonstrations_crosstask_ranking.py"
rsync -a "$LOCAL_AGENT" "$CAIR_HOST:$REMOTE_RUN_XICM/crosstask_icl_agent.py"

ssh "$CAIR_HOST" \
  "REMOTE_V1_ROOT='$REMOTE_V1_ROOT' REMOTE_RUN_XICM='$REMOTE_RUN_XICM' REMOTE_LOG_ROOT='$REMOTE_LOG_ROOT' REMOTE_CLEAN_GEOMETRY_CACHE='$REMOTE_CLEAN_GEOMETRY_CACHE' REMOTE_OLD_CONTACT_CACHE='$REMOTE_OLD_CONTACT_CACHE' REMOTE_CONTACT_CACHE_DIR='$REMOTE_CONTACT_CACHE_DIR' REMOTE_CONTACT_CACHE='$REMOTE_CONTACT_CACHE' CONDA_ENV='$CONDA_ENV' SEEDS='$SEEDS' EPISODES='$EPISODES' MODEL_NAME='$MODEL_NAME' DEMO_NUM_PER_ICL='$DEMO_NUM_PER_ICL' GPU_ID='$GPU_ID' TOTAL_TASKS='$TOTAL_TASKS' ALPHA='$ALPHA' BETA='$BETA' GAMMA='$GAMMA' RANKING_METHODS='$RANKING_METHODS' FORCE_REBUILD_CONTACT_CACHE='$FORCE_REBUILD_CONTACT_CACHE' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a RANKING_LIST <<< "$RANKING_METHODS"
TOTAL_SEED_TASKS=$(( ${#SEED_LIST[@]} * TOTAL_TASKS ))
TOTAL_ALL_SEED_TASKS=$(( ${#RANKING_LIST[@]} * TOTAL_SEED_TASKS ))
PROGRESS_JSON="$REMOTE_V1_ROOT/progress_v1.json"
RUN_SCRIPT="$REMOTE_V1_ROOT/run_v1_ablation_3seed.sh"

method_name_for_ranking() {
  local ranking="$1"
  printf "XICM_Cross.ZS_Ranking.%s_%s_icl.%s_test" "$ranking" "$MODEL_NAME" "$DEMO_NUM_PER_ICL"
}

cache_for_ranking() {
  local ranking="$1"
  if [[ "$ranking" == *".aff"* && "$ranking" != *"geo_aff"* ]]; then
    printf "%s" "$REMOTE_CONTACT_CACHE"
  else
    printf "%s" "$REMOTE_CLEAN_GEOMETRY_CACHE"
  fi
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
  local active_ranking="$2"
  local log_path="$3"
  local message="$4"
  STATUS="$status" ACTIVE_RANKING="$active_ranking" ACTIVE_LOG="$log_path" MESSAGE="$message" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" REMOTE_V1_ROOT="$REMOTE_V1_ROOT" RANKING_METHODS="$RANKING_METHODS" \
  MODEL_NAME="$MODEL_NAME" DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" SEEDS="$SEEDS" EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" TOTAL_TASKS="$TOTAL_TASKS" ALPHA="$ALPHA" BETA="$BETA" GAMMA="$GAMMA" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" TOTAL_ALL_SEED_TASKS="$TOTAL_ALL_SEED_TASKS" PROGRESS_JSON="$PROGRESS_JSON" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

rankings = [item.strip() for item in os.environ["RANKING_METHODS"].split(",") if item.strip()]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
run_root = Path(os.environ["REMOTE_RUN_XICM"])
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
per_method = {}
total_complete = 0
for ranking in rankings:
    method = f"XICM_Cross.ZS_Ranking.{ranking}_{os.environ['MODEL_NAME']}_icl.{os.environ['DEMO_NUM_PER_ICL']}_test"
    method_dir = run_root / "logs" / method
    count = 0
    per_seed = {}
    if method_dir.exists():
        for seed in seeds:
            seed_count = 0
            for path in method_dir.glob(f"*/seed{seed}/test_data.csv"):
                if finish_re.search(path.read_text(errors="replace")):
                    count += 1
                    seed_count += 1
            per_seed[f"seed{seed}"] = seed_count
    else:
        per_seed = {f"seed{seed}": 0 for seed in seeds}
    total_complete += count
    per_method[ranking] = {
        "method": method,
        "strict_final_seed_task_count": count,
        "total_seed_task_count": int(os.environ["TOTAL_SEED_TASKS"]),
        "per_seed": per_seed,
    }
payload = {
    "status": os.environ["STATUS"],
    "condition": "xicm_v1_geometry_contact_ablation_3seed",
    "active_ranking_method": os.environ["ACTIVE_RANKING"],
    "active_log_path": os.environ["ACTIVE_LOG"],
    "seed": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "demo_num_per_icl": int(os.environ["DEMO_NUM_PER_ICL"]),
    "gpu_id": os.environ["GPU_ID"],
    "retrieval_weights": {
        "alpha_dynamic": float(os.environ["ALPHA"]),
        "beta_geometry": float(os.environ["BETA"]),
        "gamma_contact": float(os.environ["GAMMA"]),
    },
    "contact_points_in_retrieval": False,
    "per_method": per_method,
    "completed_seed_task_csvs": total_complete,
    "total_seed_task_csvs": int(os.environ["TOTAL_ALL_SEED_TASKS"]),
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

if pgrep -af "v1_ablation_3seed_20260624|X-ICM_v1|XICM_Cross.ZS_Ranking.lang_vis.out.geo.*icl.18_test" | grep -v pgrep >/dev/null 2>&1; then
  write_progress "already_running" "" "" "A v1 ablation process already appears to be running."
  echo "v1 ablation already appears to be running."
  exit 0
fi

if [[ "$FORCE_REBUILD_CONTACT_CACHE" == "1" || ! -f "$REMOTE_CONTACT_CACHE" ]]; then
  mkdir -p "$REMOTE_CONTACT_CACHE_DIR"
  REMOTE_CLEAN_GEOMETRY_CACHE="$REMOTE_CLEAN_GEOMETRY_CACHE" \
  REMOTE_OLD_CONTACT_CACHE="$REMOTE_OLD_CONTACT_CACHE" \
  REMOTE_CONTACT_CACHE="$REMOTE_CONTACT_CACHE" \
  python3 - <<'PY'
from pathlib import Path
import json
import re

clean_path = Path(__import__("os").environ["REMOTE_CLEAN_GEOMETRY_CACHE"])
old_path = Path(__import__("os").environ["REMOTE_OLD_CONTACT_CACHE"])
out_path = Path(__import__("os").environ["REMOTE_CONTACT_CACHE"])

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
  local active_ranking="$2"
  local log_path="$3"
  local message="$4"
  STATUS="$status" ACTIVE_RANKING="$active_ranking" ACTIVE_LOG="$log_path" MESSAGE="$message" \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" RANKING_METHODS="$RANKING_METHODS" \
  MODEL_NAME="$MODEL_NAME" DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" SEEDS="$SEEDS" EPISODES="$EPISODES" \
  GPU_ID="$GPU_ID" ALPHA="$ALPHA" BETA="$BETA" GAMMA="$GAMMA" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" PROGRESS_JSON="$PROGRESS_JSON" \
  python3 - <<'PY'
from pathlib import Path
import json
import os
import re
from datetime import datetime, timezone

rankings = [item.strip() for item in os.environ["RANKING_METHODS"].split(",") if item.strip()]
seeds = [item.strip() for item in os.environ["SEEDS"].split(",") if item.strip()]
run_root = Path(os.environ["REMOTE_RUN_XICM"])
finish_re = re.compile(r"Finished\s+[^|]+?\s+\|\s+Final Score:\s*[-+]?\d+(?:\.\d+)?")
per_method = {}
total_complete = 0
for ranking in rankings:
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
    per_method[ranking] = {
        "method": method,
        "strict_final_seed_task_count": count,
        "total_seed_task_count": int(os.environ["TOTAL_SEED_TASKS"]),
        "per_seed": per_seed,
    }
payload = {
    "status": os.environ["STATUS"],
    "condition": "xicm_v1_geometry_contact_ablation_3seed",
    "active_ranking_method": os.environ["ACTIVE_RANKING"],
    "active_log_path": os.environ["ACTIVE_LOG"],
    "seed": os.environ["SEEDS"],
    "episodes": int(os.environ["EPISODES"]),
    "demo_num_per_icl": int(os.environ["DEMO_NUM_PER_ICL"]),
    "gpu_id": os.environ["GPU_ID"],
    "retrieval_weights": {
        "alpha_dynamic": float(os.environ["ALPHA"]),
        "beta_geometry": float(os.environ["BETA"]),
        "gamma_contact": float(os.environ["GAMMA"]),
    },
    "contact_points_in_retrieval": False,
    "per_method": per_method,
    "completed_seed_task_csvs": total_complete,
    "total_seed_task_csvs": len(rankings) * int(os.environ["TOTAL_SEED_TASKS"]),
    "run_xicm_root": os.environ["REMOTE_RUN_XICM"],
    "message": os.environ["MESSAGE"],
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(os.environ["PROGRESS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
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
export XICM_SD2_BASE_PATH="${XICM_SD2_BASE_PATH:-/data/yf23/models/Manojb-stable-diffusion-2-base}"
export MULTIPROCESSING_START_METHOD=spawn
export XICM_GA_ALPHA="$ALPHA"
export XICM_GA_BETA="$BETA"
export XICM_GA_GAMMA="$GAMMA"

IFS=',' read -r -a rankings <<< "$RANKING_METHODS"
for ranking in "${rankings[@]}"; do
  method="XICM_Cross.ZS_Ranking.${ranking}_${MODEL_NAME}_icl.${DEMO_NUM_PER_ICL}_test"
  if [[ "$ranking" == *".aff"* && "$ranking" != *"geo_aff"* ]]; then
    export XICM_GA_REVIEW_BUNDLE="$REMOTE_CONTACT_CACHE"
    condition="v1_geometry_contact_prompt"
  else
    export XICM_GA_REVIEW_BUNDLE="$REMOTE_CLEAN_GEOMETRY_CACHE"
    condition="v1_geometry_retrieval"
  fi

  completed="$(count_method_completed "$method")"
  if [[ "$completed" -ge "$TOTAL_SEED_TASKS" ]]; then
    write_progress "skipped_completed" "$ranking" "" "$ranking already has all strict seed-task final scores."
    continue
  fi

  log_path="$REMOTE_LOG_ROOT/${condition}_seed${SEEDS}_$(date -u +%Y%m%d_%H%M%S).log"
  write_progress "running" "$ranking" "$log_path" "Started $ranking."
  (
    echo "condition=$condition"
    echo "ranking=$ranking"
    echo "method=$method"
    echo "seed=$SEEDS"
    echo "episodes=$EPISODES"
    echo "demo_num_per_icl=$DEMO_NUM_PER_ICL"
    echo "gpu_id=$GPU_ID"
    echo "review_bundle=$XICM_GA_REVIEW_BUNDLE"
    echo "retrieval_weights alpha=$XICM_GA_ALPHA beta=$XICM_GA_BETA gamma=$XICM_GA_GAMMA"
    echo "contact_points_in_retrieval=false"
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
    write_progress "method_completed" "$ranking" "$log_path" "$ranking finished all strict seed-task final scores."
  else
    write_progress "failed_or_partial" "$ranking" "$log_path" "$ranking exited before all strict seed-task final scores were present."
    exit 1
  fi
done

write_progress "completed" "" "" "All configured v1 ablations finished."
RUNNER
chmod +x "$RUN_SCRIPT"

completed_all=0
for ranking in "${RANKING_LIST[@]}"; do
  method="$(method_name_for_ranking "$ranking")"
  completed_all=$((completed_all + $(count_method_completed "$method")))
done
if [[ "$completed_all" -ge "$TOTAL_ALL_SEED_TASKS" ]]; then
  write_progress "skipped_completed" "" "" "All configured v1 ablations already have strict seed-task final scores."
  echo "v1 ablations already complete: $completed_all/$TOTAL_ALL_SEED_TASKS"
  exit 0
fi

write_progress "running" "" "" "Started v1 ablation benchmark runner."
nohup env \
  REMOTE_RUN_XICM="$REMOTE_RUN_XICM" \
  REMOTE_LOG_ROOT="$REMOTE_LOG_ROOT" \
  REMOTE_CLEAN_GEOMETRY_CACHE="$REMOTE_CLEAN_GEOMETRY_CACHE" \
  REMOTE_CONTACT_CACHE="$REMOTE_CONTACT_CACHE" \
  CONDA_ENV="$CONDA_ENV" \
  SEEDS="$SEEDS" \
  EPISODES="$EPISODES" \
  MODEL_NAME="$MODEL_NAME" \
  DEMO_NUM_PER_ICL="$DEMO_NUM_PER_ICL" \
  GPU_ID="$GPU_ID" \
  TOTAL_TASKS="$TOTAL_TASKS" \
  TOTAL_SEED_TASKS="$TOTAL_SEED_TASKS" \
  RANKING_METHODS="$RANKING_METHODS" \
  ALPHA="$ALPHA" \
  BETA="$BETA" \
  GAMMA="$GAMMA" \
  PROGRESS_JSON="$PROGRESS_JSON" \
  "$RUN_SCRIPT" \
  > "$REMOTE_LOG_ROOT/v1_ablation_launcher_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

pid="$!"
echo "$pid" > "$REMOTE_V1_ROOT/v1_ablation.pid"
echo "Launched v1 ablation runner PID $pid"
echo "Progress: $PROGRESS_JSON"
echo "Run root: $REMOTE_RUN_XICM"
echo "Logs: $REMOTE_LOG_ROOT"
REMOTE_SCRIPT
