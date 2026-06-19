#!/usr/bin/env bash
set -u

REMOTE="${REMOTE:-cair}"
INTERVAL_SECONDS="${1:-10}"

UNSEEN_PARTS="${UNSEEN_PARTS:-/data/yf23/datasets/ICRA27-ROBOT/unseen_tasks.tar.parts}"
MODEL_PARTS="${MODEL_PARTS:-/data/yf23/checkpoints/ICRA27-ROBOT/dynamics_diffusion.tar.parts}"
QWEN_DIR="${QWEN_DIR:-/data/yf23/models/Qwen2.5-7B-Instruct}"
XICM_DIR="${XICM_DIR:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"

CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
UNSEEN_SIZE="${UNSEEN_SIZE:-20184780800}"
MODEL_SIZE="${MODEL_SIZE:-10436526080}"
QWEN_SIZE="${QWEN_SIZE:-15242788168}"

bar() {
  local bytes="$1"
  local total="$2"
  local width="${3:-36}"
  local filled
  filled=$(awk -v b="$bytes" -v t="$total" -v w="$width" 'BEGIN { if (t <= 0) print 0; else printf "%d", (b / t) * w }')
  if (( filled < 0 )); then filled=0; fi
  if (( filled > width )); then filled=$width; fi

  printf "["
  for ((i = 0; i < filled; i++)); do printf "#"; done
  for ((i = filled; i < width; i++)); do printf "."; done
  printf "]"
}

percent() {
  awk -v b="$1" -v t="$2" 'BEGIN { if (t <= 0) printf "0.000"; else printf "%.3f", (b / t) * 100 }'
}

gib() {
  awk -v b="$1" 'BEGIN { printf "%.3f", b / 1024 / 1024 / 1024 }'
}

mib() {
  awk -v b="$1" 'BEGIN { printf "%.2f", b / 1024 / 1024 }'
}

chunk_number_from_tmp() {
  local name="$1"
  if [[ "$name" =~ part_([0-9]+)\.tmp ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

print_part_progress() {
  local label="$1"
  local total_size="$2"
  local done_bytes="$3"
  local tmp_bytes="$4"
  local done_count="$5"
  local tmp_lines="$6"
  local total_bytes=$((done_bytes + tmp_bytes))

  printf "%s\n" "$label"
  printf "  total:   "
  bar "$total_bytes" "$total_size" 42
  printf " %s%%  %s / %s GiB\n" "$(percent "$total_bytes" "$total_size")" "$(gib "$total_bytes")" "$(gib "$total_size")"
  printf "  chunks:  %s completed\n" "$done_count"

  if [[ -n "$tmp_lines" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local tmp_name tmp_size chunk_id
      tmp_name=$(awk '{print $1}' <<<"$line")
      tmp_size=$(awk '{print $2}' <<<"$line")
      chunk_id=$(chunk_number_from_tmp "$tmp_name")
      printf "  active:  chunk %s " "$chunk_id"
      bar "$tmp_size" "$CHUNK_SIZE" 30
      printf " %s%%  %s / %s MiB  (%s)\n" "$(percent "$tmp_size" "$CHUNK_SIZE")" "$(mib "$tmp_size")" "$(mib "$CHUNK_SIZE")" "$tmp_name"
    done <<<"$tmp_lines"
  else
    printf "  active:  none\n"
  fi
}

fetch_remote_status() {
  ssh -o BatchMode=yes "$REMOTE" \
    "UNSEEN_PARTS='$UNSEEN_PARTS' MODEL_PARTS='$MODEL_PARTS' QWEN_DIR='$QWEN_DIR' XICM_DIR='$XICM_DIR' bash -s" <<'REMOTE_SCRIPT'
sum_done() {
  find "$1" -maxdepth 1 -type f ! -name '*.tmp' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
}
sum_tmp() {
  find "$1" -maxdepth 1 -type f -name '*.tmp' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
}
count_done() {
  find "$1" -maxdepth 1 -type f ! -name '*.tmp' 2>/dev/null | wc -l | tr -d ' '
}
tmp_lines() {
  find "$1" -maxdepth 1 -type f -name '*.tmp' -printf '%f %s bytes\n' 2>/dev/null | sort
}
qwen_bytes() {
  local top parts_dir
  top=$(find "$QWEN_DIR" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
  parts_dir="$QWEN_DIR/.parts"
  if [ -d "$parts_dir" ]; then
    find "$parts_dir" -mindepth 2 -maxdepth 2 -type f ! -name '*.tmp' -printf '%s\n' 2>/dev/null |
      awk -v top="$top" '{s+=$1} END {print top+s+0}'
  else
    echo "$top"
  fi
}
qwen_count() {
  local files chunks parts_dir
  files=$(find "$QWEN_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
  parts_dir="$QWEN_DIR/.parts"
  if [ -d "$parts_dir" ]; then
    chunks=$(find "$parts_dir" -mindepth 2 -maxdepth 2 -type f ! -name '*.tmp' 2>/dev/null | wc -l | tr -d ' ')
  else
    chunks=0
  fi
  echo "$files files, $chunks chunks"
}
echo "UNSEEN_BYTES=$(sum_done "$UNSEEN_PARTS")"
echo "UNSEEN_TMP_BYTES=$(sum_tmp "$UNSEEN_PARTS")"
echo "UNSEEN_COUNT=$(count_done "$UNSEEN_PARTS")"
echo "MODEL_BYTES=$(sum_done "$MODEL_PARTS")"
echo "MODEL_TMP_BYTES=$(sum_tmp "$MODEL_PARTS")"
echo "MODEL_COUNT=$(count_done "$MODEL_PARTS")"
echo "QWEN_BYTES=$(qwen_bytes)"
echo "QWEN_COUNT=$(qwen_count)"
echo "__UNSEEN_TMP__"
tmp_lines "$UNSEEN_PARTS"
echo "__MODEL_TMP__"
tmp_lines "$MODEL_PARTS"
echo "__REMOTE_XICM__"
if [ -d "$XICM_DIR" ]; then
  cd "$XICM_DIR"
  printf "seen_tasks="
  find -L data/seen_tasks/train -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '
  printf "\nunseen_tasks="
  find -L data/unseen_tasks/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '
  printf "\ndynamics_checkpoint="
  if [ -f data/dynamics_diffusion/all_diffusion_features.pkl ]; then
    ls -lh data/dynamics_diffusion/all_diffusion_features.pkl | awk '{print $5}'
  else
    printf "missing"
  fi
  printf "\nresult_csvs="
  find logs/XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test -path '*/seed0/test_data.csv' -type f 2>/dev/null | wc -l | tr -d ' '
  printf "\nlatest_log="
  ls -t logs/baseline_xicm_original_prompt/run_*.log 2>/dev/null | head -1 || true
  printf "\n__SCORES__\n"
  latest=$(ls -t logs/baseline_xicm_original_prompt/run_*.log 2>/dev/null | head -1 || true)
  if [ -n "$latest" ]; then
    grep -aE '^Finished .*Final Score' "$latest" 2>/dev/null | tail -25 || true
    printf "__LOG_TAIL__\n"
    tail -n 35 "$latest" 2>/dev/null || true
  else
    printf "__LOG_TAIL__\nno baseline log yet\n"
  fi
else
  printf "seen_tasks=missing_xicm_dir\nunseen_tasks=missing_xicm_dir\ndynamics_checkpoint=missing_xicm_dir\nresult_csvs=0\nlatest_log=\n__SCORES__\n__LOG_TAIL__\n"
fi
echo "__PROCESSES__"
ps -u yf23 -o pid,etime,stat,cmd | grep -E 'run_baseline|eval_XICM|main.py|vllm|Xvfb|xvfb' | grep -v grep || true
echo "__GPU__"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
echo "__QWEN__"
echo "QWEN_COUNT=$(qwen_count)"
REMOTE_SCRIPT
}

trap 'printf "\nStopped watcher. Remote downloads/jobs keep running.\n"; exit 0' INT TERM

while true; do
  status=$(fetch_remote_status 2>&1)

  unseen_bytes=$(awk -F= '/^UNSEEN_BYTES=/{print $2}' <<<"$status")
  unseen_tmp_bytes=$(awk -F= '/^UNSEEN_TMP_BYTES=/{print $2}' <<<"$status")
  unseen_count=$(awk -F= '/^UNSEEN_COUNT=/{print $2}' <<<"$status")
  model_bytes=$(awk -F= '/^MODEL_BYTES=/{print $2}' <<<"$status")
  model_tmp_bytes=$(awk -F= '/^MODEL_TMP_BYTES=/{print $2}' <<<"$status")
  model_count=$(awk -F= '/^MODEL_COUNT=/{print $2}' <<<"$status")
  qwen_bytes=$(awk -F= '/^QWEN_BYTES=/{print $2}' <<<"$status")
  qwen_count=$(awk -F= '/^QWEN_COUNT=/{print $2}' <<<"$status" | head -1)
  unseen_tmp=$(awk '/^__UNSEEN_TMP__/{flag=1;next}/^__MODEL_TMP__/{flag=0}flag' <<<"$status")
  model_tmp=$(awk '/^__MODEL_TMP__/{flag=1;next}/^__REMOTE_XICM__/{flag=0}flag' <<<"$status")
  remote_xicm=$(awk '/^__REMOTE_XICM__/{flag=1;next}/^__SCORES__/{flag=0}flag' <<<"$status")
  scores=$(awk '/^__SCORES__/{flag=1;next}/^__LOG_TAIL__/{flag=0}flag' <<<"$status")
  log_tail=$(awk '/^__LOG_TAIL__/{flag=1;next}/^__PROCESSES__/{flag=0}flag' <<<"$status")
  processes=$(awk '/^__PROCESSES__/{flag=1;next}/^__GPU__/{flag=0}flag' <<<"$status")
  gpu=$(awk '/^__GPU__/{flag=1;next}/^__QWEN__/{flag=0}flag' <<<"$status")

  clear
  printf "CAIR AGNOSTOS / X-ICM live progress\n"
  printf "Updated: %s | refresh: %ss | remote: %s | stop: Ctrl-C\n\n" "$(date)" "$INTERVAL_SECONDS" "$REMOTE"

  if grep -q 'ssh:' <<<"$status"; then
    printf "SSH/status error:\n%s\n\n" "$status"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  print_part_progress "AGNOSTOS unseen archive" "$UNSEEN_SIZE" "${unseen_bytes:-0}" "${unseen_tmp_bytes:-0}" "${unseen_count:-0}" "$unseen_tmp"
  printf "\n"
  print_part_progress "X-ICM dynamics archive" "$MODEL_SIZE" "${model_bytes:-0}" "${model_tmp_bytes:-0}" "${model_count:-0}" "$model_tmp"
  printf "\nQwen2.5-7B-Instruct\n"
  printf "  total:   "
  bar "${qwen_bytes:-0}" "$QWEN_SIZE" 42
  printf " %s%%  %s / %s GiB\n" "$(percent "${qwen_bytes:-0}" "$QWEN_SIZE")" "$(gib "${qwen_bytes:-0}")" "$(gib "$QWEN_SIZE")"
  printf "  files:   %s\n\n" "${qwen_count:-unknown}"

  printf "X-ICM linked data/status\n"
  printf "%s\n\n" "$remote_xicm" | sed 's/^/  /'

  printf "Finished baseline task scores\n"
  if [[ -n "$scores" ]]; then
    printf "%s\n\n" "$scores" | sed 's/^/  /'
  else
    printf "  none yet\n\n"
  fi

  printf "Active CAIR processes\n"
  if [[ -n "$processes" ]]; then
    printf "%s\n\n" "$processes" | sed 's/^/  /'
  else
    printf "  none\n\n"
  fi

  printf "GPU usage\n"
  if [[ -n "$gpu" ]]; then
    printf "%s\n\n" "$gpu" | sed 's/^/  GPU /'
  else
    printf "  unavailable\n\n"
  fi

  printf "Latest baseline log tail\n"
  printf "%s\n" "$log_tail" | tail -35 | sed 's/^/  /'

  sleep "$INTERVAL_SECONDS"
done
