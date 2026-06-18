#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cair}"
INTERVAL="${INTERVAL:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
UNSEEN_PARTS="${UNSEEN_PARTS:-/data/yf23/datasets/ICRA27-ROBOT/unseen_tasks.tar.parts}"
MODEL_PARTS="${MODEL_PARTS:-/data/yf23/checkpoints/ICRA27-ROBOT/dynamics_diffusion.tar.parts}"
UNSEEN_SIZE=20184780800
MODEL_SIZE=10436526080

bar() {
  local bytes="$1"
  local total="$2"
  local width="${3:-40}"
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

chunk_number_from_tmp() {
  local name="$1"
  if [[ "$name" =~ part_([0-9]+)\.tmp ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

print_dataset() {
  local label="$1"
  local total_size="$2"
  local done_bytes="$3"
  local tmp_bytes="$4"
  local done_count="$5"
  local tmp_lines="$6"
  local total_bytes=$((done_bytes + tmp_bytes))

  printf "%s\n" "$label"
  printf "  total:   "
  bar "$total_bytes" "$total_size" 44
  printf " %s%%  %s / %s GiB\n" "$(percent "$total_bytes" "$total_size")" "$(gib "$total_bytes")" "$(gib "$total_size")"
  printf "  chunks:  %s completed\n" "$done_count"

  if [[ -n "$tmp_lines" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local tmp_name tmp_size chunk_label chunk_pct
      tmp_name=$(awk '{print $1}' <<<"$line")
      tmp_size=$(awk '{print $2}' <<<"$line")
      chunk_label=$(chunk_number_from_tmp "$tmp_name")
      chunk_pct=$(percent "$tmp_size" "$CHUNK_SIZE")
      printf "  active:  chunk %s  " "$chunk_label"
      bar "$tmp_size" "$CHUNK_SIZE" 44
      printf " %s%%  %s / %s MiB  (%s)\n" "$chunk_pct" \
        "$(awk -v b="$tmp_size" 'BEGIN { printf "%.2f", b / 1024 / 1024 }')" \
        "$(awk -v b="$CHUNK_SIZE" 'BEGIN { printf "%.2f", b / 1024 / 1024 }')" \
        "$tmp_name"
    done <<<"$tmp_lines"
  else
    printf "  active:  none\n"
  fi
}

fetch_status() {
  ssh -o BatchMode=yes "$REMOTE" "UNSEEN_PARTS='$UNSEEN_PARTS' MODEL_PARTS='$MODEL_PARTS' bash -s" <<'REMOTE_SCRIPT'
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
echo "UNSEEN_BYTES=$(sum_done "$UNSEEN_PARTS")"
echo "UNSEEN_TMP_BYTES=$(sum_tmp "$UNSEEN_PARTS")"
echo "UNSEEN_COUNT=$(count_done "$UNSEEN_PARTS")"
echo "MODEL_BYTES=$(sum_done "$MODEL_PARTS")"
echo "MODEL_TMP_BYTES=$(sum_tmp "$MODEL_PARTS")"
echo "MODEL_COUNT=$(count_done "$MODEL_PARTS")"
echo "__UNSEEN_TMP__"
tmp_lines "$UNSEEN_PARTS"
echo "__MODEL_TMP__"
tmp_lines "$MODEL_PARTS"
REMOTE_SCRIPT
}

trap 'printf "\nStopped watcher.\n"; exit 0' INT TERM

while true; do
  status=$(fetch_status)

  unseen_bytes=$(awk -F= '/^UNSEEN_BYTES=/{print $2}' <<<"$status")
  unseen_tmp_bytes=$(awk -F= '/^UNSEEN_TMP_BYTES=/{print $2}' <<<"$status")
  unseen_count=$(awk -F= '/^UNSEEN_COUNT=/{print $2}' <<<"$status")
  model_bytes=$(awk -F= '/^MODEL_BYTES=/{print $2}' <<<"$status")
  model_tmp_bytes=$(awk -F= '/^MODEL_TMP_BYTES=/{print $2}' <<<"$status")
  model_count=$(awk -F= '/^MODEL_COUNT=/{print $2}' <<<"$status")
  unseen_tmp=$(awk '/^__UNSEEN_TMP__/{flag=1;next}/^__MODEL_TMP__/{flag=0}flag' <<<"$status")
  model_tmp=$(awk '/^__MODEL_TMP__/{flag=1;next}flag' <<<"$status")

  clear
  printf "CAIR AGNOSTOS / X-ICM Download Progress\n"
  printf "Updated: %s | refresh: %ss | stop: Ctrl-C\n\n" "$(date)" "$INTERVAL"
  print_dataset "AGNOSTOS unseen" "$UNSEEN_SIZE" "$unseen_bytes" "$unseen_tmp_bytes" "$unseen_count" "$unseen_tmp"
  printf "\n"
  print_dataset "X-ICM dynamics diffusion" "$MODEL_SIZE" "$model_bytes" "$model_tmp_bytes" "$model_count" "$model_tmp"
  printf "\n"
  printf "Note: completed chunks are preserved on CAIR. Active .tmp chunks are the pieces currently uploading.\n"

  sleep "$INTERVAL"
done
