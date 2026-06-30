#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cair}"
UNSEEN_PARTS="${UNSEEN_PARTS:-/data/yf23/datasets/ICRA27-ROBOT/unseen_tasks.tar.parts}"
MODEL_PARTS="${MODEL_PARTS:-/data/yf23/checkpoints/ICRA27-ROBOT/dynamics_diffusion.tar.parts}"
QWEN_DIR="${QWEN_DIR:-/data/yf23/models/Qwen2.5-7B-Instruct}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
UNSEEN_SIZE=20184780800
MODEL_SIZE=10436526080
QWEN_SIZE=15242788168

bar() {
  local label="$1"
  local bytes="$2"
  local total="$3"
  local width=30
  local pct filled empty
  pct=$(( bytes * 100 / total ))
  filled=$(( pct * width / 100 ))
  empty=$(( width - filled ))
  printf "%-20s [" "$label"
  for ((i = 0; i < filled; i++)); do printf "#"; done
  for ((i = 0; i < empty; i++)); do printf "."; done
  printf "] %3d%%  %.2f / %.2f GiB\n" "$pct" \
    "$(awk "BEGIN {print $bytes/1024/1024/1024}")" \
    "$(awk "BEGIN {print $total/1024/1024/1024}")"
}

status=$(
  ssh -o BatchMode=yes "$REMOTE" "UNSEEN_PARTS='$UNSEEN_PARTS' MODEL_PARTS='$MODEL_PARTS' QWEN_DIR='$QWEN_DIR' bash -s" <<'REMOTE_SCRIPT'
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
  find "$1" -maxdepth 1 -type f -name '*.tmp' -printf '%f %s bytes\n' 2>/dev/null | sort | tail -3
}
qwen_bytes() {
  local top parts_dir file
  top=$(find "$QWEN_DIR" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
  parts_dir="$QWEN_DIR/.parts"
  if [ -d "$parts_dir" ]; then
    find "$parts_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | while read -r file; do
      [ -f "$QWEN_DIR/$file" ] && continue
      find "$parts_dir/$file" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null
    done | awk -v top="$top" '{s+=$1} END {print top+s+0}'
  else
    echo "$top"
  fi
}
qwen_count() {
  local files parts_dir chunks
  files=$(find "$QWEN_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
  parts_dir="$QWEN_DIR/.parts"
  if [ -d "$parts_dir" ]; then
    chunks=$(find "$parts_dir" -mindepth 2 -maxdepth 2 -type f ! -name '*.tmp' 2>/dev/null | wc -l | tr -d ' ')
  else
    chunks=0
  fi
  echo "${files} files, ${chunks} chunks"
}
qwen_tmp_lines() {
  find "$QWEN_DIR/.parts" -mindepth 2 -maxdepth 2 -type f -name '*.tmp' -printf '%h/%f %s bytes\n' 2>/dev/null | sed "s#^$QWEN_DIR/.parts/##" | sort | tail -3
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
echo "__QWEN_TMP__"
qwen_tmp_lines
REMOTE_SCRIPT
)

unseen_bytes=$(printf "%s\n" "$status" | awk -F= '/^UNSEEN_BYTES=/{print $2}')
unseen_tmp_bytes=$(printf "%s\n" "$status" | awk -F= '/^UNSEEN_TMP_BYTES=/{print $2}')
model_bytes=$(printf "%s\n" "$status" | awk -F= '/^MODEL_BYTES=/{print $2}')
model_tmp_bytes=$(printf "%s\n" "$status" | awk -F= '/^MODEL_TMP_BYTES=/{print $2}')
qwen_bytes=$(printf "%s\n" "$status" | awk -F= '/^QWEN_BYTES=/{print $2}')
unseen_count=$(printf "%s\n" "$status" | awk -F= '/^UNSEEN_COUNT=/{print $2}')
model_count=$(printf "%s\n" "$status" | awk -F= '/^MODEL_COUNT=/{print $2}')
qwen_count=$(printf "%s\n" "$status" | awk -F= '/^QWEN_COUNT=/{print $2}')
unseen_tmp=$(printf "%s\n" "$status" | awk '/^__UNSEEN_TMP__/{flag=1;next}/^__MODEL_TMP__/{flag=0}flag')
model_tmp=$(printf "%s\n" "$status" | awk '/^__MODEL_TMP__/{flag=1;next}/^__QWEN_TMP__/{flag=0}flag')
qwen_tmp=$(printf "%s\n" "$status" | awk '/^__QWEN_TMP__/{flag=1;next}flag')
unseen_total_bytes=$((unseen_bytes + unseen_tmp_bytes))
model_total_bytes=$((model_bytes + model_tmp_bytes))
unseen_chunk_pct=$((unseen_tmp_bytes * 100 / CHUNK_SIZE))
model_chunk_pct=$((model_tmp_bytes * 100 / CHUNK_SIZE))

date
bar "AGNOSTOS unseen" "$unseen_total_bytes" "$UNSEEN_SIZE"
echo "  completed chunks: $unseen_count"
if (( unseen_tmp_bytes > 0 )); then
  echo "  current chunk: ${unseen_chunk_pct}%"
fi
printf "%s\n" "$unseen_tmp" | sed '/^$/d;s/^/  active tmp: /'
bar "X-ICM dynamics" "$model_total_bytes" "$MODEL_SIZE"
echo "  completed chunks: $model_count"
if (( model_tmp_bytes > 0 )); then
  echo "  current chunk: ${model_chunk_pct}%"
fi
printf "%s\n" "$model_tmp" | sed '/^$/d;s/^/  active tmp: /'
bar "Qwen2.5 model" "$qwen_bytes" "$QWEN_SIZE"
echo "  completed: $qwen_count"
printf "%s\n" "$qwen_tmp" | sed '/^$/d;s/^/  active tmp: /'
