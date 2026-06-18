#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
REMOTE="${REMOTE:-cair}"
REMOTE_MODEL_DIR="${REMOTE_MODEL_DIR:-/data/yf23/models/Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-$ROOT_DIR/test_files/geometry_affordance_probe/qwen2_5_7b_instruct_manifest.json}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-1}"
LOCK_DIR="${LOCK_DIR:-$ROOT_DIR/test_files/geometry_affordance_probe/qwen25_download.lock}"
SSH_OPTS=(-n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12)
RSYNC_RSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12"

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return 0
  fi

  local old_pid
  old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
  if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
    echo "$$" > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return 0
  fi

  echo "Qwen relay already running; waiting on lock at $LOCK_DIR" >&2
  while [ -d "$LOCK_DIR" ]; do
    sleep 60
    old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR"
      echo "$$" > "$LOCK_DIR/pid"
      trap 'rm -rf "$LOCK_DIR"' EXIT
      return 0
    fi
  done
  mkdir "$LOCK_DIR"
  echo "$$" > "$LOCK_DIR/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT
  return 0
}

acquire_lock

remote_cmd() {
  local attempt
  for attempt in $(seq 1 12); do
    if ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; then
      return 0
    fi
    echo "[remote] ssh command failed (attempt $attempt/12); retrying..." >&2
    sleep 10
  done
  return 1
}

remote_size_or_zero() {
  local path="$1"
  local size
  size=$(remote_cmd "[ -f '$path' ] && stat -c %s '$path' || echo 0" 2>/dev/null)
  if [[ "$size" =~ ^[0-9]+$ ]]; then
    echo "$size"
  else
    echo 0
  fi
}

stream_one_chunk() {
  local label="$1"
  local url="$2"
  local remote_parts="$3"
  local i="$4"
  local start="$5"
  local end="$6"
  local expected="$7"

  local part
  part=$(printf "%s/part_%04d" "$remote_parts" "$i")
  local have
  have=$(remote_size_or_zero "$part")

  if [ "$have" = "$expected" ]; then
    echo "[$label] chunk $((i + 1)) already complete ($expected bytes)"
    return 0
  fi

  echo "[$label] streaming chunk $((i + 1)) bytes $start-$end ($expected bytes)"
  local tmp
  tmp=$(mktemp "${TMPDIR:-/tmp}/${label}.part_${i}.XXXXXX")

  local attempt
  for attempt in $(seq 1 20); do
    echo "[$label] chunk $((i + 1)) downloading to local temp (attempt $attempt/20)"
    rm -f "$tmp"
    if curl --http1.1 -sS -L --fail --retry 5 --retry-delay 5 \
      --connect-timeout 30 --speed-time 240 --speed-limit 1024 \
      -r "$start-$end" "$url" -o "$tmp"; then
      break
    fi
    if [ "$attempt" = 20 ]; then
      echo "[$label] failed to download chunk $((i + 1)) after $attempt attempts" >&2
      rm -f "$tmp"
      return 1
    fi
    sleep 10
  done

  local local_size
  local_size=$(wc -c < "$tmp" | tr -d ' ')
  if [ "$local_size" != "$expected" ]; then
    echo "[$label] bad local chunk size for chunk $((i + 1)): expected $expected, got $local_size" >&2
    rm -f "$tmp"
    return 1
  fi

  for attempt in $(seq 1 10); do
    echo "[$label] chunk $((i + 1)) copying to CAIR (attempt $attempt/10)"
    if rsync --partial --append --progress -e "$RSYNC_RSH" "$tmp" "$REMOTE:$part.tmp" &&
      remote_cmd "mv '$part.tmp' '$part'"; then
      break
    fi
    remote_cmd "rm -f '$part.tmp'" || true
    if [ "$attempt" = 10 ]; then
      echo "[$label] failed to copy chunk $((i + 1)) after $attempt attempts" >&2
      rm -f "$tmp"
      return 1
    fi
    sleep 10
  done

  have=$(remote_size_or_zero "$part")
  if [ "$have" != "$expected" ]; then
    echo "[$label] bad chunk size for $part: expected $expected, got $have" >&2
    rm -f "$tmp"
    return 1
  fi
  rm -f "$tmp"
  echo "[$label] chunk $((i + 1)) complete"
}

stream_file_by_range() {
  local filename="$1"
  local url="$2"
  local size="$3"
  local remote_final="$REMOTE_MODEL_DIR/$filename"
  local remote_parts="$REMOTE_MODEL_DIR/.parts/$filename"
  local chunks=$(( (size + CHUNK_SIZE - 1) / CHUNK_SIZE ))
  local have

  have=$(remote_size_or_zero "$remote_final")
  if [ "$have" = "$size" ]; then
    echo "[$filename] already complete at $remote_final"
    return 0
  fi

  echo "[$filename] target: $remote_final"
  remote_cmd "mkdir -p '$remote_parts' '$REMOTE_MODEL_DIR'" || return 1

  local completed_manifest
  completed_manifest=$(mktemp "${TMPDIR:-/tmp}/${filename}.completed.XXXXXX")
  if ! remote_cmd "find '$remote_parts' -maxdepth 1 -type f ! -name '*.tmp' -printf '%f %s\n' 2>/dev/null" > "$completed_manifest"; then
    rm -f "$completed_manifest"
    echo "[$filename] failed to list completed remote chunks" >&2
    return 1
  fi

  local pids=()
  for ((i = 0; i < chunks; i++)); do
    local start=$(( i * CHUNK_SIZE ))
    local end=$(( start + CHUNK_SIZE - 1 ))
    if (( end >= size )); then
      end=$(( size - 1 ))
    fi

    local expected=$(( end - start + 1 ))
    local part_name
    part_name=$(printf "part_%04d" "$i")
    if grep -F -x -q "$part_name $expected" "$completed_manifest"; then
      echo "[$filename] chunk $((i + 1)) already complete ($expected bytes)"
      continue
    fi

    stream_one_chunk "$filename" "$url" "$remote_parts" "$i" "$start" "$end" "$expected" &
    pids+=("$!")
    if (( ${#pids[@]} >= PARALLEL_CHUNKS )); then
      if ! wait "${pids[0]}"; then
        rm -f "$completed_manifest"
        return 1
      fi
      if (( ${#pids[@]} > 1 )); then
        pids=("${pids[@]:1}")
      else
        pids=()
      fi
    fi
  done
  for pid in "${pids[@]+"${pids[@]}"}"; do
    if ! wait "$pid"; then
      rm -f "$completed_manifest"
      return 1
    fi
  done
  rm -f "$completed_manifest"

  echo "[$filename] concatenating chunks"
  remote_cmd "cat '$remote_parts'/part_* > '$remote_final.tmp' && mv '$remote_final.tmp' '$remote_final' && stat -c '%n %s bytes' '$remote_final'"
}

if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

while IFS=$'\t' read -r file url size; do
  stream_file_by_range "$file" "$url" "$size" || exit 1
done < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    manifest = json.load(f)

for item in manifest:
    print(f"{item['file']}\t{item['url']}\t{item['size']}")
PY
)

remote_cmd "cd '$REMOTE_MODEL_DIR' && ls -lh && test -f config.json && test -f tokenizer.json && test -f model.safetensors.index.json && test -f model-00004-of-00004.safetensors"
