#!/usr/bin/env bash
set -u

REMOTE="${REMOTE:-cair}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/yf23/datasets/ICRA27-ROBOT}"
REMOTE_CHECKPOINT_ROOT="${REMOTE_CHECKPOINT_ROOT:-/data/yf23/checkpoints/ICRA27-ROBOT}"
REMOTE_XICM_ROOT="${REMOTE_XICM_ROOT:-/data/yf23/projects/ICRA27-ROBOT/X-ICM}"
CHUNK_SIZE="${CHUNK_SIZE:-67108864}"
PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-2}"
SSH_OPTS=(-n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12)
RSYNC_RSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12"

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
  local label="$1"
  local url="$2"
  local size="$3"
  local remote_final="$4"

  local remote_parts="${remote_final}.parts"
  local chunks=$(( (size + CHUNK_SIZE - 1) / CHUNK_SIZE ))

  echo "[$label] target: $remote_final"
  if ! remote_cmd "mkdir -p '$remote_parts' '$(dirname "$remote_final")'"; then
    echo "[$label] failed to create remote directory for $remote_final" >&2
    return 1
  fi

  local completed_manifest
  completed_manifest=$(mktemp "${TMPDIR:-/tmp}/${label}.completed.XXXXXX")
  if ! remote_cmd "find '$remote_parts' -maxdepth 1 -type f ! -name '*.tmp' -printf '%f %s\n' 2>/dev/null" > "$completed_manifest"; then
    rm -f "$completed_manifest"
    echo "[$label] failed to list completed remote chunks" >&2
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
      echo "[$label] chunk $((i + 1)) already complete ($expected bytes)"
      continue
    fi

    stream_one_chunk "$label" "$url" "$remote_parts" "$i" "$start" "$end" "$expected" &
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

  echo "[$label] concatenating chunks"
  remote_cmd "cat '$remote_parts'/part_* > '$remote_final.tmp' && mv '$remote_final.tmp' '$remote_final' && stat -c '%n %s bytes' '$remote_final'"
}

UNSEEN_URL="https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/unseen_tasks.tar"
UNSEEN_SIZE=20184780800
MODEL_URL="https://huggingface.co/Jiaming2472/X-ICM/resolve/main/dynamics_diffusion.tar"
MODEL_SIZE=10436526080

DOWNLOAD_TARGET="${DOWNLOAD_TARGET:-all}"

case "$DOWNLOAD_TARGET" in
  all)
    stream_file_by_range "unseen_tasks" "$UNSEEN_URL" "$UNSEEN_SIZE" "$REMOTE_DATA_ROOT/unseen_tasks.tar" || exit 1
    stream_file_by_range "dynamics_diffusion" "$MODEL_URL" "$MODEL_SIZE" "$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion.tar" || exit 1
    ;;
  unseen)
    stream_file_by_range "unseen_tasks" "$UNSEEN_URL" "$UNSEEN_SIZE" "$REMOTE_DATA_ROOT/unseen_tasks.tar" || exit 1
    ;;
  model|dynamics|dynamics_diffusion)
    stream_file_by_range "dynamics_diffusion" "$MODEL_URL" "$MODEL_SIZE" "$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion.tar" || exit 1
    ;;
  *)
    echo "Unknown DOWNLOAD_TARGET=$DOWNLOAD_TARGET; expected all, unseen, or model." >&2
    exit 2
    ;;
esac

if ! remote_cmd "set -e
mkdir -p '$REMOTE_DATA_ROOT/unseen_tasks_temp' '$REMOTE_DATA_ROOT/unseen_tasks' '$REMOTE_CHECKPOINT_ROOT' '$REMOTE_XICM_ROOT/data'
if [ '$DOWNLOAD_TARGET' = 'all' ] || [ '$DOWNLOAD_TARGET' = 'unseen' ]; then
  rm -rf '$REMOTE_DATA_ROOT/unseen_tasks_temp'
  mkdir -p '$REMOTE_DATA_ROOT/unseen_tasks_temp'
  tar -xf '$REMOTE_DATA_ROOT/unseen_tasks.tar' -C '$REMOTE_DATA_ROOT/unseen_tasks_temp'
  find '$REMOTE_DATA_ROOT/unseen_tasks' -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  if [ -d '$REMOTE_DATA_ROOT/unseen_tasks_temp/unseen_tasks/test' ]; then
    mv '$REMOTE_DATA_ROOT/unseen_tasks_temp/unseen_tasks/test/'* '$REMOTE_DATA_ROOT/unseen_tasks/'
  elif [ -d '$REMOTE_DATA_ROOT/unseen_tasks_temp/unseen_tasks' ]; then
    mv '$REMOTE_DATA_ROOT/unseen_tasks_temp/unseen_tasks/'* '$REMOTE_DATA_ROOT/unseen_tasks/'
  else
    mv '$REMOTE_DATA_ROOT/unseen_tasks_temp/'* '$REMOTE_DATA_ROOT/unseen_tasks/'
  fi
  rm -rf '$REMOTE_DATA_ROOT/unseen_tasks_temp'
fi

if [ '$DOWNLOAD_TARGET' = 'all' ] || [ '$DOWNLOAD_TARGET' = 'model' ] || [ '$DOWNLOAD_TARGET' = 'dynamics' ] || [ '$DOWNLOAD_TARGET' = 'dynamics_diffusion' ]; then
  rm -rf '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion'
  tar -xf '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion.tar' -C '$REMOTE_CHECKPOINT_ROOT'
fi

rm -rf '$REMOTE_DATA_ROOT/xicm_seen_tasks_linkroot' '$REMOTE_DATA_ROOT/xicm_unseen_tasks_linkroot'
mkdir -p '$REMOTE_DATA_ROOT/xicm_seen_tasks_linkroot' '$REMOTE_DATA_ROOT/xicm_unseen_tasks_linkroot'
ln -sfn '$REMOTE_DATA_ROOT/seen_tasks' '$REMOTE_DATA_ROOT/xicm_seen_tasks_linkroot/train'
[ -d '$REMOTE_DATA_ROOT/unseen_tasks' ] && ln -sfn '$REMOTE_DATA_ROOT/unseen_tasks' '$REMOTE_DATA_ROOT/xicm_unseen_tasks_linkroot/test'
ln -sfn '$REMOTE_DATA_ROOT/xicm_seen_tasks_linkroot' '$REMOTE_XICM_ROOT/data/seen_tasks'
[ -d '$REMOTE_DATA_ROOT/unseen_tasks' ] && ln -sfn '$REMOTE_DATA_ROOT/xicm_unseen_tasks_linkroot' '$REMOTE_XICM_ROOT/data/unseen_tasks'
[ -d '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion' ] && ln -sfn '$REMOTE_CHECKPOINT_ROOT/dynamics_diffusion' '$REMOTE_XICM_ROOT/data/dynamics_diffusion'

echo seen_count=\$(find -L '$REMOTE_XICM_ROOT/data/seen_tasks/train' -mindepth 1 -maxdepth 1 -type d | wc -l)
echo unseen_count=\$(find -L '$REMOTE_XICM_ROOT/data/unseen_tasks/test' -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
ls -la '$REMOTE_XICM_ROOT/data'
"; then
  echo "final extraction/linking failed" >&2
  exit 1
fi
