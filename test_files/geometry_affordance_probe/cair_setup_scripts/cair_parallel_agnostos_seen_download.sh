#!/usr/bin/env bash
set -euo pipefail

DATA=/data/yf23/datasets/ICRA27-ROBOT
PROBE=/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe
LOG="$PROBE/download_seen_parallel.log"
PARTS=(aa ab ac ad ae)

mkdir -p "$DATA" "$PROBE"
cd "$DATA"

download_part() {
  local part="$1"
  local out="seen_tasks.part_${part}"
  local url="https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/${out}"
  local part_log="$PROBE/download_${out}.log"

  while true; do
    {
      echo "[$(date)] downloading/resuming $out"
      curl -C - -L --retry 100 --retry-all-errors --retry-delay 20 --connect-timeout 120 \
        -o "$out" "$url"
      echo "[$(date)] finished $out"
    } >> "$part_log" 2>&1 && break

    echo "[$(date)] $out failed; retrying in 60s" >> "$part_log"
    sleep 60
  done
}

{
  echo "[$(date)] Parallel AGNOSTOS seen_tasks download starting in $DATA"

  for part in "${PARTS[@]}"; do
    download_part "$part" &
  done
  wait

  echo "[$(date)] All parts downloaded; combining"
  cat seen_tasks.part_aa seen_tasks.part_ab seen_tasks.part_ac seen_tasks.part_ad seen_tasks.part_ae > seen_tasks.tar
  if command -v md5sum >/dev/null 2>&1; then
    md5sum seen_tasks.tar | tee seen_tasks.tar.md5
  fi

  echo "[$(date)] Extracting seen_tasks.tar"
  rm -rf seen_tasks_extract_tmp
  mkdir -p seen_tasks_extract_tmp
  tar -xf seen_tasks.tar -C seen_tasks_extract_tmp
  rm -rf seen_tasks
  mkdir -p seen_tasks

  if [ -d seen_tasks_extract_tmp/seen_tasks/train ]; then
    mv seen_tasks_extract_tmp/seen_tasks/train/* seen_tasks/
  elif [ -d seen_tasks_extract_tmp/seen_tasks ]; then
    mv seen_tasks_extract_tmp/seen_tasks/* seen_tasks/
  else
    mv seen_tasks_extract_tmp/* seen_tasks/
  fi

  rm -rf seen_tasks_extract_tmp
  echo "[$(date)] Done: $DATA/seen_tasks"
} 2>&1 | tee -a "$LOG"
