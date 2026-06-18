#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/Users/nicholas/Documents/ICRA27 ROBOT}"
INTERVAL_SECONDS="${1:-30}"
PROGRESS_SCRIPT="$ROOT_DIR/test_files/geometry_affordance_probe/cair_setup_scripts/check_download_progress.sh"
MAIN_LOG="$ROOT_DIR/test_files/geometry_affordance_probe/agnostos_xicm_download_loop.log"
QWEN_LOG="$ROOT_DIR/test_files/geometry_affordance_probe/qwen25_download.log"

while true; do
  clear
  printf "CAIR AGNOSTOS / X-ICM progress\n"
  printf "Updated: %s\n" "$(date)"
  printf "Refresh interval: %ss\n\n" "$INTERVAL_SECONDS"

  "$PROGRESS_SCRIPT" 2>&1 | awk '
    /^[A-Z][a-z][a-z] / { show = 1 }
    show { print }
  '

  printf "\nActive local screen sessions:\n"
  screen -ls 2>&1 | sed "s/^/  /"

  printf "\nMain loop status:\n"
  if [ -f "$MAIN_LOG" ]; then
    tail -n 12 "$MAIN_LOG" | sed "s/^/  /"
  else
    printf "  no main loop log yet\n"
  fi

  printf "\nQwen relay status:\n"
  if [ -f "$QWEN_LOG" ]; then
    tail -n 12 "$QWEN_LOG" | sed "s/^/  /"
  else
    printf "  no Qwen log yet\n"
  fi

  printf "\nPress Ctrl-C to stop this viewer. The downloads keep running in screen.\n"
  sleep "$INTERVAL_SECONDS"
done
