#!/usr/bin/env bash
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
output="${1:-${project_dir}/results/environment.txt}"
mkdir -p "$(dirname -- "${output}")"

{
  printf 'inspection_time=%s\n' "$(date --iso-8601=seconds)"
  printf '\n[OS]\n'
  uname -a
  cat /etc/os-release
  printf '\n[CPU]\n'
  lscpu
  lscpu --caches
  printf '\n[MEMORY]\n'
  free -h
  printf '\n[NUMA]\n'
  if command -v numactl >/dev/null 2>&1; then numactl --hardware; else printf 'numactl unavailable\n'; fi
  printf '\n[TOOLS]\n'
  for tool in g++ cmake make python3 taskset perf valgrind cg_annotate; do
    printf '%-12s' "${tool}"
    if command -v "${tool}" >/dev/null 2>&1; then
      command -v "${tool}"
    else
      printf 'unavailable\n'
    fi
  done
  g++ --version 2>/dev/null | head -n 1
  cmake --version 2>/dev/null | head -n 1
  make --version 2>/dev/null | head -n 1
  python3 --version 2>/dev/null || true
  valgrind --version 2>/dev/null || true
  printf '\n[PERF PERMISSION]\n'
  printf 'perf_event_paranoid='
  if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
    cat /proc/sys/kernel/perf_event_paranoid
  else
    printf 'unavailable\n'
  fi
  if command -v perf >/dev/null 2>&1 && perf --version >/dev/null 2>&1; then
    perf --version
    perf stat -e cycles,instructions -- true || true
  else
    printf 'No usable perf command in PATH; this is not fatal for MP1.\n'
  fi
} >"${output}" 2>&1

printf 'Environment report written to %s\n' "${output}"
