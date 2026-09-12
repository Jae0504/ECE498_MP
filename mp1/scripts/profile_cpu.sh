#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${project_dir}/results/perf"
cpu_core="${CPU_CORE:-0}"
mkdir -p "${result_dir}"

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target cpu_bench -j "${BUILD_JOBS:-4}"
"${build_dir}/cpu_bench" --test >"${result_dir}/correctness.txt"
if command -v perf >/dev/null 2>&1; then
  perf --version >"${result_dir}/usr_bin_perf_check.txt" 2>&1 || true
fi

find_perf() {
  if command -v perf >/dev/null 2>&1 && perf --version >/dev/null 2>&1; then
    command -v perf
    return 0
  fi
  local candidate
  shopt -s nullglob
  local candidates=(/usr/lib/linux-tools/*/perf)
  shopt -u nullglob
  local sorted_candidates=()
  if (( ${#candidates[@]} > 0 )); then
    mapfile -t sorted_candidates < <(printf '%s\n' "${candidates[@]}" | sort -Vr)
  fi
  for candidate in "${sorted_candidates[@]}"; do
    if [[ -x "${candidate}" ]] && "${candidate}" --version >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! perf_bin="$(find_perf)"; then
  {
    printf 'No usable perf binary was found for kernel %s.\n' "$(uname -r)"
    printf 'The benchmark remains runnable; hardware counters were skipped.\n'
  } | tee "${result_dir}/PERF_UNAVAILABLE.txt"
  exit 0
fi

{
  printf 'perf_binary=%s\n' "${perf_bin}"
  "${perf_bin}" --version
  printf 'kernel=%s\n' "$(uname -r)"
  printf 'perf_event_paranoid='
  cat /proc/sys/kernel/perf_event_paranoid
} >"${result_dir}/perf_environment.txt"
"${perf_bin}" list >"${result_dir}/perf_list.txt" 2>&1

if ! "${perf_bin}" stat -e cycles,instructions -- true \
    >"${result_dir}/access_check.stdout.txt" \
    2>"${result_dir}/access_check.stderr.txt"; then
  {
    printf 'perf exists but hardware counters are inaccessible.\n'
    printf 'See access_check.stderr.txt for the exact error.\n'
  } | tee "${result_dir}/PERF_PERMISSION_LIMITATION.txt"
  exit 0
fi

runner=()
if command -v taskset >/dev/null 2>&1 && taskset -c "${cpu_core}" true 2>/dev/null; then
  runner=(taskset -c "${cpu_core}")
fi

event_check_dir="${result_dir}/event_checks"
mkdir -p "${event_check_dir}"

select_supported_events() {
  local category="$1"
  shift
  local supported=()
  local event
  for event in "$@"; do
    if "${perf_bin}" stat -e "${event}" -- true \
        >"${event_check_dir}/${category}_${event}.stdout.txt" \
        2>"${event_check_dir}/${category}_${event}.stderr.txt"; then
      supported+=("${event}")
    fi
  done
  local IFS=,
  printf '%s' "${supported[*]}"
}

core_events="$(select_supported_events core \
  cycles instructions branches branch-misses)"
cache_events="$(select_supported_events cache \
  cache-references cache-misses L1-dcache-loads L1-dcache-load-misses \
  LLC-loads LLC-load-misses)"
topdown_events="topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
{
  printf 'core_events=%s\n' "${core_events}"
  printf 'cache_events=%s\n' "${cache_events}"
  if "${perf_bin}" stat -e "${topdown_events}" -- true \
      >"${result_dir}/topdown_access_check.stdout.txt" \
      2>"${result_dir}/topdown_access_check.stderr.txt"; then
    printf 'topdown_events=%s\n' "${topdown_events}"
  else
    printf 'topdown_events=unsupported by this perf/CPU combination\n'
    "${perf_bin}" stat -a -C "${cpu_core}" -e "${topdown_events}" -- true \
      >"${result_dir}/topdown_systemwide_check.stdout.txt" \
      2>"${result_dir}/topdown_systemwide_check.stderr.txt" || true
    topdown_events=""
  fi
  if grep -q 'fp_arith_inst_retired' "${result_dir}/perf_list.txt"; then
    printf 'floating_point_events=listed; inspect perf_list.txt before selecting model-specific subevents\n'
  else
    printf 'floating_point_events=no named fp_arith_inst_retired event exposed\n'
  fi
  if grep -Eq 'stalled-cycles-(frontend|backend)' "${result_dir}/perf_list.txt"; then
    printf 'stalled_cycle_events=listed; inspect perf_list.txt\n'
  else
    printf 'stalled_cycle_events=not exposed by this perf/CPU combination\n'
  fi
} >"${result_dir}/selected_events.txt"

profile_one() {
  local label="$1"
  local event_group="$2"
  shift 2
  "${perf_bin}" stat -x ';' -o "${result_dir}/${label}.txt" \
    -e "${event_group}" -- "${runner[@]}" "${build_dir}/cpu_bench" \
    "$@" --skip-tests --no-save --format csv \
    >"${result_dir}/${label}.benchmark.csv" \
    2>"${result_dir}/${label}.stderr.txt"
}

for group in core cache; do
  if [[ "${group}" == "core" ]]; then
    events="${core_events}"
  else
    events="${cache_events}"
  fi
  if [[ -z "${events}" ]]; then
    continue
  fi
  profile_one "gemm_s512_${group}" "${events}" \
    --kernel gemm --seq 512 --warmup 1 --iterations 1
  profile_one "gemv_${group}" "${events}" \
    --kernel gemv --warmup 10 --iterations 200
  profile_one "attention_s128_${group}" "${events}" \
    --kernel attention --seq 128 --warmup 10 --iterations 500
done

if [[ -n "${topdown_events}" ]]; then
  profile_one "gemm_s512_topdown" "${topdown_events}" \
    --kernel gemm --seq 512 --warmup 1 --iterations 1 || true
  profile_one "gemv_topdown" "${topdown_events}" \
    --kernel gemv --warmup 10 --iterations 200 || true
  profile_one "attention_s128_topdown" "${topdown_events}" \
    --kernel attention --seq 128 --warmup 10 --iterations 500 || true
fi

python3 "${script_dir}/parse_perf.py" --input "${result_dir}" \
  --events-output "${result_dir}/events.csv" \
  --summary-output "${result_dir}/summary.csv"
printf 'Raw and derived perf results written to %s\n' "${result_dir}"
