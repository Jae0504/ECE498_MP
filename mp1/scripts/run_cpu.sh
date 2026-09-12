#!/usr/bin/env bash
# Legacy single-thread batch. For 1/2/4/8 workers use run_thread_scaling.sh.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${project_dir}/results"
cpu_core="${CPU_CORE:-0}"
include_large="${INCLUDE_LARGE:-0}"

mkdir -p "${result_dir}"
cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target cpu_bench roofline_bench -j "${BUILD_JOBS:-4}"

runner=()
affinity_applied=no
if command -v taskset >/dev/null 2>&1 && taskset -c "${cpu_core}" true 2>/dev/null; then
  runner=(taskset -c "${cpu_core}")
  affinity_applied=yes
fi

cache_flush_mib="${CACHE_FLUSH_MIB:-}"
if [[ -z "${cache_flush_mib}" ]]; then
  cache_size_file="/sys/devices/system/cpu/cpu${cpu_core}/cache/index3/size"
  if [[ -r "${cache_size_file}" ]]; then
    cache_size="$(<"${cache_size_file}")"
    case "${cache_size}" in
      *K) cache_kib="${cache_size%K}" ;;
      *M) cache_kib="$(( ${cache_size%M} * 1024 ))" ;;
      *) cache_kib=0 ;;
    esac
    # Two times the per-core last-level-cache capacity robustly displaces the
    # 46 MiB weight matrix even on this server's unusually large LLC.
    cache_flush_mib="$(( (cache_kib * 2 + 1023) / 1024 ))"
  else
    cache_flush_mib=256
  fi
  if (( cache_flush_mib < 64 )); then cache_flush_mib=64; fi
  if (( cache_flush_mib > 1024 )); then cache_flush_mib=1024; fi
fi
printf 'cpu_core=%s\naffinity_applied=%s\ncache_flush_mib=%s\ninclude_large=%s\n' \
  "${cpu_core}" "${affinity_applied}" "${cache_flush_mib}" "${include_large}" \
  >"${result_dir}/cache_experiment.txt"

"${runner[@]}" "${build_dir}/cpu_bench" --test | tee "${result_dir}/correctness.txt"

# A short discarded compute run reduces the large cold-governor effect observed
# on ondemand systems. Kernel-specific warm-ups still occur below.
"${runner[@]}" "${build_dir}/roofline_bench" --ceiling compute \
  --no-save --fma-repeats 10000000 --warmup 0 --iterations 1 >/dev/null

kernel_csv="${result_dir}/kernel_results.csv"
"${runner[@]}" "${build_dir}/cpu_bench" --kernel gemm --seq 128 \
  --warmup 2 --iterations 5 --skip-tests --no-save --format csv >"${kernel_csv}"
"${runner[@]}" "${build_dir}/cpu_bench" --kernel gemm --seq 512 \
  --warmup 2 --iterations 3 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
if [[ "${include_large}" == "1" ]]; then
  "${runner[@]}" "${build_dir}/cpu_bench" --kernel gemm --seq 2048 \
    --warmup 1 --iterations 3 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
fi
"${runner[@]}" "${build_dir}/cpu_bench" --kernel gemv \
  --warmup 3 --iterations 9 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
if (( cache_flush_mib > 0 )); then
  "${runner[@]}" "${build_dir}/cpu_bench" --kernel gemv \
    --warmup 1 --iterations 5 --flush-cache-mib "${cache_flush_mib}" \
    --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
fi
"${runner[@]}" "${build_dir}/cpu_bench" --kernel attention --seq 128 \
  --warmup 3 --iterations 9 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
"${runner[@]}" "${build_dir}/cpu_bench" --kernel attention --seq 512 \
  --warmup 2 --iterations 5 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
if [[ "${include_large}" == "1" ]]; then
  "${runner[@]}" "${build_dir}/cpu_bench" --kernel attention --seq 2048 \
    --warmup 2 --iterations 5 --skip-tests --no-save --format csv --no-header >>"${kernel_csv}"
fi

ceiling_csv="${result_dir}/ceilings.csv"
"${runner[@]}" "${build_dir}/roofline_bench" --ceiling all \
  --warmup 1 --iterations 5 --no-save --format csv >"${ceiling_csv}"

python3 "${script_dir}/generate_roofline.py" \
  --kernels "${kernel_csv}" --ceilings "${ceiling_csv}" \
  --output "${result_dir}/roofline.csv"

printf 'CPU results written to %s\n' "${result_dir}"
