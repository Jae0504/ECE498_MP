#!/usr/bin/env bash
# Run one fixed-size problem per sample, varying its OpenMP worker count.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${RESULT_DIR:-${project_dir}/results/thread_scaling}"
read -r -a counts <<< "${THREAD_COUNTS:-1 2 4 8}"
read -r -a sequences <<< "${SEQUENCES:-128 256 512}"
warmups="${WARMUPS:-2}"
iterations="${ITERATIONS:-5}"
flush_mib="${CACHE_FLUSH_MIB:-1024}"
export OMP_DYNAMIC=false
export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

roof_threads=0
for threads in "${counts[@]}"; do
  if [[ ! "${threads}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid THREAD_COUNTS entry: ${threads}" >&2
    exit 1
  fi
  if (( threads > roof_threads )); then roof_threads="${threads}"; fi
done
if (( roof_threads == 0 || ${#sequences[@]} == 0 )); then
  echo "THREAD_COUNTS and SEQUENCES must be nonempty." >&2
  exit 1
fi
if [[ ! -x "${build_dir}/cpu_bench" || ! -x "${build_dir}/roofline_bench" ]]; then
  echo "Build cpu_bench and roofline_bench first; see mp1/README.md." >&2
  exit 1
fi
mkdir -p "${result_dir}"
for threads in "${counts[@]}"; do
  "${build_dir}/cpu_bench" --test --threads "${threads}" \
    >"${result_dir}/correctness_t${threads}.txt"
done

ceiling_args=(--warmup "${warmups}" --iterations "${iterations}"
  --roof-threads "${roof_threads}" --output "${result_dir}/ceilings.csv"
  --kernels "${result_dir}/kernel_results.csv")
measure_ceilings() {
  local fma_args=(--compute-mode register --fma-repeats "${FMA_REPEATS:-20000000}")
  if [[ -n "${FMA_ACCUMULATORS:-}" ]]; then fma_args+=(--accumulators "${FMA_ACCUMULATORS}"); fi
  "${build_dir}/roofline_bench" --ceiling bandwidth --threads "$1" \
    --stream-elements "${STREAM_ELEMENTS:-67108864}" \
    --read-elements "${READ_ELEMENTS:-201326592}" "${ceiling_args[@]}"
  "${build_dir}/roofline_bench" --ceiling compute --threads "$1" \
    "${fma_args[@]}" \
    "${ceiling_args[@]}"
}

# Establish the common skeleton first. Other thread counts are kept in the
# ceiling CSV for scaling analysis without replacing the common roof.
measure_ceilings "${roof_threads}"
for threads in "${counts[@]}"; do
  if (( threads != roof_threads )); then measure_ceilings "${threads}"; fi
  kernel_args=(--threads "${threads}" --roof-threads "${roof_threads}"
    --warmup "${warmups}" --iterations "${iterations}"
    --output "${result_dir}/kernel_results.csv" --ceilings "${result_dir}/ceilings.csv")
  for sequence in "${sequences[@]}"; do
    "${build_dir}/cpu_bench" --kernel gemm --seq "${sequence}" "${kernel_args[@]}"
    "${build_dir}/cpu_bench" --kernel attention --seq "${sequence}" "${kernel_args[@]}"
  done
  "${build_dir}/cpu_bench" --kernel gemv "${kernel_args[@]}"
  "${build_dir}/cpu_bench" --kernel gemv --flush-cache-mib "${flush_mib}" "${kernel_args[@]}"
done
printf 'Thread-scaling CSV and PNG/PDF figures: %s\n' "${result_dir}"
