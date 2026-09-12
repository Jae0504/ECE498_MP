#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${CACHEGRIND_BUILD_DIR:-${project_dir}/build-cachegrind}"
result_dir="${project_dir}/results/cachegrind"
mkdir -p "${result_dir}"

if ! command -v valgrind >/dev/null 2>&1 || \
   ! command -v cg_annotate >/dev/null 2>&1; then
  printf '%s\n' \
    'Cachegrind is unavailable. Native timing, analytical AI, and the Roofline' \
    'remain available through cpu_bench and roofline_bench; see README.md.' \
    | tee "${result_dir}/CACHEGRIND_UNAVAILABLE.txt"
  exit 0
fi

# A fixed cache model makes the exercise reproducible across student machines.
# It is intentionally a simulation, not a claim about the host cache geometry.
cachegrind_i1="${CACHEGRIND_I1:-32768,8,64}"
cachegrind_d1="${CACHEGRIND_D1:-32768,8,64}"
cachegrind_ll="${CACHEGRIND_LL:-67108864,16,64}"
flush_mib="${CACHEGRIND_FLUSH_MIB:-80}"
full_profile="${CACHEGRIND_FULL:-0}"

cmake -S "${project_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release -DCPU_NATIVE=OFF
cmake --build "${build_dir}" --target cpu_bench -j "${BUILD_JOBS:-4}"
"${build_dir}/cpu_bench" --test >"${result_dir}/correctness.txt"

{
  valgrind --version
  printf 'sudo_used=no\n'
  printf 'binary_build=Release -O3, CPU_NATIVE=OFF\n'
  printf 'I1=%s\nD1=%s\nLL=%s\n' \
    "${cachegrind_i1}" "${cachegrind_d1}" "${cachegrind_ll}"
  printf 'cold_flush_mib=%s\n' "${flush_mib}"
  printf 'full_profile=%s\n' "${full_profile}"
  printf '%s\n' \
    'Cachegrind timing/GFLOP/s is invalid; use native results/kernel_results.csv.' \
    'DLmr/DLmw are simulated last-level-cache misses, not measured DRAM traffic.'
} >"${result_dir}/environment.txt"

profile_one() {
  local label="$1"
  shift
  valgrind --vgdb=no --tool=cachegrind --cache-sim=yes --branch-sim=no \
    --I1="${cachegrind_i1}" --D1="${cachegrind_d1}" --LL="${cachegrind_ll}" \
    --cachegrind-out-file="${result_dir}/${label}.out" \
    "${build_dir}/cpu_bench" "$@" --warmup 0 --iterations 1 \
    --skip-tests --no-save --format csv \
    >"${result_dir}/${label}.instrumented_timing_DO_NOT_USE.csv" \
    2>"${result_dir}/${label}.log"
  cg_annotate --show=Ir,Dr,Dw,D1mr,D1mw,DLmr,DLmw --sort=D1mr \
    --threshold=0.1 "${result_dir}/${label}.out" \
    >"${result_dir}/${label}.annotated.txt"
}

profile_files=()
profile_one teaching_gemv_cold \
  --kernel gemv --flush-cache-mib "${flush_mib}"
profile_files+=(--include-profile teaching_gemv_cold.out)
profile_one teaching_gemm_s128_cold \
  --kernel gemm --seq 128 --flush-cache-mib "${flush_mib}"
profile_files+=(--include-profile teaching_gemm_s128_cold.out)
if [[ "${full_profile}" == "1" ]]; then
  profile_one teaching_gemm_s512_cold \
    --kernel gemm --seq 512 --flush-cache-mib "${flush_mib}"
  profile_files+=(--include-profile teaching_gemm_s512_cold.out)
fi
profile_one teaching_attention_s128 --kernel attention --seq 128
profile_files+=(--include-profile teaching_attention_s128.out)
profile_one teaching_attention_s512 --kernel attention --seq 512
profile_files+=(--include-profile teaching_attention_s512.out)

python3 "${script_dir}/parse_cachegrind.py" \
  --input "${result_dir}" "${profile_files[@]}" \
  --output "${result_dir}/summary.csv"
printf 'Cachegrind simulation results written to %s\n' "${result_dir}"
