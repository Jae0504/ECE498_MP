#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${RESULT_DIR:-${project_dir}/results}"
include_large="${INCLUDE_LARGE:-0}"
source "${script_dir}/cuda_tools.sh"

mkdir -p "${result_dir}"
result_csv="${result_dir}/kernel_results.csv"
if [[ -e "${result_csv}" ]]; then
  printf 'Results already exist: %s\nChoose a fresh RESULT_DIR for this experiment.\n' \
    "${result_csv}" >&2
  exit 1
fi
pending_csv=""
stage="tool_discovery"
finish_run() {
  local status=$?
  if [[ -n "${pending_csv}" ]]; then
    rm -f "${pending_csv}"
  fi
  if [[ ${status} -ne 0 && ${status} -ne 77 ]]; then
    printf 'status=failure\nstage=%s\nexit_status=%s\n' "${stage}" "${status}" \
      >"${result_dir}/runtime_status.txt"
  fi
}
trap finish_run EXIT
printf 'status=running\n' >"${result_dir}/runtime_status.txt"
{
  printf 'measurement=cuda_event_unbatched\n'
  printf 'gemm_gemv_sample=one_kernel_launch\n'
  printf 'attention_sample=one_qk_softmax_pv_sequence\n'
  printf 'include_large=%s\n' "${include_large}"
  printf 'attention_s128_iterations=101\n'
  printf 'attention_s512_iterations=21\n'
} >"${result_dir}/experiment.txt"

if ! nvcc_bin="$(discover_cuda_tool nvcc NVCC_BIN)"; then
  printf 'No nvcc supports CUDA_ARCHITECTURES=%s. Set NVCC_BIN or CUDA_HOME (DGX Spark: CUDA 13+).\n' \
    "${CUDA_ARCHITECTURES:-121}" \
    | tee "${result_dir}/runtime_status.txt"
  exit 1
fi

cmake_args=(
  -S "${project_dir}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-121}"
  -DCMAKE_CUDA_COMPILER="${nvcc_bin}"
)
stage="configure"
cmake "${cmake_args[@]}"
stage="build"
if ! cmake --build "${build_dir}" --target gpu_bench -j "${BUILD_JOBS:-4}"; then
  printf 'gpu_bench could not be built; see compiler output above.\n' \
    | tee "${result_dir}/runtime_status.txt"
  exit 1
fi

bash "${script_dir}/inspect_environment.sh" "${result_dir}/environment.txt"

run_correctness_suite() {
  local label="$1"
  shift
  {
    printf '\n[%s]\n' "${label}"
    "${build_dir}/gpu_bench" "$@"
  } >>"${result_dir}/correctness.txt" 2>&1
}

: >"${result_dir}/correctness.txt"
stage="correctness"
set +e
run_correctness_suite fast_and_boundary --test
test_status=$?
set -e
if [[ ${test_status} -eq 77 ]]; then
  {
    printf 'status=unavailable\n'
    printf 'GPU runtime unavailable; CUDA compile succeeded, execution deferred.\n'
    cat "${result_dir}/correctness.txt"
  } | tee "${result_dir}/runtime_status.txt"
  exit 77
fi
if [[ ${test_status} -ne 0 ]]; then
  cat "${result_dir}/correctness.txt"
  exit "${test_status}"
fi

if ! run_correctness_suite required_sizes --test-required; then
  cat "${result_dir}/correctness.txt"
  exit 1
fi

stage="timing"
pending_csv="$(mktemp "${result_dir}/.kernel_results.XXXXXX")"
"${build_dir}/gpu_bench" --kernel gemm --seq 128 --warmup 2 --iterations 5 \
  --skip-tests --format csv >"${pending_csv}"
"${build_dir}/gpu_bench" --kernel gemm --seq 512 --warmup 2 --iterations 5 \
  --skip-tests --format csv --no-header >>"${pending_csv}"
if [[ "${include_large}" == "1" ]]; then
  "${build_dir}/gpu_bench" --kernel gemm --seq 2048 --warmup 2 --iterations 5 \
    --skip-tests --format csv --no-header >>"${pending_csv}"
fi
"${build_dir}/gpu_bench" --kernel gemv --warmup 5 --iterations 21 \
  --skip-tests --format csv --no-header >>"${pending_csv}"
"${build_dir}/gpu_bench" --kernel attention --seq 128 --warmup 10 \
  --iterations 101 --skip-tests --format csv --no-header >>"${pending_csv}"
"${build_dir}/gpu_bench" --kernel attention --seq 512 --warmup 5 \
  --iterations 21 --skip-tests --format csv --no-header >>"${pending_csv}"
if [[ "${include_large}" == "1" ]]; then
  "${build_dir}/gpu_bench" --kernel attention --seq 2048 --warmup 2 \
    --iterations 9 --skip-tests --format csv --no-header >>"${pending_csv}"
fi
mv "${pending_csv}" "${result_csv}"
pending_csv=""
printf 'status=success\nresults=%s\n' "${result_csv}" \
  >"${result_dir}/runtime_status.txt"
printf 'GPU results written to %s\n' "${result_dir}"
