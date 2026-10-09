#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${RESULT_DIR:-${project_dir}/results}/sanitizer"
source "${script_dir}/cuda_tools.sh"
mkdir -p "${result_dir}"
status_file="${result_dir}/status.txt"
printf 'validation_time=%s\n' "$(date --iso-8601=seconds)" >"${status_file}"
finish_validation() {
  local status=$?
  if [[ ${status} -ne 0 && ${status} -ne 77 ]]; then
    printf 'status=failure\nexit_status=%s\n' "${status}" >>"${status_file}"
  fi
}
trap finish_validation EXIT
rm -f "${result_dir}/version.txt" "${result_dir}/memcheck.txt" \
  "${result_dir}/initcheck.txt" "${result_dir}/synccheck.txt" \
  "${result_dir}/racecheck.txt"

if ! nvcc_bin="$(discover_cuda_tool nvcc NVCC_BIN)"; then
  printf 'No nvcc supports CUDA_ARCHITECTURES=%s. Set NVCC_BIN or CUDA_HOME (DGX Spark: CUDA 13+).\n' \
    "${CUDA_ARCHITECTURES:-121}" >&2
  exit 1
fi
cmake -S "${project_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-121}" \
  -DCMAKE_CUDA_COMPILER="${nvcc_bin}"
cmake --build "${build_dir}" --target gpu_bench -j "${BUILD_JOBS:-4}"

set +e
"${build_dir}/gpu_bench" --test >"${result_dir}/preflight.txt" 2>&1
test_status=$?
set -e
if [[ ${test_status} -eq 77 ]]; then
  printf 'status=unavailable\nGPU runtime unavailable; sanitizer checks deferred.\n' \
    | tee -a "${status_file}"
  exit 77
elif [[ ${test_status} -ne 0 ]]; then
  cat "${result_dir}/preflight.txt" >&2
  exit "${test_status}"
fi
if ! sanitizer_bin="$(discover_cuda_tool compute-sanitizer COMPUTE_SANITIZER_BIN)"; then
  printf 'tool_path=unavailable\nstatus=failure\n' | tee -a "${status_file}"
  printf 'No runnable Compute Sanitizer was found. Set COMPUTE_SANITIZER_BIN.\n' >&2
  exit 1
fi
printf 'tool_path=%s\n' "${sanitizer_bin}" >>"${status_file}"
write_tool_version "${sanitizer_bin}" "${result_dir}/version.txt"

overall_status=0
for tool in memcheck initcheck synccheck racecheck; do
  output_file="${result_dir}/${tool}.txt"
  set +e
  "${sanitizer_bin}" --tool "${tool}" --error-exitcode=1 \
    "${build_dir}/gpu_bench" --test >"${output_file}" 2>&1
  tool_status=$?
  set -e
  printf '%s_exit_status=%s\n' "${tool}" "${tool_status}" \
    >>"${status_file}"
  summary_status=0
  if [[ "${tool}" == "racecheck" ]]; then
    grep -Eq 'RACECHECK SUMMARY: 0 hazards.*0 errors, 0 warnings' \
      "${output_file}" || summary_status=1
  else
    grep -q 'ERROR SUMMARY: 0 errors' "${output_file}" || summary_status=1
  fi
  if [[ ${summary_status} -eq 0 ]]; then
    printf '%s_summary=success\n' "${tool}" >>"${status_file}"
  else
    printf '%s_summary=failure\n' "${tool}" >>"${status_file}"
  fi
  if [[ ${tool_status} -ne 0 || ${summary_status} -ne 0 ]]; then
    overall_status=1
    printf '%s failed; see %s\n' "${tool}" "${output_file}" >&2
  fi
done

if [[ ${overall_status} -ne 0 ]]; then
  printf 'status=failure\n' >>"${status_file}"
  exit 1
fi
printf 'status=success\n' >>"${status_file}"
printf 'All Compute Sanitizer checks passed. Results: %s\n' "${result_dir}"
