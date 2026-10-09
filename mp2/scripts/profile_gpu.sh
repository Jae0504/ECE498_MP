#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${RESULT_DIR:-${project_dir}/results}/profile"
source "${script_dir}/cuda_tools.sh"
mkdir -p "${result_dir}"

# Markers and reports describe only this invocation. The *_actual names are
# removed for compatibility with the pre-release instructor script.
stale_outputs=(
  "${result_dir}/NCU_UNAVAILABLE.txt"
  "${result_dir}/NSYS_UNAVAILABLE.txt"
  "${result_dir}/PROFILE_DEFERRED.txt"
  "${result_dir}/ncu_version.txt"
  "${result_dir}/ncu_sets.txt"
  "${result_dir}/ncu_sections.txt"
  "${result_dir}/ncu_selected_sections.txt"
  "${result_dir}/ncu_relevant_metrics.txt"
  "${result_dir}/ncu_gemm_s512.ncu-rep"
  "${result_dir}/ncu_gemm_s512.stdout.txt"
  "${result_dir}/ncu_gemm_s512.stderr.txt"
  "${result_dir}/ncu_gemm_s512_actual.ncu-rep"
  "${result_dir}/ncu_gemv.ncu-rep"
  "${result_dir}/ncu_gemv.stdout.txt"
  "${result_dir}/ncu_gemv.stderr.txt"
  "${result_dir}/ncu_gemv_actual.ncu-rep"
  "${result_dir}/ncu_attention_s128.ncu-rep"
  "${result_dir}/ncu_attention_s128.stdout.txt"
  "${result_dir}/ncu_attention_s128.stderr.txt"
  "${result_dir}/ncu_attention_s128_actual.ncu-rep"
  "${result_dir}/ncu_attention_s512_diagnostic.ncu-rep"
  "${result_dir}/ncu_attention_s512_diagnostic.stdout.txt"
  "${result_dir}/ncu_attention_s512_diagnostic.stderr.txt"
  "${result_dir}/nsys_version.txt"
  "${result_dir}/nsys_attention_s128.nsys-rep"
  "${result_dir}/nsys_attention_s128.sqlite"
  "${result_dir}/nsys.stdout.txt"
  "${result_dir}/nsys.stderr.txt"
)
rm -f "${stale_outputs[@]}"
: >"${result_dir}/ncu_status.txt"
: >"${result_dir}/nsys_status.txt"
printf 'profile_time=%s\n' "$(date --iso-8601=seconds)" \
  >>"${result_dir}/ncu_status.txt"
printf 'profile_time=%s\n' "$(date --iso-8601=seconds)" \
  >>"${result_dir}/nsys_status.txt"

if [[ "${SKIP_BASELINE_RUN:-0}" != "1" ]]; then
  set +e
  bash "${script_dir}/run_gpu.sh"
  test_status=$?
  set -e
else
  if [[ ! -x "${build_dir}/gpu_bench" ]]; then
    printf 'gpu_bench is missing; run run_gpu.sh or unset SKIP_BASELINE_RUN.\n' >&2
    exit 1
  fi
  # Probe the current runtime rather than trusting a previous run's status.
  set +e
  "${build_dir}/gpu_bench" --test >"${result_dir}/preflight.txt" 2>&1
  test_status=$?
  set -e
fi
if [[ ${test_status} -eq 77 ]]; then
  printf 'GPU profiling skipped because CUDA runtime has no visible device.\n' \
    | tee "${result_dir}/PROFILE_DEFERRED.txt"
  printf 'status=unavailable\n' >>"${result_dir}/ncu_status.txt"
  printf 'status=unavailable\n' >>"${result_dir}/nsys_status.txt"
  exit 77
elif [[ ${test_status} -ne 0 ]]; then
  printf 'status=failure\npreflight_exit_status=%s\n' "${test_status}" \
    | tee -a "${result_dir}/ncu_status.txt" >>"${result_dir}/nsys_status.txt"
  exit "${test_status}"
fi

overall_status=0
if ncu_bin="$(discover_cuda_tool ncu NCU_BIN)"; then
  ncu_status=0
  printf 'tool_path=%s\ndiscovery=success\n' "${ncu_bin}" \
    >>"${result_dir}/ncu_status.txt"
  write_tool_version "${ncu_bin}" "${result_dir}/ncu_version.txt"
  "${ncu_bin}" --list-sets >"${result_dir}/ncu_sets.txt" 2>&1 || true
  "${ncu_bin}" --list-sections >"${result_dir}/ncu_sections.txt" 2>&1 || true

  ncu_collection_args=()
  : >"${result_dir}/ncu_selected_sections.txt"
  for section in LaunchStats Occupancy SpeedOfLight \
      ComputeWorkloadAnalysis MemoryWorkloadAnalysis; do
    if grep -q "^${section}[[:space:]]" "${result_dir}/ncu_sections.txt"; then
      ncu_collection_args+=(--section "${section}")
      printf '%s\n' "${section}" >>"${result_dir}/ncu_selected_sections.txt"
    fi
  done
  if (( ${#ncu_collection_args[@]} == 0 )); then
    ncu_collection_args=(--set basic)
    printf 'basic set fallback\n' >>"${result_dir}/ncu_selected_sections.txt"
  fi

  ncu_profile() {
    local label="$1"
    shift
    local report="${result_dir}/${label}.ncu-rep"
    local stdout_file="${result_dir}/${label}.stdout.txt"
    local stderr_file="${result_dir}/${label}.stderr.txt"
    rm -f "${report}" "${stdout_file}" "${stderr_file}"
    set +e
    "${ncu_bin}" --target-processes all "${ncu_collection_args[@]}" \
      --force-overwrite -o "${result_dir}/${label}" \
      "${build_dir}/gpu_bench" "$@" --warmup 0 --iterations 1 \
      --skip-tests >"${stdout_file}" 2>"${stderr_file}"
    local status=$?
    set -e
    printf '%s_exit_status=%s\n' "${label}" "${status}" \
      >>"${result_dir}/ncu_status.txt"
    if [[ -s "${report}" ]]; then
      printf '%s_report=generated\n' "${label}" \
        >>"${result_dir}/ncu_status.txt"
    else
      printf '%s_report=missing\n' "${label}" \
        >>"${result_dir}/ncu_status.txt"
    fi
    if [[ ${status} -ne 0 || ! -s "${report}" ]]; then
      ncu_status=1
    fi
    if grep -Eqi 'ERR_NVGPUCTRPERM|permission.*counter|counter.*permission' \
        "${stdout_file}" "${stderr_file}"; then
      printf '%s_counter_permission=restricted\n' "${label}" \
        >>"${result_dir}/ncu_status.txt"
    fi
  }

  ncu_profile ncu_gemm_s512 --kernel gemm --seq 512
  ncu_profile ncu_gemv --kernel gemv
  ncu_profile ncu_attention_s128 --kernel attention --seq 128
  if [[ "${PROFILE_ATTENTION_S512:-0}" == "1" ]]; then
    ncu_profile ncu_attention_s512_diagnostic --kernel attention --seq 512
  fi
  if [[ ${ncu_status} -eq 0 ]]; then
    printf 'status=success\n' >>"${result_dir}/ncu_status.txt"
  else
    printf 'status=failure\n' >>"${result_dir}/ncu_status.txt"
    overall_status=1
  fi
else
  {
    printf 'Nsight Compute (ncu) not found or not runnable.\n'
    printf 'Set NCU_BIN to an executable whose --version command succeeds.\n'
  } >"${result_dir}/NCU_UNAVAILABLE.txt"
  printf 'tool_path=unavailable\ndiscovery=failure\nstatus=failure\n' \
    >>"${result_dir}/ncu_status.txt"
  overall_status=1
fi

if nsys_bin="$(discover_cuda_tool nsys NSYS_BIN)"; then
  printf 'tool_path=%s\ndiscovery=success\n' "${nsys_bin}" \
    >>"${result_dir}/nsys_status.txt"
  write_tool_version "${nsys_bin}" "${result_dir}/nsys_version.txt"
  rm -f "${result_dir}/nsys_attention_s128.nsys-rep" \
    "${result_dir}/nsys_attention_s128.sqlite" \
    "${result_dir}/nsys.stdout.txt" "${result_dir}/nsys.stderr.txt"
  set +e
  "${nsys_bin}" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --stats=true --force-overwrite=true \
    -o "${result_dir}/nsys_attention_s128" \
    "${build_dir}/gpu_bench" --kernel attention --seq 128 --warmup 1 \
    --iterations 3 --skip-tests >"${result_dir}/nsys.stdout.txt" \
    2>"${result_dir}/nsys.stderr.txt"
  nsys_status=$?
  set -e
  printf 'nsys_exit_status=%s\n' "${nsys_status}" \
    >>"${result_dir}/nsys_status.txt"
  if [[ -s "${result_dir}/nsys_attention_s128.nsys-rep" ]]; then
    printf 'nsys_report=generated\n' >>"${result_dir}/nsys_status.txt"
  else
    printf 'nsys_report=missing\n' >>"${result_dir}/nsys_status.txt"
  fi
  if [[ ${nsys_status} -eq 0 && -s "${result_dir}/nsys_attention_s128.nsys-rep" ]]; then
    printf 'status=success\n' >>"${result_dir}/nsys_status.txt"
  else
    printf 'status=failure\n' >>"${result_dir}/nsys_status.txt"
    overall_status=1
  fi
else
  {
    printf 'Nsight Systems (nsys) not found or not runnable.\n'
    printf 'Set NSYS_BIN to an executable whose --version command succeeds.\n'
  } >"${result_dir}/NSYS_UNAVAILABLE.txt"
  printf 'tool_path=unavailable\ndiscovery=failure\nstatus=failure\n' \
    >>"${result_dir}/nsys_status.txt"
  overall_status=1
fi

if [[ ${overall_status} -ne 0 ]]; then
  printf 'GPU profiling incomplete; inspect the status and stdout/stderr files in %s\n' \
    "${result_dir}" >&2
  exit 1
fi
printf 'GPU profiling artifacts written to %s\n' "${result_dir}"
