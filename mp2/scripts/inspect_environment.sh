#!/usr/bin/env bash
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
output="${1:-${RESULT_DIR:-${project_dir}/results}/environment.txt}"
source "${script_dir}/cuda_tools.sh"
mkdir -p "$(dirname -- "${output}")"

record_generic_tool() {
  local name="$1"
  local path=""
  path="$(command -v -- "${name}" 2>/dev/null || true)"
  printf '\n[%s]\n' "${name}"
  if [[ -z "${path}" ]]; then
    printf 'discovery=failure\npath=unavailable\n'
    return
  fi
  printf 'discovery=success\npath=%s\n' "${path}"
  if "${path}" --version 2>&1; then
    printf 'version_check=success\n'
  else
    printf 'version_check=failure\n'
  fi
}

record_cuda_tool() {
  local name="$1"
  local override_name="$2"
  local path=""
  printf '\n[%s]\n' "${name}"
  if ! path="$(discover_cuda_tool "${name}" "${override_name}" 2>&1)"; then
    printf 'discovery=failure\npath=unavailable\ndetail=%s\n' "${path}"
    return
  fi
  printf 'discovery=success\npath=%s\noverride=%s\n' \
    "${path}" "${override_name}"
  if "${path}" --version 2>&1; then
    printf 'version_check=success\n'
  else
    printf 'version_check=failure\n'
  fi
}

{
  printf 'inspection_time=%s\n' "$(date --iso-8601=seconds)"
  printf '\n[OS]\n'
  uname -a
  cat /etc/os-release
  printf '\n[CPU]\n'
  lscpu
  printf '\n[MEMORY]\n'
  free -h
  printf '\n[NVIDIA GPU AND DRIVER]\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total \
      --format=csv
  else
    printf 'nvidia-smi unavailable\n'
  fi
  printf '\n[TOOL DISCOVERY]\n'
  record_generic_tool g++
  record_generic_tool cmake
  record_generic_tool make
  record_cuda_tool nvcc NVCC_BIN
  record_cuda_tool ncu NCU_BIN
  record_cuda_tool nsys NSYS_BIN
  record_cuda_tool compute-sanitizer COMPUTE_SANITIZER_BIN
} >"${output}" 2>&1

printf 'Environment report written to %s\n' "${output}"
