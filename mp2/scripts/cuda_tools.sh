#!/usr/bin/env bash

# Shared discovery for versioned CUDA installations. A nonempty override is
# authoritative: an invalid override fails instead of silently selecting a
# different installation.
cuda_tool_is_runnable() {
  local tool_name="$1"
  local candidate="$2"
  local smoke_output=""
  if [[ ! -f "${candidate}" || ! -x "${candidate}" ]] ||
     ! "${candidate}" --version >/dev/null 2>&1; then
    return 1
  fi

  # A legacy nvcc may run successfully but cannot compile the course target.
  # Check explicit numeric architectures before accepting it from PATH.
  if [[ "${tool_name}" == "nvcc" ]]; then
    local gpu_codes=""
    local architecture=""
    local -a architectures=()
    gpu_codes="$("${candidate}" --list-gpu-code 2>/dev/null)" || return 1
    IFS=';' read -r -a architectures <<<"${CUDA_ARCHITECTURES:-121}"
    for architecture in "${architectures[@]}"; do
      architecture="${architecture%-real}"
      architecture="${architecture%-virtual}"
      case "${architecture}" in
        native|all|all-major) continue ;;
      esac
      if ! grep -Fxq "sm_${architecture}" <<<"${gpu_codes}"; then
        return 1
      fi
    done
  fi

  # Some distro packages leave compute-sanitizer's launcher in PATH without
  # its injection library. --version still succeeds in that broken state.
  if [[ "${tool_name}" == "compute-sanitizer" ]]; then
    smoke_output="$("${candidate}" --tool memcheck /bin/true 2>&1 || true)"
    if grep -Eqi \
        'Unable to find injection library|error while loading shared libraries' \
        <<<"${smoke_output}"; then
      return 1
    fi
  fi
  return 0
}

discover_cuda_tool() {
  local tool_name="$1"
  local override_name="$2"
  local override_value="${!override_name:-}"
  local candidate=""
  local -a candidates=()
  local -a versioned_candidates=()

  if [[ -n "${override_value}" ]]; then
    if [[ "${override_value}" != */* ]]; then
      candidate="$(command -v -- "${override_value}" 2>/dev/null || true)"
    else
      candidate="${override_value}"
    fi
    if [[ -n "${candidate}" ]] &&
       cuda_tool_is_runnable "${tool_name}" "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    printf '%s=%s is not a usable %s executable for the requested target.\n' \
      "${override_name}" "${override_value}" "${tool_name}" >&2
    if [[ "${tool_name}" == "nvcc" ]]; then
      printf 'Required CUDA architectures: %s (DGX Spark: 121, CUDA 13+).\n' \
        "${CUDA_ARCHITECTURES:-121}" >&2
    fi
    return 1
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    candidates+=("${CUDA_HOME}/bin/${tool_name}")
  fi
  if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
    candidates+=("${CUDAToolkit_ROOT}/bin/${tool_name}")
  fi
  candidate="$(command -v -- "${tool_name}" 2>/dev/null || true)"
  if [[ -n "${candidate}" ]]; then
    candidates+=("${candidate}")
  fi
  candidates+=("/usr/local/cuda/bin/${tool_name}")

  shopt -s nullglob
  versioned_candidates=(/usr/local/cuda-*/bin/"${tool_name}")
  shopt -u nullglob
  if (( ${#versioned_candidates[@]} != 0 )); then
    while IFS= read -r candidate; do
      candidates+=("${candidate}")
    done < <(printf '%s\n' "${versioned_candidates[@]}" | sort -Vr)
  fi

  for candidate in "${candidates[@]}"; do
    if cuda_tool_is_runnable "${tool_name}" "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

write_tool_version() {
  local tool_path="$1"
  local output_path="$2"
  {
    printf 'path=%s\n' "${tool_path}"
    if "${tool_path}" --version; then
      printf 'version_check=success\n'
    else
      printf 'version_check=failure\n'
      return 1
    fi
  } >"${output_path}" 2>&1
}
