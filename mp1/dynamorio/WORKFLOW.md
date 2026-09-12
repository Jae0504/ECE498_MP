# Roofline 생성 → native 측정 → 원하는 metric 분석

학생용 전체 가이드는 [mp1/README.md](../README.md)이다. 빌드, native 파라미터
조절, metric별 명령, 1/2/4/8-thread 비교를 한 문서에서 확인할 수 있다.
아래는 8-thread wrapper를 사용하는 간단한 참고 순서이다.

아래 명령은 **저장소 루트**에서 실행한다. 8개 물리 코어를 사용하는 MP1
baseline이다. Roofline ceiling과 실행 시간은 실제 CPU에서 측정한다.
DynamoRIO는 `mp1/cpu/`의 동일한 GEMM/GEMV/Attention 커널에서 선택한 ROI를
한 번 추적하고, 이후 저장한 trace로 분석한다.

## 1. 빌드와 경로 준비

```sh
sh mp1/dynamorio/build.sh --jobs 8

# 할당받은 CPU 8개. 0-7을 자신의 할당에 맞게 바꾼다.
export MP1_CPUS=0-7
MP1_PY="$PWD/mp1/dynamorio/.venv/bin/python"
MP1_CAL="$PWD/results/my_roofline"
MP1_RUN="$PWD/results/my_metrics"
```

`Build PASS` 이후 진행한다. 동일한 소스/CPU로 이미 통과했다면
재빌드는 필요 없다. 새 실험에는 새 출력 디렉터리 이름을 사용한다.
`MP1_RUN`은 첫 측정 명령이 환경/빌드 정보를 기록하며 초기화하므로 아직 파일을
넣지 않는다.

## 2. CPU ceiling 검증 후 Roofline 먼저 생성

```sh
taskset -c "$MP1_CPUS" "$MP1_PY" mp1/scripts/calibrate_roofline.py \
  --bench mp1/build-dynamorio-native/roofline_bench \
  --threads 8 --max-threads 8 --sizes-mib 1024 2048 \
  --output "$MP1_CAL"

"$MP1_PY" mp1/scripts/plot_roofline_calibration.py "$MP1_CAL"

"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_RUN/native_kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CAL/roofline.csv"
```

`my_roofline/roofline.png`와 `.pdf`가 생성된다. 아직 커널을 측정하지 않았으므로
곡선만 있고 점은 없다. `--kernels` 파일이 아직 없어도 정상이다.

위 명령은 총 working set 1 GiB와 2 GiB를 순차적으로 측정하면서 1/2/4/8 코어 및
독립 FMA accumulator 수를 바꾼다. 기본값은 각 조건에서 warmup 2회,
측정 9회, 조건 순서를 뒤집은 2라운드이다. 포화 판정은
`SATURATION_REPORT.md`, `verification.json`, `bandwidth_saturation.png`,
`fma_saturation.png`에서 확인한다.

**8코어 ceiling은 해당 할당의 기준값이며 전체 소켓 메모리 포화를 보장하지 않는다.**
이전 Xeon 6761P 물리 서버 검증에서는 48–64 물리 코어에서 bandwidth plateau를
확인했지만, EWS의 현재 할당에도 그 결과가 적용되는 것은 아니다.
이번 실행의 포화 여부는 새 보고서의 판정을 따른다. 지정한 크기가 LLC 대비
충분히 크지 않으면 포화 조건을 통과하지 못할 수 있다.

가장 큰 working set은 `MemAvailable`의 1/4 이하여야 한다. 약 13 GiB가 available인
EWS에서는 위 1/2 GiB 설정을 사용할 수 있다. 메모리가 더 부족하면 `free -h`로
확인하고 두 크기를 줄인다. `--sizes-mib`를 생략하면 LLC로부터 큰 배열을 자동
선택하므로 VM에서 메모리 제한 오류가 발생할 수 있다.

완료된 calibration을 재사용하거나 중단 후 이어갈 때는 첫 명령에 `--resume`을
추가한다. 설정/바이너리가 달라졌으면 새 출력 경로가 필요하다. Intel Xeon 6761P에서
하드웨어 cycle로 FMA utilization도 확인하려면 `--fma-per-cycle 2`를 추가한다.
이 옵션에는 `perf` 권한이 필요하고 CPU가 바뀌면 해당 CPU의 throughput을 사용해야 한다.

## 3. 커널별 native 시간과 GFLOP/s

GEMV 하나만 측정:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics native
```

다섯 구성을 모두 측정하려면 다음을 실행한다. 이미 완료한 GEMV는 재사용한다.

```sh
for mp1_case in gemv gemm128 gemm512 attention128 attention512; do
  taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
    "$mp1_case" "$MP1_RUN" --metrics native
done
```

`my_metrics/native_kernel_results.csv`에 2회 warmup 후 9회 측정의 median 시간,
GFLOP/s, 알고리즘 FLOPs/minimum bytes/AI가 저장된다. Attention 행에는 전체
Attention 실행 중 측정한 `qk_ms`, `softmax_ms`, `pv_ms`의 개별 median도 포함된다.

같은 plot 명령을 다시 실행하면 먼저 만든 Roofline에 커널 점이 추가된다.

```sh
"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_RUN/native_kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CAL/roofline.csv"
```

## 4. 원하는 trace metric / cache simulation

`--metrics` 뒤에 아래 이름을 공백으로 나열한다.

| 선택값 | 결과 | 방식 |
|---|---|---|
| `native` | 실행 시간, GFLOP/s, Attention phase 시간; FLOPs/minimum bytes/AI | CPU 시간 + 알고리즘 수식 |
| `counts` | Instructions, Loads, Stores | trace basic counts |
| `opcodes` | opcode별 실행 횟수, SIMD/FMA 명령 분포 | trace opcode mix |
| `reuse` | 평균/중앙값 reuse distance, histogram, data cold fraction | 정확한 reuse distance 분석 |
| `cache` | L1D/L2/LLC accesses, misses, miss rate | cache 모델 simulation |
| `memory` | Dynamic Memory Reference Bytes, Unique Cache Lines | 데이터 operand bytes / 고유 64-byte line 분석 |

GEMV의 instruction/load/store 수를 확인한 뒤 같은 trace에 cache와 memory 분석을 추가:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics counts
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics cache memory
```

GEMM S=128의 cache miss rate만 분석하거나, opcode/reuse 분석을 추가:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s128.sh "$MP1_RUN" \
  --metrics cache
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s128.sh "$MP1_RUN" \
  --metrics opcodes reuse
```

기본 ROI는 전체 커널이다. 초기화, correctness test, warmup은 ROI 밖이며,
ROI 안의 OpenMP/호출 관리 명령은 trace에 포함될 수 있다. `counts`는 ZIP 변환과
8-thread/누락 없는 trace 검증에 필요하므로 `cache` 등만 선택해도 최초 한 번
자동 실행된다. 요청하지 않은 opcode/reuse/cache/memory 분석이나 GEMV pilot은
실행하지 않는다.

**`cache`가 simulation이다.** 시간/GFLOP/s에 `instrumented_region_wall_ms`를
사용하지 않는다. Cache 모델은 `cache_model.cfg`의 L1D 48 KiB, L2 2 MiB,
LLC 256 MiB, 8 cores, LRU, cold start, virtual addresses, no prefetch이다.
이는 이전 gnr2의 실제 LLC 336 MiB를 근사한 reference 모델이며,
현재 EWS의 하드웨어 miss counter 측정값이나 실제 cache 사양이 아니다.
`memory`의 operand bytes는 DRAM bytes가 아니다. `reuse` 평균/중앙값은
thread별 instruction+data line history를 포함한다.

## 5. Attention을 QK / softmax / PV로 분리

전체 Attention과 softmax ROI를 각각 분석:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --metrics counts cache
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --phase softmax --metrics counts cache
```

세 phase를 각각 분석:

```sh
for mp1_phase in qk softmax pv; do
  taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
    --phase "$mp1_phase" --metrics counts cache memory
done
```

S=512는 `measure_attention_s512.sh`로 바꾼다. 각 phase의 입력 준비는 추적
밖에서 수행하며, phase마다 독립 trace와 cold cache simulation을 사용한다.
따라서 phase miss들을 더해 전체 Attention miss로 해석하면 안 된다.
Native phase 시간은 3단계의 `--metrics native`로 구한다.
`--phase softmax --metrics native` 같은 조합은 오류로 거부한다.

## 결과와 재실행

| 경로 | 내용 |
|---|---|
| `$MP1_CAL/roofline.png`, `.pdf`, `.csv` | calibration으로 만든 8코어 Roofline과 native 점 |
| `$MP1_CAL/SATURATION_REPORT.md` | 이번 ceiling의 포화 검증 |
| `$MP1_RUN/native_kernel_results.csv` | native 성능과 analytical metrics |
| `$MP1_RUN/metrics/gemv.json` | 지금까지 완료한 GEMV trace metric |
| `$MP1_RUN/metrics/attention128_softmax.json` | S=128 softmax ROI metric |
| `$MP1_RUN/logs/{counts,opcodes,reuse,cache,memory}_*.log` | 선택한 분석의 원본 출력 |
| `$MP1_RUN/traces/` | 재분석 가능한 trace |
| `$MP1_RUN/commands.jsonl`, `experiment_metadata.json` | 명령, CPU 배치, 소스/바이너리 정보 |

`--metrics` 명령은 완료된 단계와 native series를 자동 재사용한다. 새로운 timing
sample이나 trace가 필요하면 새 `MP1_RUN`을 지정한다. 추적 자체가 중단되어
불완전한 trace가 남은 경우에도 새 디렉터리를 사용한다. 선택 측정은 raw trace도
보존한다. GEMM128은 최소 6 GiB, GEMM512는 18 GiB 여유 공간을 검사한다.

전체 보고서/그림/검증 CSV를 한꺼번에 만들려면 별도 디렉터리에서 기존
`sh mp1/dynamorio/measure_all.sh results/my_full_report`를 사용한다. 이 경로는
자체 8코어 reference ceiling과 모든 metric을 측정한다. 선택 측정의 JSON은
full-suite PASS 보고서가 아니다. Calibration과 full-suite의 `ceilings.csv`는
형식이 다르므로 이 가이드의 plot에는 반드시 **`$MP1_CAL/ceilings.csv`**를 사용한다.

## 전체 소켓 포화 검증이 필요한 경우

다음은 Xeon 6761P 물리 서버에서 64개 물리 코어를 할당받은 경우의 참고 명령이다.
8-thread EWS 기본 실험에는 위 명령을 사용한다. 큰 서버 실험은 별도 경로에 저장한다.

```sh
taskset -c 0-63 "$MP1_PY" mp1/scripts/calibrate_roofline.py \
  --bench mp1/build-dynamorio-native/roofline_bench \
  --threads 8 --max-threads 64 --fma-per-cycle 2 \
  --output results/my_socket_calibration
"$MP1_PY" mp1/scripts/plot_roofline_calibration.py results/my_socket_calibration
```

포화 조건을 충족할 때 생성되는 `saturated_ceilings.csv`는 최대 측정 코어 수
(여기서는 64)의 ceiling이다. 8코어 workload 비교에는 원래 `ceilings.csv`의
`--roof-threads 8`을 사용한다. 64코어 roof를 사용하면 그래프와 해석에도 해당
resource scope를 명시한다.
