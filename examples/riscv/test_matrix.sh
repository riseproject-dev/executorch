#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Local mirror of riscv64.yml's matrix using two docker containers:
#
#   - executorch-docker-driver (ubuntu:26.04 + gcc-15): baremetal only.
#     26.04 is the only release shipping libstdc++-riscv64-unknown-elf-picolibc.
#   - executorch-24             (ubuntu:24.04 + gcc-14): linux only.
#     24.04's glibc-riscv64 is built without mandatory V, so qemu's basic
#     rv64 cpu can execute it (26.04's glibc SIGILLs on vsetivli at startup).
#
# Each cell runs examples/riscv/run.sh once with --build_dir=/tmp/cmsweep-*
# so cells don't fight over the same cmake output. Per-cell stdout/stderr
# lands in riscv_test/${model}_${backend}_${os}_${arch}_riscv.out (same
# layout run.sh's --quantize variants use).
#
# Usage:
#   examples/riscv/test_matrix.sh                    # full sweep
#   examples/riscv/test_matrix.sh --model=mv2        # one model, all configs
#   examples/riscv/test_matrix.sh --os=baremetal     # one OS
#   examples/riscv/test_matrix.sh --quantize-only    # skip the no-q half
#   examples/riscv/test_matrix.sh --setup-only       # bootstrap containers, don't run
#
# Re-runs are cheap when the per-cell build dirs survive (set --keep-build).

set -uxo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
et_root_dir=$(realpath "${script_dir}/../..")

model_filter=""
os_filter=""
arch_filter=""
variant_filter=""
quantize_mode="both"   # both | only | none
setup_only=false
keep_build=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
Options:
  --model=<NAME>     Only run cells for this model
  --os=<linux|baremetal>
  --arch=<rv64|rv32>
  --variant=<scalar|rvv>
  --quantize-only    Skip the non-quantized cells
  --no-quantize      Skip the quantized cells
  --setup-only       Make sure both containers are ready, then exit
  --keep-build       Reuse /tmp/cmsweep-* dirs instead of starting fresh
  -h, --help
EOF
}

for arg in "$@"; do
    case $arg in
        --model=*)     model_filter="${arg#*=}"   ;;
        --os=*)        os_filter="${arg#*=}"      ;;
        --arch=*)      arch_filter="${arg#*=}"    ;;
        --variant=*)   variant_filter="${arg#*=}" ;;
        --quantize-only) quantize_mode="only"     ;;
        --no-quantize)   quantize_mode="none"     ;;
        --setup-only)  setup_only=true            ;;
        --keep-build)  keep_build=true            ;;
        -h|--help)     usage; exit 0              ;;
        *)             echo "Unknown: $arg" >&2; usage; exit 1 ;;
    esac
done

# Container names + image tags match what the CI workflow consumes.
BAREMETAL_CTR=executorch-riscv-baremetal
LINUX_CTR=executorch-riscv-linux

# `add`/`mv2`/`resnet18` are the only models with XNNPACK quantization recipes
# in MODEL_NAME_TO_OPTIONS — others raise at AOT time when --quantize is set.
QUANTIZED_MODELS="add mv2 resnet18"
ALL_MODELS="add mv2 resnet18 mobilebert llama2 yolo26"

# qemu-cpu-ext sweeps; keep parity with the JSON arrays in riscv64.yml.
SCALAR_EXT="zba=true,zbb=true,zbs=true,v=false"
RVV_EXT="zba=true,zbb=true,zbs=true,v=true,vlen=128,elen=64,vext_spec=v1.0"

# ---- container bootstrap (idempotent) -------------------------------------

ensure_baremetal() {
    if ! docker ps -a --format '{{.Names}}' | grep -qx "${BAREMETAL_CTR}"; then
        echo "[matrix] starting ${BAREMETAL_CTR} (ubuntu:26.04)"
        docker run -d --platform linux/arm64 --name "${BAREMETAL_CTR}" \
            -e DEBIAN_FRONTEND=noninteractive \
            -v "${et_root_dir}":/executorch -w /executorch \
            ubuntu:26.04 sleep infinity >/dev/null
    fi
    docker start "${BAREMETAL_CTR}" >/dev/null
    if ! docker exec "${BAREMETAL_CTR}" test -d /executorch/.venv-docker; then
        echo "[matrix] bootstrapping ${BAREMETAL_CTR} (this takes a few minutes)"
        docker exec "${BAREMETAL_CTR}" bash -c '
            set -e
            apt-get update -qq && apt-get install -y -qq --no-install-recommends \
                python3 python3-pip ca-certificates sudo
            python3 -m pip install --break-system-packages --quiet uv
            uv python install 3.13
            cd /executorch
            uv venv --python 3.13 --seed .venv-docker
            source .venv-docker/bin/activate
            pip install --upgrade --quiet pip
            pip install --quiet executorch
            bash examples/riscv/setup.sh
            pip install --quiet -r examples/riscv/requirements.txt
            # Make the local backend visible to the wheel.
            ln -sfn /executorch/backends/riscv \
                /executorch/.venv-docker/lib/python3.13/site-packages/executorch/backends/riscv
        '
    fi
}

ensure_linux() {
    if ! docker ps -a --format '{{.Names}}' | grep -qx "${LINUX_CTR}"; then
        echo "[matrix] starting ${LINUX_CTR} (ubuntu:24.04)"
        docker run -d --platform linux/arm64 --name "${LINUX_CTR}" \
            -e DEBIAN_FRONTEND=noninteractive \
            -v "${et_root_dir}":/executorch -w /executorch \
            ubuntu:24.04 sleep infinity >/dev/null
    fi
    docker start "${LINUX_CTR}" >/dev/null
    if ! docker exec "${LINUX_CTR}" test -d /executorch/.venv-docker-24; then
        echo "[matrix] bootstrapping ${LINUX_CTR} (this takes a few minutes)"
        docker exec -e GCC_VERSION=14 "${LINUX_CTR}" bash -c '
            set -e
            apt-get update -qq && apt-get install -y -qq --no-install-recommends \
                python3 python3-pip python3-venv ca-certificates sudo curl
            python3 -m pip install --break-system-packages --quiet uv
            uv python install 3.13
            cd /executorch
            uv venv --python 3.13 --seed .venv-docker-24
            source .venv-docker-24/bin/activate
            pip install --upgrade --quiet pip
            pip install --quiet executorch
            bash examples/riscv/setup.sh
            pip install --quiet -r examples/riscv/requirements.txt
            ln -sfn /executorch/backends/riscv \
                /executorch/.venv-docker-24/lib/python3.13/site-packages/executorch/backends/riscv
        '
    fi
}

ensure_baremetal
ensure_linux
if ${setup_only}; then exit 0; fi

# ---- one cell --------------------------------------------------------------

# Args: ctr venv os arch variant ext model quantize_flag
run_cell() {
    local ctr=$1 venv=$2 os=$3 arch=$4 variant=$5 ext=$6 model=$7 q=$8
    local cell="${model}/${os}-${arch}/${variant}${q:++q}"
    local bd="/tmp/cmsweep-${model}-${os}-${arch}-${variant}${q:+-q}"
    if ! ${keep_build}; then
        docker exec "${ctr}" rm -rf "${bd}"
    fi
    if docker exec "${ctr}" bash -lc "
            cd /executorch && source ${venv}/bin/activate &&
            timeout 1800 bash examples/riscv/run.sh \
              --model=${model} --backend=riscv ${q} \
              --os=${os} --arch=${arch} \
              --qemu-cpu-ext='${ext}' \
              --build_dir=${bd} --timeout=900
        " >/dev/null 2>&1; then
        # run.sh writes the per-cell log to riscv_test/<model>_<backend>_<os>_<arch>_riscv.out
        # run.sh's "Bundled I/O check PASSED" goes to its stdout (which we
        # silence above). The .out file has the per-testset `Test_result:
        # PASS` markers that run.sh's own grep keys on — match that instead.
        # Inside the surrounding double-quoted string, bare "${q}" works
        # correctly inside $( ... ); backslash-escaping the inner quotes
        # produces literal characters that flip the -n test to always-true.
        local log="/executorch/riscv_test/${model}$([ -n "${q}" ] && echo _q)_riscv_${os}_${arch}_riscv.out"
        if docker exec "${ctr}" grep -q "Test_result: PASS" "${log}" \
                2>/dev/null \
           && ! docker exec "${ctr}" grep -q "Test_result: FAIL" "${log}" \
                2>/dev/null; then
            echo "  PASS  ${cell}"
            return 0
        fi
    fi
    echo "  FAIL  ${cell}"
    return 1
}

# ---- iterate ---------------------------------------------------------------

passed=0; total=0
for os_arch in "linux:rv64" "baremetal:rv64" "baremetal:rv32"; do
    os="${os_arch%%:*}"; arch="${os_arch##*:}"
    if [[ -n "${os_filter}" && "${os}" != "${os_filter}" ]]; then continue; fi
    if [[ -n "${arch_filter}" && "${arch}" != "${arch_filter}" ]]; then continue; fi
    if [[ "${os}" == "linux" ]]; then ctr="${LINUX_CTR}";    venv=/executorch/.venv-docker-24
    else                              ctr="${BAREMETAL_CTR}"; venv=/executorch/.venv-docker; fi

    for variant_lbl in "scalar:${SCALAR_EXT}" "rvv:${RVV_EXT}"; do
        variant="${variant_lbl%%:*}"; ext="${variant_lbl#*:}"
        if [[ -n "${variant_filter}" && "${variant}" != "${variant_filter}" ]]; then continue; fi

        # non-quantized models
        if [[ "${quantize_mode}" != "only" ]]; then
            for m in ${ALL_MODELS}; do
                if [[ -n "${model_filter}" && "${m}" != "${model_filter}" ]]; then continue; fi
                total=$((total+1))
                run_cell "${ctr}" "${venv}" "${os}" "${arch}" "${variant}" "${ext}" "${m}" "" \
                    && passed=$((passed+1))
            done
        fi
        # quantized — only the 3 models with XNNPACK recipes
        if [[ "${quantize_mode}" != "none" ]]; then
            for m in ${QUANTIZED_MODELS}; do
                if [[ -n "${model_filter}" && "${m}" != "${model_filter}" ]]; then continue; fi
                total=$((total+1))
                run_cell "${ctr}" "${venv}" "${os}" "${arch}" "${variant}" "${ext}" "${m}" "--quantize" \
                    && passed=$((passed+1))
            done
        fi
    done
done

echo
echo "===== ${passed}/${total} cells passed ====="
test "${passed}" -eq "${total}"
