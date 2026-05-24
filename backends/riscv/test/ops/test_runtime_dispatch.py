# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QEMU-only smoke test for the runtime dispatcher.

Builds the dispatch + variant TUs into a tiny driver and runs it under
qemu-user with each ``-cpu`` config. This stays local to the backend (no
ExecuTorch runtime or torch needed) so the test is meaningful even before
the rest of the runtime has been built for riscv64.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


QEMU_USER = shutil.which("qemu-riscv64-static")
RV_GCC = shutil.which("riscv64-linux-gnu-gcc")


def _cc(out: Path, src: Path, march: str, extra: list[str] | None = None) -> None:
    cmd = [RV_GCC, "-march=" + march, "-O2"]
    if extra:
        cmd.extend(extra)
    cmd.extend(["-c", "-o", str(out), str(src)])
    subprocess.check_call(cmd)


def _link(exe: Path, objs: list[Path]) -> None:
    subprocess.check_call(
        [RV_GCC, "-march=rv64gc", "-static", "-o", str(exe)] + [str(o) for o in objs]
    )


_MAIN_C = (
    "#include <stdio.h>\n"
    "#include <stddef.h>\n"
    "void riscv_add_f32(const float*, const float*, float*, size_t);\n"
    "int main(void) {\n"
    "  enum { N = 1024 };\n"
    "  float a[N], b[N], c[N];\n"
    "  for (int i = 0; i < N; ++i) { a[i] = i * 0.5f; b[i] = i * 0.25f; }\n"
    "  riscv_add_f32(a, b, c, N);\n"
    "  for (int i = 0; i < N; ++i) {\n"
    "    float want = a[i] + b[i];\n"
    "    if (c[i] != want) { printf(\"Test_result: FAIL\\n\"); return 1; }\n"
    "  }\n"
    "  printf(\"Test_result: PASS\\n\");\n"
    "  return 0;\n"
    "}\n"
)


def _build_test_binary(
    root: Path, tmp_path: Path, variants: list[str]
) -> Path:
    march = {
        "scalar": "rv64gc",
        "rvv": "rv64gcv",
    }

    objs: list[Path] = []
    for v in variants:
        obj = tmp_path / f"{v}.o"
        _cc(obj, root / f"kernels/{v}/op_add.c", march[v])
        objs.append(obj)

    dispatch_defs = [f"-DRISCV_KERNELS_HAVE_{v.upper()}=1" for v in variants]
    dispatch_o = tmp_path / "dispatch.o"
    _cc(dispatch_o, root / "kernels/dispatch/op_add_dispatch.c", "rv64gc", dispatch_defs)
    objs.append(dispatch_o)

    features_o = tmp_path / "features.o"
    _cc(features_o, root / "runtime/riscv_features.c", "rv64gc")
    features_linux_o = tmp_path / "features_linux.o"
    _cc(features_linux_o, root / "runtime/riscv_features_linux.c", "rv64gc")
    objs.extend([features_o, features_linux_o])

    main_c = tmp_path / "main.c"
    main_c.write_text(_MAIN_C)
    main_o = tmp_path / "main.o"
    _cc(main_o, main_c, "rv64gc")
    objs.insert(0, main_o)

    exe = tmp_path / "test_add"
    _link(exe, objs)
    return exe


@pytest.mark.skipif(
    QEMU_USER is None or RV_GCC is None,
    reason="needs qemu-riscv64-static and gcc-riscv64-linux-gnu",
)
@pytest.mark.parametrize(
    "qemu_cpu",
    [
        "rv64",
        "rv64,v=true,vlen=128",
        "rv64,v=true,vlen=256",
        "rv64,v=true,vlen=512",
    ],
)
def test_runtime_dispatch_under_qemu(qemu_cpu: str, tmp_path: Path) -> None:
    """Path 1: every variant compiled in, dispatcher picks at runtime."""
    root = Path(__file__).resolve().parents[2]
    exe = _build_test_binary(root, tmp_path, ["scalar", "rvv"])
    out = subprocess.run(
        [QEMU_USER, "-cpu", qemu_cpu, str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Test_result: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(
    QEMU_USER is None or RV_GCC is None,
    reason="needs qemu-riscv64-static and gcc-riscv64-linux-gnu",
)
@pytest.mark.parametrize(
    "qemu_cpu",
    [
        "rv64",
        "rv64,v=true,vlen=128",
        "rv64,v=true,vlen=256",
    ],
)
def test_path2_scalar_only_subset(qemu_cpu: str, tmp_path: Path) -> None:
    """Path 2: compile-time subset of just scalar. Must still pass on every
    CPU config including those that advertise V — the dispatcher should
    pick scalar because no other variant was linked."""
    root = Path(__file__).resolve().parents[2]
    exe = _build_test_binary(root, tmp_path, ["scalar"])
    out = subprocess.run(
        [QEMU_USER, "-cpu", qemu_cpu, str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Test_result: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(
    QEMU_USER is None or RV_GCC is None,
    reason="needs qemu-riscv64-static and gcc-riscv64-linux-gnu",
)
def test_path2_rvv_only_on_rvv_cpu(tmp_path: Path) -> None:
    """Path 2: only RVV compiled in. Must work on a CPU that has V."""
    root = Path(__file__).resolve().parents[2]
    exe = _build_test_binary(root, tmp_path, ["rvv"])
    out = subprocess.run(
        [QEMU_USER, "-cpu", "rv64,v=true,vlen=128", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Test_result: PASS" in out.stdout, out.stdout
