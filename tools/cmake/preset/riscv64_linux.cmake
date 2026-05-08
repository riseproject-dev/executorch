# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Mirrors tools/cmake/preset/arm_ethosu_linux.cmake for riscv64 Linux. Bundle
# IO is on so the standard executor_runner self-checks reference outputs and
# emits Test_result: PASS / FAIL.

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_DEVTOOLS ON)
set_overridable_option(EXECUTORCH_ENABLE_BUNDLE_IO ON)
