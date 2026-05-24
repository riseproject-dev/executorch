/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Baseline scalar fp32 add. Compiles under whatever -march= the cross
 * toolchain set (rv64gc on linux, rv64iafd / rv32imafdc on baremetal); no
 * extension intrinsics, so the same TU works in every supported config and
 * acts as the dispatcher's fallback.
 */

#include <stddef.h>

void riscv_add_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}
