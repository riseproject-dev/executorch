/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Placeholder RVV variant: forwards to scalar. Replaced with a real
 * vfmax + vfmin strip-mine in the follow-up RVV pass.
 */

#include <stddef.h>

extern void riscv_hardtanh_f32_scalar(
    const float* in,
    float* out,
    size_t n,
    float lo,
    float hi);

void riscv_hardtanh_f32_rvv(
    const float* in,
    float* out,
    size_t n,
    float lo,
    float hi) {
  riscv_hardtanh_f32_scalar(in, out, n, lo, hi);
}
