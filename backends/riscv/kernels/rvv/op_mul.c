/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Placeholder RVV variant — forwards to scalar. Real impl mirrors op_add's
 * strip-mined vfmul.vv with __riscv_vsetvl_e32m8 / __riscv_vfmul_vv_f32m8.
 */

#include <stddef.h>

extern void riscv_mul_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n);

void riscv_mul_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  riscv_mul_f32_scalar(a, b, out, n);
}
