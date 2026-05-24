/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_bmm_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t B,
    size_t M,
    size_t N,
    size_t K);

void riscv_bmm_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t B,
    size_t M,
    size_t N,
    size_t K) {
  riscv_bmm_f32_scalar(a, b, out, B, M, N, K);
}
