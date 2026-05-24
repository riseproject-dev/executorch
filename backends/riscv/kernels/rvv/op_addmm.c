/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_addmm_f32_scalar(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    float beta);

void riscv_addmm_f32_rvv(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    float beta) {
  riscv_addmm_f32_scalar(a, b, c, out, M, N, K, alpha, beta);
}
