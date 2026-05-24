/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_batch_norm_f32_nchw_scalar(
    const float* in,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    float eps,
    float* out,
    size_t N,
    size_t C,
    size_t HW);

void riscv_batch_norm_f32_nchw_rvv(
    const float* in,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    float eps,
    float* out,
    size_t N,
    size_t C,
    size_t HW) {
  riscv_batch_norm_f32_nchw_scalar(in, weight, bias, mean, var, eps, out, N, C, HW);
}
