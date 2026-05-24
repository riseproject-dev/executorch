/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_sub_f32_scalar(
    const float* a, const float* b, float* out, size_t n, float alpha);

void riscv_sub_f32_rvv(
    const float* a, const float* b, float* out, size_t n, float alpha) {
  riscv_sub_f32_scalar(a, b, out, n, alpha);
}
