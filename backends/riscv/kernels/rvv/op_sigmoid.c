/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_sigmoid_f32_scalar(const float* in, float* out, size_t n);

void riscv_sigmoid_f32_rvv(const float* in, float* out, size_t n) {
  riscv_sigmoid_f32_scalar(in, out, n);
}
