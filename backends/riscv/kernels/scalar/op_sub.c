/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

void riscv_sub_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    float alpha) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] - alpha * b[i];
}
