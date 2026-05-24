/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

void riscv_relu_f32_scalar(const float* in, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float v = in[i];
    out[i] = v > 0.0f ? v : 0.0f;
  }
}
