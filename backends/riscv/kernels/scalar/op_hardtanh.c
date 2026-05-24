/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

void riscv_hardtanh_f32_scalar(
    const float* in,
    float* out,
    size_t n,
    float lo,
    float hi) {
  for (size_t i = 0; i < n; ++i) {
    float v = in[i];
    if (v < lo)
      v = lo;
    else if (v > hi)
      v = hi;
    out[i] = v;
  }
}
