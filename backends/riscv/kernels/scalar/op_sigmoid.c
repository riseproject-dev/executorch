/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Numerically-stable sigmoid: for negative inputs use the exp(x) / (1+exp(x))
 * form so the exp argument stays <= 0 and doesn't overflow at fp32 range.
 */

#include <math.h>
#include <stddef.h>

void riscv_sigmoid_f32_scalar(const float* in, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float x = in[i];
    if (x >= 0.0f) {
      float e = expf(-x);
      out[i] = 1.0f / (1.0f + e);
    } else {
      float e = expf(x);
      out[i] = e / (1.0f + e);
    }
  }
}
