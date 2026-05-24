/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

void riscv_mm_f32_scalar(
    const float* a, const float* b, float* out,
    size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) acc += a[m * K + k] * b[k * N + n];
      out[m * N + n] = acc;
    }
  }
}
