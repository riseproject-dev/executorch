/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * out[b, m, n] = sum_k a[b, m, k] * b[b, k, n], all contiguous.
 * Same triple-loop as addmm with an outer batch wrapper.
 */

#include <stddef.h>

void riscv_bmm_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t B,
    size_t M,
    size_t N,
    size_t K) {
  for (size_t batch = 0; batch < B; ++batch) {
    const float* a_p = a + batch * M * K;
    const float* b_p = b + batch * K * N;
    float* out_p = out + batch * M * N;
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          acc += a_p[m * K + k] * b_p[k * N + n];
        }
        out_p[m * N + n] = acc;
      }
    }
  }
}
