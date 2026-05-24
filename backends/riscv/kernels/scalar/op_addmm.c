/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * out[m, n] = beta * c[m, n] + alpha * sum_k a[m, k] * b[k, n].
 * Naive triple-loop; the inner loop accumulates row-major B so adjacent k
 * iterations hit adjacent cache lines. RVV variant (later) widens the
 * inner reduction with vfmacc.
 *
 * Conventions match aten::addmm: a is (M, K), b is (K, N), c is broadcastable
 * to (M, N) — the C++ glue resolves broadcast and either passes a contiguous
 * (M, N) bias here or NULL (with beta == 0) to skip the bias load.
 */

#include <stddef.h>

void riscv_addmm_f32_scalar(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    float beta) {
  for (size_t m = 0; m < M; ++m) {
    const float* ar = a + m * K;
    float* outr = out + m * N;
    const float* cr = (c != NULL) ? (c + m * N) : NULL;
    for (size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        acc += ar[k] * b[k * N + n];
      }
      float prior = (cr != NULL) ? (beta * cr[n]) : 0.0f;
      outr[n] = prior + alpha * acc;
    }
  }
}
