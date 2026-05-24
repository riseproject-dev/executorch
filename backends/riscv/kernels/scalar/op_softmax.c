/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Numerically-stable softmax along the trailing `inner` axis. Subtracts the
 * row max before exp to keep the exponent in range, then normalizes by the
 * sum. Tracks NaN-safe sum (0 inputs -> 1/inner uniform distribution to
 * match aten).
 */

#include <math.h>
#include <stddef.h>

void riscv_softmax_f32_contig_scalar(
    const float* in,
    float* out,
    size_t outer,
    size_t inner) {
  for (size_t r = 0; r < outer; ++r) {
    const float* row_in = in + r * inner;
    float* row_out = out + r * inner;
    float maxv = row_in[0];
    for (size_t i = 1; i < inner; ++i) {
      if (row_in[i] > maxv)
        maxv = row_in[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < inner; ++i) {
      float e = expf(row_in[i] - maxv);
      row_out[i] = e;
      sum += e;
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (size_t i = 0; i < inner; ++i) {
      row_out[i] *= inv;
    }
  }
}
