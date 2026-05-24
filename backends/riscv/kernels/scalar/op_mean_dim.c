/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Reduce over the trailing `inner` floats of every contiguous row, write
 * one mean per `outer` row. NCHW global-pool feeds C+W collapsed before
 * the call so this stays a single 2-loop kernel.
 */

#include <stddef.h>

void riscv_mean_dim_f32_contig_scalar(
    const float* in,
    float* out,
    size_t outer,
    size_t inner) {
  if (inner == 0) {
    for (size_t i = 0; i < outer; ++i) {
      out[i] = 0.0f;
    }
    return;
  }
  const float scale = 1.0f / (float)inner;
  for (size_t r = 0; r < outer; ++r) {
    const float* row = in + r * inner;
    float acc = 0.0f;
    for (size_t i = 0; i < inner; ++i) {
      acc += row[i];
    }
    out[r] = acc * scale;
  }
}
