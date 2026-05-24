/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_mean_dim_f32_contig_scalar(
    const float* in,
    float* out,
    size_t outer,
    size_t inner);

void riscv_mean_dim_f32_contig_rvv(
    const float* in,
    float* out,
    size_t outer,
    size_t inner) {
  riscv_mean_dim_f32_contig_scalar(in, out, outer, inner);
}
