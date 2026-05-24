/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 mean over the trailing axis. Accumulates with vfredusum into
 * a scalar register, then scales by 1/inner once per row. m4 here, not m8,
 * because vfredusum's reduction destination is m1 — m8 would tax the
 * compiler's register allocator for marginal throughput gain.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_mean_dim_f32_contig_rvv(
    const float* in,
    float* out,
    size_t outer,
    size_t inner) {
  if (inner == 0) {
    for (size_t i = 0; i < outer; ++i) out[i] = 0.0f;
    return;
  }
  const float scale = 1.0f / (float)inner;
  for (size_t r = 0; r < outer; ++r) {
    const float* row = in + r * inner;
    size_t left = inner;
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    while (left > 0) {
      size_t vl = __riscv_vsetvl_e32m4(left);
      vfloat32m4_t v = __riscv_vle32_v_f32m4(row, vl);
      acc = __riscv_vfredusum_vs_f32m4_f32m1(v, acc, vl);
      row += vl;
      left -= vl;
    }
    out[r] = __riscv_vfmv_f_s_f32m1_f32(acc) * scale;
  }
}
