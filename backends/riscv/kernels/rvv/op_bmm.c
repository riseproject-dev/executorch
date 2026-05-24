/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 batched matmul, same outer-product layout as op_mm with
 * an outer batch wrapper. Per (b, m, n_block): K iterations of one
 * scalar broadcast + contiguous row load + vfmacc, no reductions.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_bmm_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t B,
    size_t M,
    size_t N,
    size_t K) {
  for (size_t batch = 0; batch < B; ++batch) {
    const float* ab = a + batch * M * K;
    const float* bb = b + batch * K * N;
    float* outb = out + batch * M * N;
    for (size_t m = 0; m < M; ++m) {
      const float* a_row = ab + m * K;
      float* out_row = outb + m * N;
      size_t n = 0;
      while (n < N) {
        size_t vl = __riscv_vsetvl_e32m8(N - n);
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (size_t k = 0; k < K; ++k) {
          vfloat32m8_t vb = __riscv_vle32_v_f32m8(bb + k * N + n, vl);
          acc = __riscv_vfmacc_vf_f32m8(acc, a_row[k], vb, vl);
        }
        __riscv_vse32_v_f32m8(out_row + n, acc, vl);
        n += vl;
      }
    }
  }
}
