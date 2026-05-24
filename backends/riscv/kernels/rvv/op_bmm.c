/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 batched matmul. Outer loop over batch, then the same
 * strided-load + vfmul + vfredusum kernel as op_mm. Keeping them as
 * separate TUs (mm.c iterates one matrix, bmm.c iterates B) lets the
 * compiler unroll per-context — fusing them would force the inner loop
 * to carry a branch on B.
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
      for (size_t n = 0; n < N; ++n) {
        size_t left = K;
        const float* a_p = a_row;
        const float* b_p = bb + n;
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        while (left > 0) {
          size_t vl = __riscv_vsetvl_e32m4(left);
          vfloat32m4_t va = __riscv_vle32_v_f32m4(a_p, vl);
          vfloat32m4_t vb =
              __riscv_vlse32_v_f32m4(b_p, (ptrdiff_t)(N * sizeof(float)), vl);
          vfloat32m4_t prod = __riscv_vfmul_vv_f32m4(va, vb, vl);
          acc = __riscv_vfredusum_vs_f32m4_f32m1(prod, acc, vl);
          a_p += vl;
          b_p += vl * N;
          left -= vl;
        }
        outb[m * N + n] = __riscv_vfmv_f_s_f32m1_f32(acc);
      }
    }
  }
}
