/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 matmul. Per (m, n) output, vfmacc-accumulate along K with a
 * strip-mined inner loop, then reduce the per-lane partials into a scalar
 * with vfredusum. Modelled on llama.cpp's ggml-cpu/arch/riscv/quants.c
 * ggml_vec_dot_q8_0_q8_0 (same shape: load two LMUL-sized rows, vwmacc
 * into wider accumulator, reduce, scale) but staying in fp32 throughout
 * since aten::mm is unquantized.
 *
 * Layout: a is row-major (M, K), b is row-major (K, N). The inner loop walks
 * b column-by-K so adjacent k iterations cross rows; a single contiguous
 * column of B is gathered via strided loads (stride = N*sizeof(float)). The
 * portable scalar version does the same thing scalarly — this kernel just
 * widens the inner reduction.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_mm_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t M,
    size_t N,
    size_t K) {
  for (size_t m = 0; m < M; ++m) {
    const float* a_row = a + m * K;
    for (size_t n = 0; n < N; ++n) {
      size_t left = K;
      const float* a_p = a_row;
      const float* b_p = b + n; /* walks column n in strides of N */
      vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
      while (left > 0) {
        size_t vl = __riscv_vsetvl_e32m4(left);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a_p, vl);
        /* Strided load of column n of B; stride is N elements = N*4 bytes. */
        vfloat32m4_t vb =
            __riscv_vlse32_v_f32m4(b_p, (ptrdiff_t)(N * sizeof(float)), vl);
        vfloat32m4_t prod = __riscv_vfmul_vv_f32m4(va, vb, vl);
        acc = __riscv_vfredusum_vs_f32m4_f32m1(prod, acc, vl);
        a_p += vl;
        b_p += vl * N;
        left -= vl;
      }
      out[m * N + n] = __riscv_vfmv_f_s_f32m1_f32(acc);
    }
  }
}
