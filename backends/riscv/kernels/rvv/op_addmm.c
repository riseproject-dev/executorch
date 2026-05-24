/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 addmm = beta*c + alpha*(a @ b). Same vfredusum dot pattern
 * as op_mm with two scalar mults at the tail. C++ glue materialises the
 * bias broadcast into the output tensor before the call, so we read `c`
 * here as a contiguous (M, N) — see ops/op_addmm.cpp.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_addmm_f32_rvv(
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
    const float* a_row = a + m * K;
    float* out_row = out + m * N;
    const float* c_row = (c != 0) ? (c + m * N) : 0;
    for (size_t n = 0; n < N; ++n) {
      size_t left = K;
      const float* a_p = a_row;
      const float* b_p = b + n;
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
      float dot = __riscv_vfmv_f_s_f32m1_f32(acc);
      float prior = (c_row != 0) ? (beta * c_row[n]) : 0.0f;
      out_row[n] = prior + alpha * dot;
    }
  }
}
