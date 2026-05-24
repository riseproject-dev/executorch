/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 addmm = beta*c + alpha*(a @ b). Outer-product GEMM (see
 * op_mm.c for the rationale) with two scalar mults at the tail. C++
 * glue materialises the bias broadcast into the output tensor before
 * the call, so we read `c` here as a contiguous (M, N) — see
 * ops/op_addmm.cpp.
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
    size_t n = 0;
    while (n < N) {
      size_t vl = __riscv_vsetvl_e32m8(N - n);
      vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
      for (size_t k = 0; k < K; ++k) {
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + k * N + n, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a_row[k], vb, vl);
      }
      /* Final lane post-process: out = beta*c + alpha*acc. Done in two
       * vfmacc.vf — the first folds alpha in, the second adds beta*c. */
      vfloat32m8_t result;
      if (c_row != 0) {
        result = __riscv_vle32_v_f32m8(c_row + n, vl);
        result = __riscv_vfmul_vf_f32m8(result, beta, vl);
        result = __riscv_vfmacc_vf_f32m8(result, alpha, acc, vl);
      } else {
        result = __riscv_vfmul_vf_f32m8(acc, alpha, vl);
      }
      __riscv_vse32_v_f32m8(out_row + n, result, vl);
      n += vl;
    }
  }
}
