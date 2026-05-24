/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 matmul, outer-product / rank-1-update formulation:
 *
 *   for m:
 *     for n_block of width vl:
 *       acc = 0
 *       for k:
 *         acc = vfmacc.vf(acc, a[m, k], b[k, n_block : n_block + vl])
 *       store(out[m, n_block : n_block + vl], acc)
 *
 * Each inner-loop iteration is one scalar broadcast (a[m, k]) + one
 * contiguous vector load (a row of B) + one fused multiply-add. No
 * strided loads, no vfredusum — both of which the dot-product layout
 * pays per output element. RVV implementations issue vfmacc.vf at peak
 * rate (one per cycle on saturated FMA pipelines), so the K-loop is
 * compute-bound rather than memory- or reduction-bound.
 *
 * Modelled on the same pattern llama.cpp's GEMM kernels use on RVV
 * (ggml-cpu/arch/riscv/repack.cpp packs the weight matrix so the inner
 * loop reads contiguously); the unquantised fp32 case here doesn't
 * need pre-packing because aten::mm already gives row-major B.
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
    float* out_row = out + m * N;
    size_t n = 0;
    while (n < N) {
      size_t vl = __riscv_vsetvl_e32m8(N - n);
      vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
      for (size_t k = 0; k < K; ++k) {
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + k * N + n, vl);
        acc = __riscv_vfmacc_vf_f32m8(acc, a_row[k], vb, vl);
      }
      __riscv_vse32_v_f32m8(out_row + n, acc, vl);
      n += vl;
    }
  }
}
