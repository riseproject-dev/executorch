/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 rsqrt = 1 / sqrt(x). vfrsqrt7 gives a 7-bit approximation;
 * one Newton-Raphson step (y' = y*(1.5 - 0.5*x*y*y)) bumps it to ~14 bits
 * which is enough for bf16 / int8 inference paths but loses a few fp32
 * ulp vs the scalar 1/sqrtf. Good enough for the rms-norm use in llama2 —
 * matches the precision portable's optimized rsqrt uses elsewhere.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_rsqrt_f32_rvv(const float* in, float* out, size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in, vl);
    vfloat32m8_t vy = __riscv_vfrsqrt7_v_f32m8(vx, vl);
    /* Newton-Raphson refinement: y *= 1.5 - 0.5*x*y*y */
    vfloat32m8_t y2 = __riscv_vfmul_vv_f32m8(vy, vy, vl);
    vfloat32m8_t xy2 = __riscv_vfmul_vv_f32m8(vx, y2, vl);
    vfloat32m8_t half_xy2 = __riscv_vfmul_vf_f32m8(xy2, 0.5f, vl);
    vfloat32m8_t three_half = __riscv_vfrsub_vf_f32m8(half_xy2, 1.5f, vl);
    vy = __riscv_vfmul_vv_f32m8(vy, three_half, vl);
    __riscv_vse32_v_f32m8(out, vy, vl);
    in += vl;
    out += vl;
    n -= vl;
  }
}
