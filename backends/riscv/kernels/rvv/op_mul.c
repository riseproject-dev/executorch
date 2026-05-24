/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 mul. Mirrors op_add's strip-mine — load two LMUL=8 chunks,
 * vfmul.vv, store — so the tail is handled by the same vsetvl loop. m8 is
 * deliberate: it lets a single hardware loop saturate the issue width on
 * any RVV implementation while staying within 8/8 register pressure.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_mul_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);
    vfloat32m8_t vc = __riscv_vfmul_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(out, vc, vl);
    a += vl;
    b += vl;
    out += vl;
    n -= vl;
  }
}
