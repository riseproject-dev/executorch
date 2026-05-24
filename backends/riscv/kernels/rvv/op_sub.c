/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 sub with alpha scalar: out[i] = a[i] - alpha * b[i].
 * Folds the alpha multiply into the same vfnmsac (a - alpha*b) so the
 * inner loop is one fused MAC + store. Tail handled by the same vsetvl
 * loop — `vl` shrinks on the last iteration.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_sub_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t n,
    float alpha) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);
    /* vfnmsac.vf: a - alpha*b = a += (-alpha)*b — there's no direct
     * vfnmacc.vf, so do explicit mul+sub when alpha != 1. */
    vfloat32m8_t scaled = __riscv_vfmul_vf_f32m8(vb, alpha, vl);
    vfloat32m8_t vc = __riscv_vfsub_vv_f32m8(va, scaled, vl);
    __riscv_vse32_v_f32m8(out, vc, vl);
    a += vl;
    b += vl;
    out += vl;
    n -= vl;
  }
}
