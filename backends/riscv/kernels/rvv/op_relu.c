/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 relu: max(x, 0). vfmax.vf with a broadcast 0 — no
 * branches, no mask, single strip-mined loop.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_relu_f32_rvv(const float* in, float* out, size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in, vl);
    vfloat32m8_t vy = __riscv_vfmax_vf_f32m8(vx, 0.0f, vl);
    __riscv_vse32_v_f32m8(out, vy, vl);
    in += vl;
    out += vl;
    n -= vl;
  }
}
