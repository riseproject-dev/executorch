/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 add. Compiled with -march=rv64gcv; the dispatcher only
 * routes here when riscv_features_detect() reports RISCV_FEATURE_V. The
 * loop uses vsetvl strip-mining so a single body handles the tail without a
 * separate scalar epilogue.
 */

#include <riscv_vector.h>
#include <stddef.h>

void riscv_add_f32_rvv(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);
    vfloat32m8_t vc = __riscv_vfadd_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(out, vc, vl);
    a += vl;
    b += vl;
    out += vl;
    n -= vl;
  }
}
