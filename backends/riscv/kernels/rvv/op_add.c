/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 add — fp32 baseline + fused int8 requantising variant.
 *
 * Compiled with the per-target rvv -march (rv64gcv on linux, rv64iafdv /
 * rv32imafdcv on baremetal); the dispatcher only routes here when
 * riscv_features_detect() reports RISCV_FEATURE_V. The loops use vsetvl
 * strip-mining so a single body handles the tail without a separate
 * scalar epilogue.
 */

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

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

/* int8 -> int16 -> int32 -> fp32 widen chain so the zero-point subtract
 * doesn't overflow, then two vfmacc.vf with the precomputed scale
 * ratios, vfcvt back to int32 (RNE), add out_zp, clamp, narrow back to
 * int8. LMUL=2 on the int8 side maps to LMUL=8 on the fp32 intermediate
 * which keeps register pressure manageable across the full chain. */

void riscv_add_int8_rvv(
    const int8_t* a,
    const int8_t* b,
    int8_t* out,
    size_t n,
    int32_t a_zero_point,
    float a_scale,
    int32_t b_zero_point,
    float b_scale,
    int32_t out_zero_point,
    float out_scale,
    int32_t out_quant_min,
    int32_t out_quant_max) {
  float a_mul = a_scale / out_scale;
  float b_mul = b_scale / out_scale;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);
    vint8m2_t va_i8 = __riscv_vle8_v_i8m2(a, vl);
    vint8m2_t vb_i8 = __riscv_vle8_v_i8m2(b, vl);
    vint16m4_t va_i16 = __riscv_vsext_vf2_i16m4(va_i8, vl);
    vint16m4_t vb_i16 = __riscv_vsext_vf2_i16m4(vb_i8, vl);
    vint32m8_t va_i32 = __riscv_vsext_vf2_i32m8(va_i16, vl);
    vint32m8_t vb_i32 = __riscv_vsext_vf2_i32m8(vb_i16, vl);
    va_i32 = __riscv_vsub_vx_i32m8(va_i32, a_zero_point, vl);
    vb_i32 = __riscv_vsub_vx_i32m8(vb_i32, b_zero_point, vl);
    vfloat32m8_t va_f = __riscv_vfcvt_f_x_v_f32m8(va_i32, vl);
    vfloat32m8_t vb_f = __riscv_vfcvt_f_x_v_f32m8(vb_i32, vl);
    vfloat32m8_t acc = __riscv_vfmul_vf_f32m8(va_f, a_mul, vl);
    acc = __riscv_vfmacc_vf_f32m8(acc, b_mul, vb_f, vl);
    vint32m8_t qi = __riscv_vfcvt_x_f_v_i32m8(acc, vl);
    qi = __riscv_vadd_vx_i32m8(qi, out_zero_point, vl);
    qi = __riscv_vmax_vx_i32m8(qi, out_quant_min, vl);
    qi = __riscv_vmin_vx_i32m8(qi, out_quant_max, vl);
    vint16m4_t qi_16 = __riscv_vncvt_x_x_w_i16m4(qi, vl);
    vint8m2_t qi_8 = __riscv_vncvt_x_x_w_i8m2(qi_16, vl);
    __riscv_vse8_v_i8m2(out, qi_8, vl);
    a += vl;
    b += vl;
    out += vl;
    n -= vl;
  }
}
