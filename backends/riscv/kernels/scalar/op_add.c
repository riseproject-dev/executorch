/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Element-wise add — fp32 baseline + int8 requantising fused variant.
 *
 * Both compile under whatever -march= the cross toolchain set (rv64gc on
 * linux, rv64iafd / rv32imafdc on baremetal); no extension intrinsics,
 * so the same TU works in every supported config and acts as the
 * dispatcher's fallback.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void riscv_add_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

/* Fused requantising int8 add: same semantics as the PT2E
 * dequantize_per_tensor -> add -> quantize_per_tensor triplet, but stays
 * in int8 on the hot path (one int<->float cast per lane instead of two
 * dq + one add + one q). The two scale ratios are precomputed once before
 * the loop; the inner body is then two int->float MACs + a round + a clamp.
 */

static inline int8_t saturate_i8(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return (int8_t)v;
}

void riscv_add_int8_scalar(
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
  for (size_t i = 0; i < n; ++i) {
    float fa = (float)((int32_t)a[i] - a_zero_point) * a_mul;
    float fb = (float)((int32_t)b[i] - b_zero_point) * b_mul;
    /* roundf -> nearest-even on RVV cores; matches what
     * quantize_val<float, int8_t> does in the portable kernel. */
    int32_t r = (int32_t)roundf(fa + fb) + out_zero_point;
    out[i] = saturate_i8(r, out_quant_min, out_quant_max);
  }
}
