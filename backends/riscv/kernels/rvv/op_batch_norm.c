/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 inference batch norm. y = x * scale + offset, with
 * scale = weight / sqrt(var + eps) and offset = bias - mean * scale
 * precomputed once per channel. The inner H*W loop then becomes a
 * single vfmadd.vf which is the densest fp32 op RVV exposes — chips with
 * paired FMUL+FADD pipelines hit peak throughput here.
 *
 * scale/offset use the scalar libm sqrtf for now; the per-channel cost
 * is C calls vs N*C*HW MACs so it doesn't pay to vectorise the prep.
 */

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>

void riscv_batch_norm_f32_nchw_rvv(
    const float* in,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    float eps,
    float* out,
    size_t N,
    size_t C,
    size_t HW) {
  for (size_t c = 0; c < C; ++c) {
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float w = (weight != 0) ? weight[c] : 1.0f;
    float b = (bias != 0) ? bias[c] : 0.0f;
    float scale = w * inv_std;
    float offset = b - mean[c] * scale;
    for (size_t n = 0; n < N; ++n) {
      const float* in_p = in + (n * C + c) * HW;
      float* out_p = out + (n * C + c) * HW;
      size_t left = HW;
      while (left > 0) {
        size_t vl = __riscv_vsetvl_e32m8(left);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(in_p, vl);
        /* vfmacc.vf: vy = scale*vx + offset (vfmadd's destination is its
         * own register, vfmacc accumulates into the operand — pick the
         * one that fuses cleanly with the broadcast of `offset`). */
        vfloat32m8_t vy = __riscv_vfmv_v_f_f32m8(offset, vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, scale, vx, vl);
        __riscv_vse32_v_f32m8(out_p, vy, vl);
        in_p += vl;
        out_p += vl;
        left -= vl;
      }
    }
  }
}
