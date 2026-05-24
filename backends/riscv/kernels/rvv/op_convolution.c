/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * RVV 1.0 fp32 2-D NCHW conv. Vectorises the inner-most Wout dimension
 * via vfmacc.vf along the (Cin/group, Kh, Kw) reduction. Each vector lane
 * is one output pixel; per (ci, kh, kw) we add the scalar weight times a
 * contiguous row from `in` into the accumulator.
 *
 * Fast path requirements: stride_w=1, dilation=1, pad_w=0, and the
 * wo-block sits far enough left that `wo + vl + Kw - 1 <= Win` — i.e.
 * none of the (Kw - 1) trailing kw reads overruns the input row. The
 * right-edge tail (Kw - 1 output columns) falls through to scalar. For
 * stride_w != 1 or any padding the whole call defers to scalar.
 *
 * Modelled on llama.cpp ggml-cpu/arch/riscv's strip-mined inner-product
 * pattern (vfmul into a wider accumulator + strided loads); the twist
 * here is that the reduction is *outside* the vector loop — vector lanes
 * are output pixels, not reduction terms.
 */

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

extern void riscv_convolution_f32_scalar(
    const float* in, const float* weight, const float* bias, float* out,
    size_t N, size_t Cin, size_t Hin, size_t Win,
    size_t Cout, size_t Hout, size_t Wout,
    size_t Kh, size_t Kw, size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w,
    size_t groups);

void riscv_convolution_f32_rvv(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    size_t N,
    size_t Cin,
    size_t Hin,
    size_t Win,
    size_t Cout,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t groups) {
  /* Defer non-fast configs to the scalar reference. The fast path also
   * requires that Kw - 1 <= Win so the right-edge tail can be computed
   * (Win < Kw is degenerate; scalar handles it correctly). */
  if (stride_w != 1 || dilation_w != 1 || dilation_h != 1 || pad_w != 0 ||
      Kw > Win) {
    riscv_convolution_f32_scalar(
        in, weight, bias, out, N, Cin, Hin, Win, Cout, Hout, Wout,
        Kh, Kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups);
    return;
  }
  size_t Cin_per_g = Cin / groups;
  size_t Cout_per_g = Cout / groups;
  /* Last contiguous-safe output column: any wo <= safe_wo has the full
   * kw range [0..Kw) reachable from a contiguous read. The remaining
   * (Wout - safe_wo - 1) trailing columns are computed scalarly so we
   * don't have to mask per-kw. */
  size_t safe_wo_end = (Win >= Kw - 1) ? (Win - (Kw - 1)) : 0;
  if (safe_wo_end > Wout) safe_wo_end = Wout;

  for (size_t n = 0; n < N; ++n) {
    for (size_t g = 0; g < groups; ++g) {
      for (size_t co = 0; co < Cout_per_g; ++co) {
        size_t co_abs = g * Cout_per_g + co;
        float bias_v = (bias != 0) ? bias[co_abs] : 0.0f;
        for (size_t ho = 0; ho < Hout; ++ho) {
          float* out_row =
              out + ((n * Cout + co_abs) * Hout + ho) * Wout;
          /* Vectorised core: wo in [0, safe_wo_end). */
          size_t wo = 0;
          while (wo < safe_wo_end) {
            size_t vl = __riscv_vsetvl_e32m4(safe_wo_end - wo);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(bias_v, vl);
            for (size_t ci = 0; ci < Cin_per_g; ++ci) {
              size_t ci_abs = g * Cin_per_g + ci;
              for (size_t kh = 0; kh < Kh; ++kh) {
                ptrdiff_t hi =
                    (ptrdiff_t)(ho * stride_h + kh) - (ptrdiff_t)pad_h;
                if (hi < 0 || (size_t)hi >= Hin) continue;
                const float* row_base =
                    in + ((n * Cin + ci_abs) * Hin + (size_t)hi) * Win;
                for (size_t kw = 0; kw < Kw; ++kw) {
                  vfloat32m4_t vx =
                      __riscv_vle32_v_f32m4(row_base + wo + kw, vl);
                  float w = weight
                      [((co_abs * Cin_per_g + ci) * Kh + kh) * Kw + kw];
                  acc = __riscv_vfmacc_vf_f32m4(acc, w, vx, vl);
                }
              }
            }
            __riscv_vse32_v_f32m4(out_row + wo, acc, vl);
            wo += vl;
          }
          /* Scalar tail: (Wout - safe_wo_end) right-edge columns where
           * the kw reads might overrun `in`. Tiny relative cost — Kw-1
           * columns per (n, co, ho). */
          for (size_t wo_tail = safe_wo_end; wo_tail < Wout; ++wo_tail) {
            float acc = bias_v;
            for (size_t ci = 0; ci < Cin_per_g; ++ci) {
              size_t ci_abs = g * Cin_per_g + ci;
              for (size_t kh = 0; kh < Kh; ++kh) {
                ptrdiff_t hi =
                    (ptrdiff_t)(ho * stride_h + kh) - (ptrdiff_t)pad_h;
                if (hi < 0 || (size_t)hi >= Hin) continue;
                for (size_t kw = 0; kw < Kw; ++kw) {
                  size_t wi = wo_tail + kw;
                  if (wi >= Win) continue;
                  float x = in
                      [((n * Cin + ci_abs) * Hin + (size_t)hi) * Win + wi];
                  float w = weight
                      [((co_abs * Cin_per_g + ci) * Kh + kh) * Kw + kw];
                  acc += x * w;
                }
              }
            }
            out_row[wo_tail] = acc;
          }
        }
      }
    }
  }
}
