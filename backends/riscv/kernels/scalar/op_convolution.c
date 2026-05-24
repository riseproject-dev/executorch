/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Generic NCHW 2-D conv, grouped + dilated, transposed=false. Iterates the
 * 7 nested loops (n, g, co, ho, wo, kh, kw, ci) directly — no im2col, no
 * blocking. Goal: be a clear, correct scalar baseline that the RVV variant
 * replaces with a tiled MAC inner loop.
 *
 * Layout: input/output are NCHW contiguous; weight is (Cout, Cin/groups,
 * Kh, Kw) contiguous; bias is (Cout,) or NULL.
 */

#include <stddef.h>

void riscv_convolution_f32_scalar(
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
  size_t Cin_per_group = Cin / groups;
  size_t Cout_per_group = Cout / groups;

  for (size_t n = 0; n < N; ++n) {
    for (size_t g = 0; g < groups; ++g) {
      for (size_t co = 0; co < Cout_per_group; ++co) {
        size_t co_abs = g * Cout_per_group + co;
        float bias_v = (bias != NULL) ? bias[co_abs] : 0.0f;
        for (size_t ho = 0; ho < Hout; ++ho) {
          for (size_t wo = 0; wo < Wout; ++wo) {
            float acc = bias_v;
            for (size_t ci = 0; ci < Cin_per_group; ++ci) {
              size_t ci_abs = g * Cin_per_group + ci;
              for (size_t kh = 0; kh < Kh; ++kh) {
                /* Cast to ptrdiff so negative (out-of-pad) coords work. */
                ptrdiff_t hi =
                    (ptrdiff_t)(ho * stride_h + kh * dilation_h) -
                    (ptrdiff_t)pad_h;
                if (hi < 0 || (size_t)hi >= Hin)
                  continue;
                for (size_t kw = 0; kw < Kw; ++kw) {
                  ptrdiff_t wi =
                      (ptrdiff_t)(wo * stride_w + kw * dilation_w) -
                      (ptrdiff_t)pad_w;
                  if (wi < 0 || (size_t)wi >= Win)
                    continue;
                  float x = in
                      [((n * Cin + ci_abs) * Hin + (size_t)hi) * Win +
                       (size_t)wi];
                  float w = weight
                      [((co_abs * Cin_per_group + ci) * Kh + kh) * Kw + kw];
                  acc += x * w;
                }
              }
            }
            out[((n * Cout + co_abs) * Hout + ho) * Wout + wo] = acc;
          }
        }
      }
    }
  }
}
