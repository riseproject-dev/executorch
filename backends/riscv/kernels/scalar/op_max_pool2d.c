/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * NCHW max-pool with optional indices. Padded reads are skipped; if every
 * position in a window is padded the output is left at -INFINITY (the
 * portable kernel does the same). Indices, when requested, are absolute
 * input offsets in (h * Win + w) coordinates per channel — matches what
 * portable's max_pool2d_with_indices emits.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void riscv_max_pool2d_f32_scalar(
    const float* in,
    float* out,
    int64_t* indices,
    size_t N,
    size_t C,
    size_t Hin,
    size_t Win,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w) {
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      const float* in_p = in + (n * C + c) * Hin * Win;
      float* out_p = out + (n * C + c) * Hout * Wout;
      int64_t* idx_p =
          (indices != NULL) ? (indices + (n * C + c) * Hout * Wout) : NULL;
      for (size_t ho = 0; ho < Hout; ++ho) {
        for (size_t wo = 0; wo < Wout; ++wo) {
          float best = -INFINITY;
          int64_t best_idx = -1;
          for (size_t kh = 0; kh < Kh; ++kh) {
            ptrdiff_t hi =
                (ptrdiff_t)(ho * stride_h + kh * dilation_h) - (ptrdiff_t)pad_h;
            if (hi < 0 || (size_t)hi >= Hin)
              continue;
            for (size_t kw = 0; kw < Kw; ++kw) {
              ptrdiff_t wi = (ptrdiff_t)(wo * stride_w + kw * dilation_w) -
                  (ptrdiff_t)pad_w;
              if (wi < 0 || (size_t)wi >= Win)
                continue;
              size_t off = (size_t)hi * Win + (size_t)wi;
              float v = in_p[off];
              if (v > best || best_idx < 0) {
                best = v;
                best_idx = (int64_t)off;
              }
            }
          }
          out_p[ho * Wout + wo] = best;
          if (idx_p != NULL)
            idx_p[ho * Wout + wo] = best_idx;
        }
      }
    }
  }
}
