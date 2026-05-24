/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * y[n, c, h, w] = (x[n, c, h, w] - mean[c]) * inv_std[c] * weight[c] + bias[c]
 *   inv_std[c] = 1 / sqrt(var[c] + eps)
 *
 * Precomputes per-channel scale + offset once so the inner H*W loop is two
 * loads + fma + store per element. weight/bias may be NULL (affine=False).
 */

#include <math.h>
#include <stddef.h>

void riscv_batch_norm_f32_nchw_scalar(
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
    float w = (weight != NULL) ? weight[c] : 1.0f;
    float b = (bias != NULL) ? bias[c] : 0.0f;
    float scale = w * inv_std;
    float offset = b - mean[c] * scale;
    for (size_t n = 0; n < N; ++n) {
      const float* in_p = in + (n * C + c) * HW;
      float* out_p = out + (n * C + c) * HW;
      for (size_t i = 0; i < HW; ++i) {
        out_p[i] = in_p[i] * scale + offset;
      }
    }
  }
}
