/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <stddef.h>

void riscv_rsqrt_f32_scalar(const float* in, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / sqrtf(in[i]);
}
