/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Baseline scalar fp32 add. Compiled with -march=rv64gc; safe to run on any
 * RV64GC CPU. Acts as the fallback when no extension-specific variant is
 * selected by the dispatcher.
 */

#include <stddef.h>

void riscv_add_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}
