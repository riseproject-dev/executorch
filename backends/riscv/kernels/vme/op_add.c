/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Stub for the RISC-V Vector Matrix Extension (VME). The directory and
 * symbol exist so the dispatch table and CMake plumbing stay complete; the
 * real implementation is a follow-up. Forwards to the scalar variant so
 * the answer is still correct if the dispatcher ever routes here.
 */

#include <stddef.h>

void riscv_add_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n);

void riscv_add_f32_vme(const float* a, const float* b, float* out, size_t n) {
  riscv_add_f32_scalar(a, b, out, n);
}
