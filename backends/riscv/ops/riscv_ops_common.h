/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <executorch/runtime/kernel/kernel_includes.h>

using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Error = executorch::runtime::Error;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

extern "C" {
#endif

/*
 * Public dispatcher entry points. Each lowered op has a single
 * extern "C" entry point that the C++ glue calls; the dispatcher
 * (kernels/dispatch/) resolves the per-variant implementation lazily on
 * first use using riscv_features_detect().
 */
void riscv_add_f32(const float* a, const float* b, float* out, size_t n);

/*
 * Per-variant kernel symbols. Declared here so the dispatcher can build its
 * selection table; the symbols are defined in kernels/<variant>/op_add.c
 * compiled with the appropriate -march= flag. Variants that aren't compiled
 * into the build are excluded via RISCV_KERNELS_HAVE_<VARIANT> macros set
 * by CMake.
 */
typedef void (*riscv_add_f32_fn)(
    const float* a,
    const float* b,
    float* out,
    size_t n);

void riscv_add_f32_scalar(
    const float* a,
    const float* b,
    float* out,
    size_t n);
void riscv_add_f32_rvv(const float* a, const float* b, float* out, size_t n);
void riscv_add_f32_p(const float* a, const float* b, float* out, size_t n);
void riscv_add_f32_vme(const float* a, const float* b, float* out, size_t n);
void riscv_add_f32_ime(const float* a, const float* b, float* out, size_t n);
void riscv_add_f32_ame(const float* a, const float* b, float* out, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif
