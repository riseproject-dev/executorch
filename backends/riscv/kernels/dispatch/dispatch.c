/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Single TU for every kernel's runtime-dispatch trampoline. Compiled with
 * the toolchain's baseline -march (no V); each riscv_<name>(...) entry
 * picks the highest-priority variant that satisfies riscv_features_detect()
 * and caches the function pointer for subsequent calls. Variants not built
 * into this binary are dropped via RISCV_KERNELS_HAVE_<V> from CMake.
 *
 * Adding an op:
 *   1. RISCV_DECLARE_KERNEL it in ops/riscv_ops_common.h.
 *   2. Implement riscv_<name>_scalar (and riscv_<name>_rvv, even if it's
 *      just a tail-call to the scalar version) in kernels/{scalar,rvv}/.
 *   3. RISCV_DISPATCH_F32_RVV(<name>, <arg list>, <arg names>) here.
 */

#include "../../runtime/riscv_features.h"
#include "../../ops/riscv_ops_common.h"

#include <stddef.h>

/* Emit a riscv_<name>() trampoline that chooses between rvv (when V is
 * detected and the rvv TU was compiled in) and scalar. RVV-only chosen
 * over scalar when both are available. With only scalar in the build the
 * compiler folds the table to a direct call. */
#define RISCV_DISPATCH_RVV_THEN_SCALAR(name, params, args)                  \
  static riscv_##name##_fn g_chosen_##name = NULL;                          \
  static riscv_##name##_fn choose_##name(void) {                            \
    const riscv_features_t* f = riscv_features_detect();                    \
    (void)f;                                                                \
    if (0) {                                                                \
    }                                                                       \
    RISCV_DISPATCH_RVV_ARM(name)                                            \
    RISCV_DISPATCH_SCALAR_ARM(name)                                         \
    return NULL;                                                            \
  }                                                                         \
  void riscv_##name params {                                                \
    riscv_##name##_fn fn = g_chosen_##name;                                 \
    if (fn == NULL) {                                                       \
      fn = choose_##name();                                                 \
      g_chosen_##name = fn;                                                 \
    }                                                                       \
    fn args;                                                                \
  }

#ifdef RISCV_KERNELS_HAVE_RVV
#define RISCV_DISPATCH_RVV_ARM(name) \
  else if ((f->bits & RISCV_FEATURE_V) != 0) { return riscv_##name##_rvv; }
#else
#define RISCV_DISPATCH_RVV_ARM(name)
#endif

#ifdef RISCV_KERNELS_HAVE_SCALAR
#define RISCV_DISPATCH_SCALAR_ARM(name) else { return riscv_##name##_scalar; }
#else
#define RISCV_DISPATCH_SCALAR_ARM(name)
#endif

/* clang-format off */
RISCV_DISPATCH_RVV_THEN_SCALAR(
    add_f32,
    (const float* a, const float* b, float* out, size_t n),
    (a, b, out, n))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    hardtanh_f32,
    (const float* in, float* out, size_t n, float lo, float hi),
    (in, out, n, lo, hi))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    relu_f32,
    (const float* in, float* out, size_t n),
    (in, out, n))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    mul_f32,
    (const float* a, const float* b, float* out, size_t n),
    (a, b, out, n))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    mean_dim_f32_contig,
    (const float* in, float* out, size_t outer, size_t inner),
    (in, out, outer, inner))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    addmm_f32,
    (const float* a, const float* b, const float* c, float* out,
     size_t M, size_t N, size_t K, float alpha, float beta),
    (a, b, c, out, M, N, K, alpha, beta))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    batch_norm_f32_nchw,
    (const float* in, const float* weight, const float* bias,
     const float* mean, const float* var, float eps,
     float* out, size_t N, size_t C, size_t HW),
    (in, weight, bias, mean, var, eps, out, N, C, HW))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    convolution_f32,
    (const float* in, const float* weight, const float* bias, float* out,
     size_t N, size_t Cin, size_t Hin, size_t Win,
     size_t Cout, size_t Hout, size_t Wout,
     size_t Kh, size_t Kw, size_t stride_h, size_t stride_w,
     size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w,
     size_t groups),
    (in, weight, bias, out, N, Cin, Hin, Win, Cout, Hout, Wout,
     Kh, Kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups))

RISCV_DISPATCH_RVV_THEN_SCALAR(
    max_pool2d_f32,
    (const float* in, float* out, int64_t* indices,
     size_t N, size_t C, size_t Hin, size_t Win, size_t Hout, size_t Wout,
     size_t Kh, size_t Kw, size_t stride_h, size_t stride_w,
     size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w),
    (in, out, indices, N, C, Hin, Win, Hout, Wout,
     Kh, Kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w))
/* clang-format on */
