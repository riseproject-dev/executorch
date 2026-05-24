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
 * Public dispatcher entry points (one per kernel). Each entry point resolves
 * the best per-variant impl lazily on first use via riscv_features_detect()
 * and caches the chosen function pointer. Per-variant symbols live in
 * kernels/<variant>/op_<name>.c and are gated by RISCV_KERNELS_HAVE_<V>=1.
 *
 * RISCV_DECLARE_KERNEL collapses the three declarations (entry point +
 * scalar variant + rvv variant) into one line so adding an op is a few-line
 * change here, not a forest of typedefs.
 */
#define RISCV_DECLARE_KERNEL(name, signature) \
  typedef void (*riscv_##name##_fn) signature; \
  void riscv_##name signature;                 \
  void riscv_##name##_scalar signature;        \
  void riscv_##name##_rvv signature;

RISCV_DECLARE_KERNEL(
    add_f32,
    (const float* a, const float* b, float* out, size_t n))

RISCV_DECLARE_KERNEL(
    hardtanh_f32,
    (const float* in, float* out, size_t n, float min_v, float max_v))

RISCV_DECLARE_KERNEL(relu_f32, (const float* in, float* out, size_t n))

RISCV_DECLARE_KERNEL(
    mul_f32,
    (const float* a, const float* b, float* out, size_t n))

/* mean over the last `inner` elements of every row; rows are `outer` long.
 * Used for the (B, C, H, W) -> (B, C) reduction that mobilenetv2/resnet18
 * emit via mean.dim before the final fc; portable's mean handles arbitrary
 * dims via stride math, we keep this PoC contiguous-only. */
RISCV_DECLARE_KERNEL(
    mean_dim_f32_contig,
    (const float* in, float* out, size_t outer, size_t inner))

/* out[m, n] = beta * c[m, n] + alpha * sum_k a[m, k] * b[k, n].
 * Tight scalar triple-loop, no blocking — placeholder for an RVV gemm later. */
RISCV_DECLARE_KERNEL(
    addmm_f32,
    (const float* a,
     const float* b,
     const float* c,
     float* out,
     size_t M,
     size_t N,
     size_t K,
     float alpha,
     float beta))

/* Fused affine + batch norm. y = (x - mean) / sqrt(var + eps) * gamma + beta.
 * Inputs assumed contiguous NCHW with weight/bias/mean/var indexed by channel. */
RISCV_DECLARE_KERNEL(
    batch_norm_f32_nchw,
    (const float* in,
     const float* weight, /* may be NULL */
     const float* bias, /* may be NULL */
     const float* mean,
     const float* var,
     float eps,
     float* out,
     size_t N,
     size_t C,
     size_t HW))

/* Generic im2col-style conv2d. Forwards the heavy lifting to the portable
 * kernel layout via a tight loop over output positions. No depth-wise or
 * grouped-conv specialisation in the scalar PoC — that's an rvv-pass target. */
RISCV_DECLARE_KERNEL(
    convolution_f32,
    (const float* in,
     const float* weight,
     const float* bias, /* may be NULL */
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
     size_t groups))

RISCV_DECLARE_KERNEL(
    max_pool2d_f32,
    (const float* in,
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
     size_t dilation_w))

#ifdef __cplusplus
} /* extern "C" */
#endif
