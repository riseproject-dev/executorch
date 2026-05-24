/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * aten::addmm(self, mat1, mat2, beta, alpha) -> self * beta + mat1 @ mat2 * alpha.
 * self is broadcastable to (M, N); ConvertToRiscvPass only forwards the
 * non-broadcast / 1-D bias case below.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

#include <cstring>

namespace riscv {
namespace native {

namespace {

inline float scalar_to_float(const executorch::aten::Scalar& s) {
  if (s.isFloatingPoint()) return static_cast<float>(s.to<double>());
  if (s.isIntegral(false)) return static_cast<float>(s.to<int64_t>());
  if (s.isBoolean()) return s.to<bool>() ? 1.0f : 0.0f;
  ET_CHECK_MSG(false, "riscv: unsupported scalar tag");
  return 0.0f;
}


void broadcast_bias_to_mn(
    const float* src,
    const Tensor& src_t,
    float* dst,
    size_t M,
    size_t N) {
  /* The two shapes we expect from ConvertToRiscvPass: full (M, N) and 1-D
   * (N,) broadcast across rows. Anything else should have been left for
   * portable. */
  size_t ndim = src_t.dim();
  if (ndim == 2 && (size_t)src_t.size(0) == M && (size_t)src_t.size(1) == N) {
    std::memcpy(dst, src, sizeof(float) * M * N);
    return;
  }
  if (ndim == 1 && (size_t)src_t.size(0) == N) {
    for (size_t m = 0; m < M; ++m) {
      std::memcpy(dst + m * N, src, sizeof(float) * N);
    }
    return;
  }
  ET_CHECK_MSG(
      false,
      "riscv::addmm unsupported bias shape (ndim=%zu)",
      ndim);
}

} // namespace

Tensor& addmm_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const executorch::aten::Scalar& beta,
    const executorch::aten::Scalar& alpha,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          mat1.scalar_type() == ScalarType::Float &&
          mat2.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::addmm only supports float32");
  ET_CHECK_MSG(
      mat1.dim() == 2 && mat2.dim() == 2 && out.dim() == 2,
      "riscv::addmm expects 2-D mat1/mat2/out");
  size_t M = (size_t)mat1.size(0);
  size_t K = (size_t)mat1.size(1);
  size_t N = (size_t)mat2.size(1);
  ET_CHECK_MSG(
      (size_t)mat2.size(0) == K && (size_t)out.size(0) == M &&
          (size_t)out.size(1) == N,
      "riscv::addmm shape mismatch (M=%zu, K=%zu, N=%zu)",
      M,
      K,
      N);

  /* Materialise self into (M, N) so the C kernel sees a contiguous bias. */
  float beta_v = scalar_to_float(beta);
  float alpha_v = scalar_to_float(alpha);
  float* out_p = out.mutable_data_ptr<float>();
  if (beta_v != 0.0f) {
    broadcast_bias_to_mn(self.const_data_ptr<float>(), self, out_p, M, N);
    riscv_addmm_f32(
        mat1.const_data_ptr<float>(),
        mat2.const_data_ptr<float>(),
        out_p,
        out_p,
        M,
        N,
        K,
        alpha_v,
        beta_v);
  } else {
    riscv_addmm_f32(
        mat1.const_data_ptr<float>(),
        mat2.const_data_ptr<float>(),
        nullptr,
        out_p,
        M,
        N,
        K,
        alpha_v,
        0.0f);
  }
  return out;
}

} // namespace native
} // namespace riscv
