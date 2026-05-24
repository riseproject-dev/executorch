/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Batched matmul. ConvertToRiscvPass forwards only the contiguous, fp32
 * case here; the C++ glue collapses the leading batch dims and hands the
 * (B, M, N, K) tuple to the C kernel.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

Tensor& bmm_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          mat2.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::bmm only supports float32");
  ET_CHECK_MSG(self.dim() == 3 && mat2.dim() == 3 && out.dim() == 3,
               "riscv::bmm requires 3-D tensors");
  size_t B = (size_t)self.size(0);
  size_t M = (size_t)self.size(1);
  size_t K = (size_t)self.size(2);
  size_t N = (size_t)mat2.size(2);
  ET_CHECK_MSG(
      (size_t)mat2.size(0) == B && (size_t)mat2.size(1) == K &&
          (size_t)out.size(0) == B && (size_t)out.size(1) == M &&
          (size_t)out.size(2) == N,
      "riscv::bmm shape mismatch");

  riscv_bmm_f32(
      self.const_data_ptr<float>(),
      mat2.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      B, M, N, K);
  return out;
}

Tensor& mm_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          mat2.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::mm only supports float32");
  ET_CHECK_MSG(self.dim() == 2 && mat2.dim() == 2 && out.dim() == 2,
               "riscv::mm requires 2-D tensors");
  size_t M = (size_t)self.size(0);
  size_t K = (size_t)self.size(1);
  size_t N = (size_t)mat2.size(1);
  ET_CHECK_MSG((size_t)mat2.size(0) == K && (size_t)out.size(0) == M &&
                   (size_t)out.size(1) == N,
               "riscv::mm shape mismatch");
  riscv_mm_f32(
      self.const_data_ptr<float>(),
      mat2.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      M, N, K);
  return out;
}

} // namespace native
} // namespace riscv
