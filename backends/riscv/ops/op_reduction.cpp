/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Reduction ops. mean.dim handles the (B, C, H, W) -> (B, C) collapse the
 * vision models emit before their final fc; the kernel takes a contiguous
 * (outer, inner) view so the C side stays a two-loop reduction.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

Tensor& mean_dim_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    executorch::aten::OptionalArrayRef<int64_t> dim,
    bool keepdim,
    executorch::aten::optional<ScalarType> dtype,
    Tensor& out) {
  (void)context;
  (void)keepdim;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::mean.dim only supports float32");
  ET_CHECK_MSG(
      !dtype.has_value() || dtype.value() == ScalarType::Float,
      "riscv::mean.dim dtype override not supported");
  ET_CHECK_MSG(dim.has_value(), "riscv::mean.dim requires explicit dim list");

  /* The pass only forwards the contiguous tail-reduction case: dims must
   * be the last K axes. We reconstruct (outer, inner) directly from sizes
   * since we can rely on that contract. */
  auto dims = dim.value();
  size_t ndim = (size_t)self.dim();
  size_t kept = ndim - (size_t)dims.size();
  size_t outer = 1, inner = 1;
  for (size_t i = 0; i < kept; ++i) {
    outer *= (size_t)self.size(i);
  }
  for (size_t i = kept; i < ndim; ++i) {
    inner *= (size_t)self.size(i);
  }
  ET_CHECK_MSG(
      (size_t)out.numel() == outer,
      "riscv::mean.dim output numel (%zd) != outer (%zu)",
      (ssize_t)out.numel(),
      outer);
  riscv_mean_dim_f32_contig(
      self.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      outer,
      inner);
  return out;
}

} // namespace native
} // namespace riscv
