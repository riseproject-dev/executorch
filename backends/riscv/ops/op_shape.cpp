/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Layout / memory-reshape ops. These don't compute anything — they shuffle
 * bytes — so the implementations live directly in C++ instead of going
 * through the dispatch table. view_copy and the dim_order-tagged clone are
 * straight memcpy; permute_copy is the only one that needs strided work.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

#include <cstring>

namespace riscv {
namespace native {

namespace {

size_t element_size(ScalarType st) {
  using executorch::aten::elementSize;
  return elementSize(st);
}

} // namespace

Tensor& view_copy_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> /*size*/,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == out.scalar_type(),
      "riscv::view_copy dtype mismatch");
  ET_CHECK_MSG(
      self.numel() == out.numel(),
      "riscv::view_copy numel mismatch (%zd vs %zd)",
      (ssize_t)self.numel(),
      (ssize_t)out.numel());
  size_t bytes = (size_t)self.numel() * element_size(self.scalar_type());
  std::memcpy(out.mutable_data_ptr(), self.const_data_ptr(), bytes);
  return out;
}

Tensor& _clone_dim_order_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    bool /*non_blocking*/,
    executorch::aten::OptionalArrayRef<int64_t> /*dim_order*/,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == out.scalar_type() && self.numel() == out.numel(),
      "riscv::_clone_dim_order shape/dtype mismatch");
  size_t bytes = (size_t)self.numel() * element_size(self.scalar_type());
  std::memcpy(out.mutable_data_ptr(), self.const_data_ptr(), bytes);
  return out;
}

namespace {

/* General N-D permute: walks the output linearly, computes the source offset
 * from the output's multi-index reinterpreted through the permutation. Slow,
 * but correct for any rank/dtype combination we hit. */
template <typename T>
void permute_copy_impl(
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> dims,
    Tensor& out) {
  size_t ndim = (size_t)self.dim();
  const T* src = self.const_data_ptr<T>();
  T* dst = out.mutable_data_ptr<T>();

  int64_t src_strides[16];
  int64_t out_sizes[16];
  ET_CHECK_MSG(ndim <= 16, "riscv::permute_copy: ndim>16 not supported");

  /* Default row-major strides on the source. */
  int64_t s = 1;
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    src_strides[i] = s;
    s *= self.size(i);
  }
  for (size_t i = 0; i < ndim; ++i) {
    out_sizes[i] = out.size(i);
  }

  int64_t total = self.numel();
  for (int64_t flat = 0; flat < total; ++flat) {
    /* Decode `flat` against the output sizes -> per-axis indices, then
     * re-index the source using dims[axis] as the source axis number. */
    int64_t rem = flat;
    int64_t src_off = 0;
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
      int64_t idx = rem % out_sizes[i];
      rem /= out_sizes[i];
      src_off += idx * src_strides[dims[i]];
    }
    dst[flat] = src[src_off];
  }
}

} // namespace

Tensor& permute_copy_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> dims,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == out.scalar_type(),
      "riscv::permute_copy dtype mismatch");
  ET_CHECK_MSG(
      (size_t)dims.size() == (size_t)self.dim(),
      "riscv::permute_copy dims rank mismatch");

  switch (self.scalar_type()) {
    case ScalarType::Float:
      permute_copy_impl<float>(self, dims, out);
      break;
    case ScalarType::Int:
      permute_copy_impl<int32_t>(self, dims, out);
      break;
    case ScalarType::Long:
      permute_copy_impl<int64_t>(self, dims, out);
      break;
    case ScalarType::Byte:
    case ScalarType::Char:
      permute_copy_impl<int8_t>(self, dims, out);
      break;
    default:
      ET_CHECK_MSG(false, "riscv::permute_copy unsupported dtype");
  }
  return out;
}

} // namespace native
} // namespace riscv
