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

/* expand_copy / unsqueeze_copy / slice_copy: same byte-shuffling pattern as
 * permute_copy. The output tensor already has the correct shape (alloc'd by
 * the memory planner); we walk it and read from the matching source offset.
 */

namespace {

template <typename T>
void expand_copy_impl(const Tensor& self, Tensor& out) {
  /* expand_copy materialises broadcast: stride along an axis is 0 when the
   * source size there is 1 and the output is larger. */
  size_t ndim = (size_t)out.dim();
  ET_CHECK_MSG(ndim <= 16, "riscv::expand_copy: ndim>16 not supported");
  size_t src_ndim = (size_t)self.dim();
  /* Right-align src dims against out dims; missing leading dims = 1. */
  int64_t src_sizes[16];
  int64_t out_sizes[16];
  for (size_t i = 0; i < ndim; ++i) {
    out_sizes[i] = out.size(i);
    int64_t off = (int64_t)i - (int64_t)(ndim - src_ndim);
    src_sizes[i] = (off < 0) ? 1 : self.size((int)off);
  }
  int64_t src_strides[16];
  int64_t s = 1;
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    /* Stride along a broadcast axis is 0. */
    src_strides[i] = (src_sizes[i] == out_sizes[i]) ? s : 0;
    s *= src_sizes[i];
  }
  const T* src = self.const_data_ptr<T>();
  T* dst = out.mutable_data_ptr<T>();
  int64_t total = out.numel();
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t rem = flat, src_off = 0;
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
      int64_t idx = rem % out_sizes[i];
      rem /= out_sizes[i];
      src_off += idx * src_strides[i];
    }
    dst[flat] = src[src_off];
  }
}

template <typename T>
void slice_copy_impl(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t step,
    Tensor& out) {
  size_t ndim = (size_t)self.dim();
  ET_CHECK_MSG(ndim <= 16, "riscv::slice_copy: ndim>16 not supported");
  int64_t src_strides[16];
  int64_t out_sizes[16];
  int64_t s = 1;
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    src_strides[i] = s;
    s *= self.size(i);
  }
  for (size_t i = 0; i < ndim; ++i) {
    out_sizes[i] = out.size(i);
  }
  const T* src = self.const_data_ptr<T>();
  T* dst = out.mutable_data_ptr<T>();
  int64_t total = out.numel();
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t rem = flat, src_off = 0;
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
      int64_t idx = rem % out_sizes[i];
      rem /= out_sizes[i];
      int64_t src_idx = (i == (size_t)dim) ? (start + idx * step) : idx;
      src_off += src_idx * src_strides[i];
    }
    dst[flat] = src[src_off];
  }
}

#define SHAPE_DTYPE_DISPATCH(stmt) \
  switch (self.scalar_type()) {    \
    case ScalarType::Float: { typedef float _T; stmt; break; } \
    case ScalarType::Int: { typedef int32_t _T; stmt; break; } \
    case ScalarType::Long: { typedef int64_t _T; stmt; break; } \
    case ScalarType::Byte: case ScalarType::Char: { typedef int8_t _T; stmt; break; } \
    case ScalarType::Bool: { typedef bool _T; stmt; break; } \
    default: ET_CHECK_MSG(false, "riscv::shape op: unsupported dtype"); \
  }

} // namespace

Tensor& expand_copy_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> /*size*/,
    bool /*implicit*/,
    Tensor& out) {
  (void)context;
  SHAPE_DTYPE_DISPATCH(expand_copy_impl<_T>(self, out));
  return out;
}

Tensor& unsqueeze_copy_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t /*dim*/,
    Tensor& out) {
  (void)context;
  /* unsqueeze inserts a size-1 axis without moving data → straight memcpy. */
  ET_CHECK_MSG(self.numel() == out.numel(), "riscv::unsqueeze_copy numel mismatch");
  size_t bytes = (size_t)self.numel() * element_size(self.scalar_type());
  std::memcpy(out.mutable_data_ptr(), self.const_data_ptr(), bytes);
  return out;
}

Tensor& slice_copy_Tensor_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    executorch::aten::optional<int64_t> start_opt,
    executorch::aten::optional<int64_t> end_opt,
    int64_t step,
    Tensor& out) {
  (void)context;
  (void)end_opt;
  int64_t ndim = self.dim();
  if (dim < 0) dim += ndim;
  int64_t start = start_opt.has_value() ? start_opt.value() : 0;
  if (start < 0) start += self.size((int)dim);
  if (start < 0) start = 0;
  SHAPE_DTYPE_DISPATCH(slice_copy_impl<_T>(self, dim, start, step, out));
  return out;
}

Tensor& cat_out(
    KernelRuntimeContext& context,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(tensors.size() > 0, "riscv::cat: empty input list");
  ScalarType dt = tensors[0].scalar_type();
  size_t esz = element_size(dt);
  int64_t ndim = out.dim();
  if (dim < 0) dim += ndim;
  /* Outer / inner block sizes: outer = prod(sizes[:dim]),
   * inner_per_input[i] = sizes[i][dim] * prod(sizes[dim+1:]) bytes. */
  int64_t outer = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= out.size((int)i);
  int64_t trailing = 1;
  for (int64_t i = dim + 1; i < ndim; ++i) trailing *= out.size((int)i);
  uint8_t* dst = (uint8_t*)out.mutable_data_ptr();
  size_t dst_stride = (size_t)(out.size((int)dim) * trailing) * esz;
  for (int64_t o = 0; o < outer; ++o) {
    size_t off = 0;
    for (size_t t = 0; t < tensors.size(); ++t) {
      const Tensor& src = tensors[t];
      size_t src_dim_sz = (size_t)src.size((int)dim);
      size_t chunk = src_dim_sz * (size_t)trailing * esz;
      const uint8_t* sp =
          (const uint8_t*)src.const_data_ptr() + (size_t)o * chunk;
      std::memcpy(dst + (size_t)o * dst_stride + off, sp, chunk);
      off += chunk;
    }
  }
  return out;
}

} // namespace native
} // namespace riscv
