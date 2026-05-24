/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Element-wise + utility ops that the transformer / yolo models pull in.
 * Implemented directly in C++ — they're either trivial loops (where,
 * eq.Scalar) or tensor-fillers (full, scalar_tensor) where a dedicated C
 * kernel would be more boilerplate than body. RVV-targeted ops (mul, add)
 * go through the dispatcher in op_elementwise.cpp instead.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

#include <cmath>
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

inline int64_t scalar_to_int(const executorch::aten::Scalar& s) {
  if (s.isIntegral(false)) return s.to<int64_t>();
  if (s.isFloatingPoint()) return (int64_t)s.to<double>();
  if (s.isBoolean()) return s.to<bool>() ? 1 : 0;
  ET_CHECK_MSG(false, "riscv: unsupported scalar tag");
  return 0;
}

template <typename T>
void fill_value(Tensor& out, T v) {
  T* p = out.mutable_data_ptr<T>();
  for (int64_t i = 0; i < out.numel(); ++i) p[i] = v;
}

template <typename T>
void fill_value_dispatched(Tensor& out, double v) {
  fill_value<T>(out, static_cast<T>(v));
}

} // namespace

// -- mul.Scalar -------------------------------------------------------------

Tensor& mul_Scalar_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const executorch::aten::Scalar& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::mul.Scalar only supports float32");
  ET_CHECK_MSG(self.numel() == out.numel(), "riscv::mul.Scalar numel mismatch");
  float k = scalar_to_float(other);
  const float* sp = self.const_data_ptr<float>();
  float* op = out.mutable_data_ptr<float>();
  for (int64_t i = 0; i < self.numel(); ++i) op[i] = sp[i] * k;
  return out;
}

// -- where ------------------------------------------------------------------

Tensor& where_self_out(
    KernelRuntimeContext& context,
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == out.scalar_type() &&
          other.scalar_type() == out.scalar_type(),
      "riscv::where dtype mismatch");
  ET_CHECK_MSG(
      self.numel() == out.numel() && other.numel() == out.numel() &&
          condition.numel() == out.numel(),
      "riscv::where numel mismatch (no broadcast supported)");
  /* condition is bool in aten, but executorch's allocator may give a Byte
   * tensor — treat both as a 1-byte truthy test. */
  const uint8_t* c = (const uint8_t*)condition.const_data_ptr();
  if (out.scalar_type() == ScalarType::Float) {
    const float* a = self.const_data_ptr<float>();
    const float* b = other.const_data_ptr<float>();
    float* o = out.mutable_data_ptr<float>();
    for (int64_t i = 0; i < out.numel(); ++i) o[i] = c[i] ? a[i] : b[i];
  } else if (out.scalar_type() == ScalarType::Long) {
    const int64_t* a = self.const_data_ptr<int64_t>();
    const int64_t* b = other.const_data_ptr<int64_t>();
    int64_t* o = out.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < out.numel(); ++i) o[i] = c[i] ? a[i] : b[i];
  } else {
    ET_CHECK_MSG(false, "riscv::where unsupported dtype");
  }
  return out;
}

// -- logical_not ------------------------------------------------------------

Tensor& logical_not_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(self.numel() == out.numel(), "riscv::logical_not numel mismatch");
  /* Both are typically Bool (1 byte); aten guarantees the result is Bool. */
  const uint8_t* sp = (const uint8_t*)self.const_data_ptr();
  uint8_t* op = (uint8_t*)out.mutable_data_ptr();
  for (int64_t i = 0; i < self.numel(); ++i) op[i] = sp[i] ? 0 : 1;
  return out;
}

// -- eq.Scalar / ge.Scalar --------------------------------------------------

namespace {

template <bool IsGE>
void cmp_scalar(const Tensor& self, double val, Tensor& out) {
  uint8_t* op = (uint8_t*)out.mutable_data_ptr();
  int64_t n = self.numel();
  if (self.scalar_type() == ScalarType::Float) {
    const float* sp = self.const_data_ptr<float>();
    float k = (float)val;
    for (int64_t i = 0; i < n; ++i) {
      op[i] = IsGE ? (sp[i] >= k) : (sp[i] == k);
    }
  } else if (self.scalar_type() == ScalarType::Long) {
    const int64_t* sp = self.const_data_ptr<int64_t>();
    int64_t k = (int64_t)val;
    for (int64_t i = 0; i < n; ++i) {
      op[i] = IsGE ? (sp[i] >= k) : (sp[i] == k);
    }
  } else {
    ET_CHECK_MSG(false, "riscv::cmp.Scalar unsupported dtype");
  }
}

} // namespace

Tensor& eq_Scalar_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const executorch::aten::Scalar& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(self.numel() == out.numel(), "riscv::eq.Scalar numel mismatch");
  cmp_scalar<false>(self, (double)scalar_to_float(other), out);
  return out;
}

Tensor& ge_Scalar_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const executorch::aten::Scalar& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(self.numel() == out.numel(), "riscv::ge.Scalar numel mismatch");
  cmp_scalar<true>(self, (double)scalar_to_float(other), out);
  return out;
}

// -- full / full_like / scalar_tensor --------------------------------------

namespace {

void fill_tensor_with(Tensor& out, double v) {
  switch (out.scalar_type()) {
    case ScalarType::Float: fill_value_dispatched<float>(out, v); break;
    case ScalarType::Int: fill_value_dispatched<int32_t>(out, v); break;
    case ScalarType::Long: fill_value_dispatched<int64_t>(out, v); break;
    case ScalarType::Bool: fill_value_dispatched<bool>(out, v); break;
    case ScalarType::Byte: fill_value_dispatched<uint8_t>(out, v); break;
    case ScalarType::Char: fill_value_dispatched<int8_t>(out, v); break;
    default: ET_CHECK_MSG(false, "riscv::full: unsupported dtype");
  }
}

} // namespace

Tensor& full_out(
    KernelRuntimeContext& context,
    executorch::aten::ArrayRef<int64_t> /*size*/,
    const executorch::aten::Scalar& fill_value,
    Tensor& out) {
  (void)context;
  fill_tensor_with(out, (double)scalar_to_float(fill_value));
  return out;
}

Tensor& full_like_out(
    KernelRuntimeContext& context,
    const Tensor& /*self*/,
    const executorch::aten::Scalar& fill_value,
    executorch::aten::optional<executorch::aten::MemoryFormat> /*mf*/,
    Tensor& out) {
  (void)context;
  fill_tensor_with(out, (double)scalar_to_float(fill_value));
  return out;
}

Tensor& scalar_tensor_out(
    KernelRuntimeContext& context,
    const executorch::aten::Scalar& s,
    Tensor& out) {
  (void)context;
  fill_tensor_with(out, (double)scalar_to_float(s));
  return out;
}

// -- arange.start_step ------------------------------------------------------

Tensor& arange_start_step_out(
    KernelRuntimeContext& context,
    const executorch::aten::Scalar& start,
    const executorch::aten::Scalar& end,
    const executorch::aten::Scalar& step,
    Tensor& out) {
  (void)context;
  (void)end;
  double s = (double)scalar_to_float(start);
  double st = (double)scalar_to_float(step);
  int64_t n = out.numel();
  if (out.scalar_type() == ScalarType::Float) {
    float* op = out.mutable_data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) op[i] = (float)(s + i * st);
  } else if (out.scalar_type() == ScalarType::Long) {
    int64_t* op = out.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) op[i] = (int64_t)(s + i * st);
  } else {
    ET_CHECK_MSG(false, "riscv::arange unsupported dtype");
  }
  return out;
}

// -- constant_pad_nd --------------------------------------------------------

Tensor& constant_pad_nd_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    executorch::aten::ArrayRef<int64_t> pad,
    const executorch::aten::Scalar& value,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::constant_pad_nd only supports float32");
  /* pad is given as [last_dim_lo, last_dim_hi, second_last_lo, ...] in
   * aten convention — only the trailing 2*K entries are meaningful. We
   * walk the output linearly, decode multi-index against out sizes, then
   * subtract the lo pad from the trailing K axes to land on src. */
  size_t ndim = (size_t)out.dim();
  size_t pad_pairs = pad.size() / 2;
  ET_CHECK_MSG(pad_pairs <= ndim, "riscv::pad: too many pad entries");
  ET_CHECK_MSG(ndim <= 16, "riscv::pad: ndim>16 not supported");
  int64_t pad_lo[16] = {0};
  for (size_t i = 0; i < pad_pairs; ++i) {
    pad_lo[ndim - 1 - i] = pad[2 * i];
  }
  int64_t out_sizes[16];
  int64_t src_sizes[16];
  int64_t src_strides[16];
  int64_t s = 1;
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    out_sizes[i] = out.size((int)i);
    src_sizes[i] = self.size((int)i);
    src_strides[i] = s;
    s *= src_sizes[i];
  }
  float fill = scalar_to_float(value);
  const float* sp = self.const_data_ptr<float>();
  float* op = out.mutable_data_ptr<float>();
  int64_t total = out.numel();
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t rem = flat;
    int64_t src_off = 0;
    bool in_bounds = true;
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
      int64_t idx_out = rem % out_sizes[i];
      rem /= out_sizes[i];
      int64_t idx_src = idx_out - pad_lo[i];
      if (idx_src < 0 || idx_src >= src_sizes[i]) {
        in_bounds = false;
        break;
      }
      src_off += idx_src * src_strides[i];
    }
    op[flat] = in_bounds ? sp[src_off] : fill;
  }
  return out;
}

// -- embedding --------------------------------------------------------------

Tensor& embedding_out(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Tensor& indices,
    int64_t /*padding_idx*/,
    bool /*scale_grad*/,
    bool /*sparse*/,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      weight.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::embedding only supports float32 weight");
  ET_CHECK_MSG(weight.dim() == 2, "riscv::embedding weight must be 2-D");
  size_t emb_dim = (size_t)weight.size(1);
  size_t vocab = (size_t)weight.size(0);
  const float* w = weight.const_data_ptr<float>();
  float* o = out.mutable_data_ptr<float>();
  int64_t n = indices.numel();
  if (indices.scalar_type() == ScalarType::Long) {
    const int64_t* ix = indices.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      int64_t row = ix[i];
      ET_CHECK_MSG(row >= 0 && (size_t)row < vocab, "riscv::embedding OOB index");
      std::memcpy(o + i * emb_dim, w + (size_t)row * emb_dim, emb_dim * sizeof(float));
    }
  } else if (indices.scalar_type() == ScalarType::Int) {
    const int32_t* ix = indices.const_data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) {
      int32_t row = ix[i];
      ET_CHECK_MSG(row >= 0 && (size_t)row < vocab, "riscv::embedding OOB index");
      std::memcpy(o + i * emb_dim, w + (size_t)row * emb_dim, emb_dim * sizeof(float));
    }
  } else {
    ET_CHECK_MSG(false, "riscv::embedding indices must be int32/int64");
  }
  return out;
}

// -- any.dim ----------------------------------------------------------------

Tensor& any_dim_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  (void)context;
  (void)keepdim;
  int64_t ndim = self.dim();
  if (dim < 0) dim += ndim;
  /* Outer/inner contigous decomposition along `dim`; inner stride within
   * dim is `trailing`. Reduce by OR-ing the bytes. */
  int64_t trailing = 1;
  for (int64_t i = dim + 1; i < ndim; ++i) trailing *= self.size((int)i);
  int64_t reduce_n = self.size((int)dim);
  int64_t outer = self.numel() / (reduce_n * trailing);
  const uint8_t* sp = (const uint8_t*)self.const_data_ptr();
  uint8_t* op = (uint8_t*)out.mutable_data_ptr();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t t = 0; t < trailing; ++t) {
      uint8_t acc = 0;
      for (int64_t r = 0; r < reduce_n; ++r) {
        acc |= sp[(o * reduce_n + r) * trailing + t] ? 1 : 0;
      }
      op[o * trailing + t] = acc;
    }
  }
  return out;
}

} // namespace native
} // namespace riscv
