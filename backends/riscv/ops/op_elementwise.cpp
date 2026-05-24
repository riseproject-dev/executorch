/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Element-wise unary + binary kernels. ConvertToRiscvPass only routes
 * matching-shape, float32, no-broadcast cases here, so the C++ glue stays
 * a few type-checks + a single call into the runtime dispatcher.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {
namespace {

/* aten::hardtanh's min/max often arrive as integer Scalars (relu6 -> 0..6),
 * and executorch's Scalar::to<double>() asserts isFloatingPoint(). Hand-roll
 * a numeric cast that handles both tags. */
inline float scalar_to_float(const executorch::aten::Scalar& s) {
  if (s.isFloatingPoint()) {
    return static_cast<float>(s.to<double>());
  }
  if (s.isIntegral(/*includeBool=*/false)) {
    return static_cast<float>(s.to<int64_t>());
  }
  if (s.isBoolean()) {
    return s.to<bool>() ? 1.0f : 0.0f;
  }
  ET_CHECK_MSG(false, "riscv: unsupported scalar tag");
  return 0.0f;
}

} // namespace

Tensor& hardtanh_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const executorch::aten::Scalar& min_val,
    const executorch::aten::Scalar& max_val,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::hardtanh only supports float32");
  ET_CHECK_MSG(
      self.numel() == out.numel(),
      "riscv::hardtanh numel mismatch (%zd vs %zd)",
      (ssize_t)self.numel(),
      (ssize_t)out.numel());
  riscv_hardtanh_f32(
      self.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(self.numel()),
      scalar_to_float(min_val),
      scalar_to_float(max_val));
  return out;
}

Tensor& relu_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::relu only supports float32");
  ET_CHECK_MSG(
      self.numel() == out.numel(),
      "riscv::relu numel mismatch (%zd vs %zd)",
      (ssize_t)self.numel(),
      (ssize_t)out.numel());
  riscv_relu_f32(
      self.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(self.numel()));
  return out;
}

Tensor& mul_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          other.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::mul only supports float32");
  ET_CHECK_MSG(
      self.numel() == other.numel() && self.numel() == out.numel(),
      "riscv::mul numel mismatch (%zd, %zd, %zd)",
      (ssize_t)self.numel(),
      (ssize_t)other.numel(),
      (ssize_t)out.numel());
  riscv_mul_f32(
      self.const_data_ptr<float>(),
      other.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(self.numel()));
  return out;
}

} // namespace native
} // namespace riscv
