/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

Tensor& add_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  (void)context;
  ET_CHECK_MSG(
      self.scalar_type() == ScalarType::Float &&
          other.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::add only supports float32");
  ET_CHECK_MSG(
      self.numel() == other.numel() && self.numel() == out.numel(),
      "riscv::add requires matching shapes (numel=%zd, %zd, %zd)",
      (ssize_t)self.numel(),
      (ssize_t)other.numel(),
      (ssize_t)out.numel());

  riscv_add_f32(
      self.const_data_ptr<float>(),
      other.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      static_cast<size_t>(self.numel()));
  return out;
}

} // namespace native
} // namespace riscv
