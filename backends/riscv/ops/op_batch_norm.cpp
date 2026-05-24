/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Inference-only batch norm. save_mean / save_invstd are unused in
 * _legit_no_training (output tensors exist solely to satisfy the codegen
 * signature) so we leave them untouched and only fill `out`.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

::std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_training_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const executorch::aten::optional<Tensor>& weight,
    const executorch::aten::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_invstd) {
  (void)context;
  (void)momentum;
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::batch_norm only supports float32");
  ET_CHECK_MSG(input.dim() == 4, "riscv::batch_norm expects NCHW");
  size_t N = (size_t)input.size(0);
  size_t C = (size_t)input.size(1);
  size_t HW = (size_t)(input.size(2) * input.size(3));

  const float* w_p =
      weight.has_value() ? weight.value().const_data_ptr<float>() : nullptr;
  const float* b_p =
      bias.has_value() ? bias.value().const_data_ptr<float>() : nullptr;

  riscv_batch_norm_f32_nchw(
      input.const_data_ptr<float>(),
      w_p,
      b_p,
      running_mean.const_data_ptr<float>(),
      running_var.const_data_ptr<float>(),
      static_cast<float>(eps),
      out.mutable_data_ptr<float>(),
      N,
      C,
      HW);
  return {out, save_mean, save_invstd};
}

} // namespace native
} // namespace riscv
