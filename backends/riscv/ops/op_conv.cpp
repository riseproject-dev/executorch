/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * 2-D NCHW convolution. ConvertToRiscvPass filters out everything the
 * scalar kernel doesn't handle (transposed=true, non-NCHW layouts) so the
 * glue only has to validate the contracted shape and unpack the int-array
 * args before handing off.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

Tensor& convolution_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const executorch::aten::optional<Tensor>& bias,
    executorch::aten::ArrayRef<int64_t> stride,
    executorch::aten::ArrayRef<int64_t> padding,
    executorch::aten::ArrayRef<int64_t> dilation,
    bool transposed,
    executorch::aten::ArrayRef<int64_t> output_padding,
    int64_t groups,
    Tensor& out) {
  (void)context;
  (void)output_padding;
  ET_CHECK_MSG(!transposed, "riscv::convolution: transposed=true not supported");
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Float &&
          weight.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::convolution only supports float32");
  ET_CHECK_MSG(
      input.dim() == 4 && weight.dim() == 4 && out.dim() == 4,
      "riscv::convolution expects 4-D NCHW input/weight/out");
  ET_CHECK_MSG(
      stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
      "riscv::convolution expects 2-D stride/padding/dilation");

  const float* b_p =
      bias.has_value() ? bias.value().const_data_ptr<float>() : nullptr;

  riscv_convolution_f32(
      input.const_data_ptr<float>(),
      weight.const_data_ptr<float>(),
      b_p,
      out.mutable_data_ptr<float>(),
      (size_t)input.size(0),
      (size_t)input.size(1),
      (size_t)input.size(2),
      (size_t)input.size(3),
      (size_t)out.size(1),
      (size_t)out.size(2),
      (size_t)out.size(3),
      (size_t)weight.size(2),
      (size_t)weight.size(3),
      (size_t)stride[0],
      (size_t)stride[1],
      (size_t)padding[0],
      (size_t)padding[1],
      (size_t)dilation[0],
      (size_t)dilation[1],
      (size_t)groups);
  return out;
}

} // namespace native
} // namespace riscv
