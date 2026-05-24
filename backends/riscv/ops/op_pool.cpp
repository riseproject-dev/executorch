/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * 2-D NCHW max-pool with mandatory indices tensor (matches the aten op's
 * out variant). `ceil_mode` isn't relevant here — output shape comes from
 * the caller-provided `out` tensor; we just walk it.
 */

#include "riscv_ops_common.h"

#include <executorch/runtime/platform/assert.h>

namespace riscv {
namespace native {

::std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    executorch::aten::ArrayRef<int64_t> kernel_size,
    executorch::aten::ArrayRef<int64_t> stride,
    executorch::aten::ArrayRef<int64_t> padding,
    executorch::aten::ArrayRef<int64_t> dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices) {
  (void)context;
  (void)ceil_mode;
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Float &&
          out.scalar_type() == ScalarType::Float,
      "riscv::max_pool2d only supports float32");
  ET_CHECK_MSG(input.dim() == 4 && out.dim() == 4, "riscv::max_pool2d expects NCHW");
  ET_CHECK_MSG(kernel_size.size() == 2, "riscv::max_pool2d expects 2-D kernel");

  int64_t sh = stride.size() > 0 ? stride[0] : kernel_size[0];
  int64_t sw = stride.size() > 0 ? stride[1] : kernel_size[1];
  int64_t ph = padding.size() > 0 ? padding[0] : 0;
  int64_t pw = padding.size() > 0 ? padding[1] : 0;
  int64_t dh = dilation.size() > 0 ? dilation[0] : 1;
  int64_t dw = dilation.size() > 0 ? dilation[1] : 1;

  riscv_max_pool2d_f32(
      input.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      indices.mutable_data_ptr<int64_t>(),
      (size_t)input.size(0),
      (size_t)input.size(1),
      (size_t)input.size(2),
      (size_t)input.size(3),
      (size_t)out.size(2),
      (size_t)out.size(3),
      (size_t)kernel_size[0],
      (size_t)kernel_size[1],
      (size_t)sh,
      (size_t)sw,
      (size_t)ph,
      (size_t)pw,
      (size_t)dh,
      (size_t)dw);
  return {out, indices};
}

} // namespace native
} // namespace riscv
