/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>
#include <stdint.h>

extern void riscv_max_pool2d_f32_scalar(
    const float* in,
    float* out,
    int64_t* indices,
    size_t N,
    size_t C,
    size_t Hin,
    size_t Win,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w);

void riscv_max_pool2d_f32_rvv(
    const float* in,
    float* out,
    int64_t* indices,
    size_t N,
    size_t C,
    size_t Hin,
    size_t Win,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w) {
  riscv_max_pool2d_f32_scalar(
      in,
      out,
      indices,
      N,
      C,
      Hin,
      Win,
      Hout,
      Wout,
      Kh,
      Kw,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w);
}
