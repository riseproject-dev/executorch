/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stddef.h>

extern void riscv_convolution_f32_scalar(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    size_t N,
    size_t Cin,
    size_t Hin,
    size_t Win,
    size_t Cout,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t groups);

void riscv_convolution_f32_rvv(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    size_t N,
    size_t Cin,
    size_t Hin,
    size_t Win,
    size_t Cout,
    size_t Hout,
    size_t Wout,
    size_t Kh,
    size_t Kw,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t groups) {
  riscv_convolution_f32_scalar(
      in,
      weight,
      bias,
      out,
      N,
      Cin,
      Hin,
      Win,
      Cout,
      Hout,
      Wout,
      Kh,
      Kw,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups);
}
