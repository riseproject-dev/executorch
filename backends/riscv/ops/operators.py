# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom op definitions for the RISC-V backend.

Each entry registers a ``riscv::<op>`` namespaced version of the matching
``aten::<op>`` so the ConvertToRiscvPass can swap nodes without conflicting
with portable's kernels. The functional impls just call into torch (eager
runtime) so the AOT trace stays numerically identical; the runtime kernels
that actually execute live in ``ops/op_*.cpp`` and dispatch to the
``riscv_features_detect()``-chosen variant.

Adding an op: ``lib.define`` both the functional + out-variant signatures,
``@register_fake`` the functional (for meta tracing), and ``@impl`` both
variants to forward to torch.
"""

from typing import List, Optional

import torch
from torch.library import impl, Library, register_fake

lib = Library("riscv", "DEF")


# --- add -------------------------------------------------------------------

lib.define("add(Tensor self, Tensor other) -> Tensor")
lib.define("add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::add")
def _add_fake(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "add", "CompositeExplicitAutograd")
def _add_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.add(self, other)


@impl(lib, "add.out", "CompositeExplicitAutograd")
def _add_out_impl(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    torch.add(self, other, out=out)
    return out


# --- hardtanh --------------------------------------------------------------

lib.define("hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor")
lib.define(
    "hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::hardtanh")
def _hardtanh_fake(
    self: torch.Tensor, min_val: float = -1, max_val: float = 1
) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "hardtanh", "CompositeExplicitAutograd")
def _hardtanh_impl(
    self: torch.Tensor, min_val: float = -1, max_val: float = 1
) -> torch.Tensor:
    return torch.hardtanh(self, min_val, max_val)


@impl(lib, "hardtanh.out", "CompositeExplicitAutograd")
def _hardtanh_out_impl(
    self: torch.Tensor,
    min_val: float = -1,
    max_val: float = 1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.clamp(self, min=min_val, max=max_val, out=out)
    return out


# --- relu ------------------------------------------------------------------

lib.define("relu(Tensor self) -> Tensor")
lib.define("relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::relu")
def _relu_fake(self: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "relu", "CompositeExplicitAutograd")
def _relu_impl(self: torch.Tensor) -> torch.Tensor:
    return torch.relu(self)


@impl(lib, "relu.out", "CompositeExplicitAutograd")
def _relu_out_impl(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    torch.clamp(self, min=0, out=out)
    return out


# --- mul -------------------------------------------------------------------

lib.define("mul(Tensor self, Tensor other) -> Tensor")
lib.define("mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::mul")
def _mul_fake(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "mul", "CompositeExplicitAutograd")
def _mul_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.mul(self, other)


@impl(lib, "mul.out", "CompositeExplicitAutograd")
def _mul_out_impl(
    self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    torch.mul(self, other, out=out)
    return out


# --- mean.dim --------------------------------------------------------------
# `aten::mean.dim` takes an Optional<List<int>> dim; out variant keeps the
# same signature plus the destination tensor.

lib.define(
    "mean(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"
)
lib.define(
    "mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::mean")
def _mean_fake(
    self: torch.Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.mean(self, dim=dim, keepdim=keepdim, dtype=dtype)


@impl(lib, "mean", "CompositeExplicitAutograd")
def _mean_impl(
    self: torch.Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.mean(self, dim=dim, keepdim=keepdim, dtype=dtype)


@impl(lib, "mean.out", "CompositeExplicitAutograd")
def _mean_out_impl(
    self: torch.Tensor,
    dim: Optional[List[int]],
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.mean(self, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    return out


# --- addmm -----------------------------------------------------------------

lib.define(
    "addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"
)
lib.define(
    "addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::addmm")
def _addmm_fake(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    return torch.empty((mat1.shape[0], mat2.shape[1]), dtype=self.dtype)


@impl(lib, "addmm", "CompositeExplicitAutograd")
def _addmm_impl(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
) -> torch.Tensor:
    return torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha)


@impl(lib, "addmm.out", "CompositeExplicitAutograd")
def _addmm_out_impl(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha, out=out)
    return out


# --- _native_batch_norm_legit_no_training ---------------------------------

lib.define(
    "_native_batch_norm_legit_no_training("
    "Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, "
    "Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)"
)
lib.define(
    "_native_batch_norm_legit_no_training.out("
    "Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, "
    "Tensor running_var, float momentum, float eps, *, "
    "Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!))"
)


@register_fake("riscv::_native_batch_norm_legit_no_training")
def _bn_fake(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
):
    return (
        torch.empty_like(input),
        torch.empty(0, dtype=input.dtype),
        torch.empty(0, dtype=input.dtype),
    )


@impl(lib, "_native_batch_norm_legit_no_training", "CompositeExplicitAutograd")
def _bn_impl(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
):
    return torch._native_batch_norm_legit_no_training(
        input, weight, bias, running_mean, running_var, momentum, eps
    )


@impl(lib, "_native_batch_norm_legit_no_training.out", "CompositeExplicitAutograd")
def _bn_out_impl(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
    *,
    out: torch.Tensor,
    save_mean: torch.Tensor,
    save_invstd: torch.Tensor,
):
    o, sm, si = torch._native_batch_norm_legit_no_training(
        input, weight, bias, running_mean, running_var, momentum, eps
    )
    out.copy_(o)
    return out, save_mean, save_invstd


# --- convolution -----------------------------------------------------------

lib.define(
    "convolution("
    "Tensor input, Tensor weight, Tensor? bias, "
    "int[] stride, int[] padding, int[] dilation, "
    "bool transposed, int[] output_padding, int groups"
    ") -> Tensor"
)
lib.define(
    "convolution.out("
    "Tensor input, Tensor weight, Tensor? bias, "
    "int[] stride, int[] padding, int[] dilation, "
    "bool transposed, int[] output_padding, int groups, *, "
    "Tensor(a!) out"
    ") -> Tensor(a!)"
)


@register_fake("riscv::convolution")
def _conv_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
) -> torch.Tensor:
    return torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


@impl(lib, "convolution", "CompositeExplicitAutograd")
def _conv_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
) -> torch.Tensor:
    return torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


@impl(lib, "convolution.out", "CompositeExplicitAutograd")
def _conv_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    y = torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    out.copy_(y)
    return out


# --- max_pool2d_with_indices ----------------------------------------------

lib.define(
    "max_pool2d_with_indices("
    "Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, "
    "int[2] dilation=1, bool ceil_mode=False"
    ") -> (Tensor, Tensor)"
)
lib.define(
    "max_pool2d_with_indices.out("
    "Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, "
    "int[2] dilation=1, bool ceil_mode=False, *, "
    "Tensor(a!) out, Tensor(b!) indices"
    ") -> (Tensor(a!), Tensor(b!))"
)


@register_fake("riscv::max_pool2d_with_indices")
def _maxpool_fake(
    self: torch.Tensor,
    kernel_size: List[int],
    stride: List[int] = (),
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
):
    return torch.nn.functional.max_pool2d_with_indices(
        self,
        kernel_size,
        stride or kernel_size,
        padding,
        dilation,
        ceil_mode,
    )


@impl(lib, "max_pool2d_with_indices", "CompositeExplicitAutograd")
def _maxpool_impl(
    self: torch.Tensor,
    kernel_size: List[int],
    stride: List[int] = (),
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
):
    return torch.nn.functional.max_pool2d_with_indices(
        self,
        kernel_size,
        stride or kernel_size,
        padding,
        dilation,
        ceil_mode,
    )


@impl(lib, "max_pool2d_with_indices.out", "CompositeExplicitAutograd")
def _maxpool_out_impl(
    self: torch.Tensor,
    kernel_size: List[int],
    stride: List[int] = (),
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
    *,
    out: torch.Tensor,
    indices: torch.Tensor,
):
    o, ix = torch.nn.functional.max_pool2d_with_indices(
        self,
        kernel_size,
        stride or kernel_size,
        padding,
        dilation,
        ceil_mode,
    )
    out.copy_(o)
    indices.copy_(ix)
    return out, indices


# --- view_copy / permute_copy / _clone_dim_order --------------------------

lib.define("view_copy(Tensor self, int[] size) -> Tensor")
lib.define("view_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::view_copy")
def _view_copy_fake(self: torch.Tensor, size: List[int]) -> torch.Tensor:
    return torch.empty(size, dtype=self.dtype)


@impl(lib, "view_copy", "CompositeExplicitAutograd")
def _view_copy_impl(self: torch.Tensor, size: List[int]) -> torch.Tensor:
    return self.reshape(size).clone()


@impl(lib, "view_copy.out", "CompositeExplicitAutograd")
def _view_copy_out_impl(self: torch.Tensor, size: List[int], *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self.reshape(size))
    return out


lib.define("permute_copy(Tensor self, int[] dims) -> Tensor")
lib.define("permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::permute_copy")
def _permute_copy_fake(self: torch.Tensor, dims: List[int]) -> torch.Tensor:
    return torch.empty([self.shape[d] for d in dims], dtype=self.dtype)


@impl(lib, "permute_copy", "CompositeExplicitAutograd")
def _permute_copy_impl(self: torch.Tensor, dims: List[int]) -> torch.Tensor:
    return self.permute(dims).contiguous()


@impl(lib, "permute_copy.out", "CompositeExplicitAutograd")
def _permute_copy_out_impl(self: torch.Tensor, dims: List[int], *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self.permute(dims))
    return out


lib.define(
    "_clone_dim_order(Tensor self, *, bool non_blocking=False, int[]? dim_order=None) -> Tensor"
)
lib.define(
    "_clone_dim_order.out(Tensor self, *, bool non_blocking=False, int[]? dim_order=None, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::_clone_dim_order")
def _clone_fake(self: torch.Tensor, *, non_blocking: bool = False, dim_order: Optional[List[int]] = None) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "_clone_dim_order", "CompositeExplicitAutograd")
def _clone_impl(self: torch.Tensor, *, non_blocking: bool = False, dim_order: Optional[List[int]] = None) -> torch.Tensor:
    return self.clone()


@impl(lib, "_clone_dim_order.out", "CompositeExplicitAutograd")
def _clone_out_impl(
    self: torch.Tensor,
    *,
    non_blocking: bool = False,
    dim_order: Optional[List[int]] = None,
    out: torch.Tensor,
) -> torch.Tensor:
    out.copy_(self)
    return out
