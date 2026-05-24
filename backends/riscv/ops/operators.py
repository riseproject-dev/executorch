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
    # Use reshape so -1 in `size` gets inferred from numel (matches aten).
    return self.reshape(size)


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


# --- mm / sub / sigmoid / rsqrt -------------------------------------------

lib.define("mm(Tensor self, Tensor mat2) -> Tensor")
lib.define("mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::mm")
def _mm_fake(self, mat2):
    return torch.empty((self.shape[0], mat2.shape[1]), dtype=self.dtype)


@impl(lib, "mm", "CompositeExplicitAutograd")
def _mm_impl(self, mat2):
    return torch.mm(self, mat2)


@impl(lib, "mm.out", "CompositeExplicitAutograd")
def _mm_out_impl(self, mat2, *, out):
    torch.mm(self, mat2, out=out)
    return out


lib.define("sub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
lib.define("sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::sub")
def _sub_fake(self, other, *, alpha=1):
    return torch.empty_like(self)


@impl(lib, "sub", "CompositeExplicitAutograd")
def _sub_impl(self, other, *, alpha=1):
    return torch.sub(self, other, alpha=alpha)


@impl(lib, "sub.out", "CompositeExplicitAutograd")
def _sub_out_impl(self, other, *, alpha=1, out):
    torch.sub(self, other, alpha=alpha, out=out)
    return out


lib.define("sigmoid(Tensor self) -> Tensor")
lib.define("sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::sigmoid")
def _sigmoid_fake(self):
    return torch.empty_like(self)


@impl(lib, "sigmoid", "CompositeExplicitAutograd")
def _sigmoid_impl(self):
    return torch.sigmoid(self)


@impl(lib, "sigmoid.out", "CompositeExplicitAutograd")
def _sigmoid_out_impl(self, *, out):
    torch.sigmoid(self, out=out)
    return out


lib.define("rsqrt(Tensor self) -> Tensor")
lib.define("rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::rsqrt")
def _rsqrt_fake(self):
    return torch.empty_like(self)


@impl(lib, "rsqrt", "CompositeExplicitAutograd")
def _rsqrt_impl(self):
    return torch.rsqrt(self)


@impl(lib, "rsqrt.out", "CompositeExplicitAutograd")
def _rsqrt_out_impl(self, *, out):
    torch.rsqrt(self, out=out)
    return out


# --- bmm -------------------------------------------------------------------

lib.define("bmm(Tensor self, Tensor mat2) -> Tensor")
lib.define("bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::bmm")
def _bmm_fake(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (self.shape[0], self.shape[1], mat2.shape[2]), dtype=self.dtype
    )


@impl(lib, "bmm", "CompositeExplicitAutograd")
def _bmm_impl(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return torch.bmm(self, mat2)


@impl(lib, "bmm.out", "CompositeExplicitAutograd")
def _bmm_out_impl(self: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    torch.bmm(self, mat2, out=out)
    return out


# --- _softmax --------------------------------------------------------------

lib.define("_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
lib.define("_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::_softmax")
def _softmax_fake(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "_softmax", "CompositeExplicitAutograd")
def _softmax_impl(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    return torch._softmax(self, dim, half_to_float)


@impl(lib, "_softmax.out", "CompositeExplicitAutograd")
def _softmax_out_impl(self: torch.Tensor, dim: int, half_to_float: bool, *, out: torch.Tensor) -> torch.Tensor:
    y = torch._softmax(self, dim, half_to_float)
    out.copy_(y)
    return out


# --- mul.Scalar ------------------------------------------------------------

lib.define("mul_Scalar(Tensor self, Scalar other) -> Tensor")
lib.define("mul_Scalar.out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::mul_Scalar")
def _mul_scalar_fake(self: torch.Tensor, other) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "mul_Scalar", "CompositeExplicitAutograd")
def _mul_scalar_impl(self: torch.Tensor, other) -> torch.Tensor:
    return torch.mul(self, other)


@impl(lib, "mul_Scalar.out", "CompositeExplicitAutograd")
def _mul_scalar_out_impl(self: torch.Tensor, other, *, out: torch.Tensor) -> torch.Tensor:
    torch.mul(self, other, out=out)
    return out


# --- where -----------------------------------------------------------------

lib.define("where_self(Tensor condition, Tensor self, Tensor other) -> Tensor")
lib.define("where_self.out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::where_self")
def _where_fake(condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "where_self", "CompositeExplicitAutograd")
def _where_impl(condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, self, other)


@impl(lib, "where_self.out", "CompositeExplicitAutograd")
def _where_out_impl(
    condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    y = torch.where(condition, self, other)
    out.copy_(y)
    return out


# --- logical_not -----------------------------------------------------------

lib.define("logical_not(Tensor self) -> Tensor")
lib.define("logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::logical_not")
def _logical_not_fake(self: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self, dtype=torch.bool)


@impl(lib, "logical_not", "CompositeExplicitAutograd")
def _logical_not_impl(self: torch.Tensor) -> torch.Tensor:
    return torch.logical_not(self)


@impl(lib, "logical_not.out", "CompositeExplicitAutograd")
def _logical_not_out_impl(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    torch.logical_not(self, out=out)
    return out


# --- eq.Scalar / ge.Scalar -------------------------------------------------

for cmp_name, torch_fn in (("eq_Scalar", torch.eq), ("ge_Scalar", torch.ge)):
    lib.define(f"{cmp_name}(Tensor self, Scalar other) -> Tensor")
    lib.define(f"{cmp_name}.out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")


def _make_cmp_impls(cmp_name: str, torch_fn):
    @register_fake(f"riscv::{cmp_name}")
    def _cmp_fake(self: torch.Tensor, other) -> torch.Tensor:
        return torch.empty_like(self, dtype=torch.bool)

    @impl(lib, cmp_name, "CompositeExplicitAutograd")
    def _cmp_impl(self: torch.Tensor, other) -> torch.Tensor:
        return torch_fn(self, other)

    @impl(lib, f"{cmp_name}.out", "CompositeExplicitAutograd")
    def _cmp_out_impl(self: torch.Tensor, other, *, out: torch.Tensor) -> torch.Tensor:
        torch_fn(self, other, out=out)
        return out

    return _cmp_fake, _cmp_impl, _cmp_out_impl


_make_cmp_impls("eq_Scalar", torch.eq)
_make_cmp_impls("ge_Scalar", torch.ge)


# --- full / full_like / scalar_tensor / arange ----------------------------

lib.define("full(SymInt[] size, Scalar fill_value) -> Tensor")
lib.define("full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::full")
def _full_fake(size: List[int], fill_value) -> torch.Tensor:
    return torch.empty(size, dtype=torch.float32)


@impl(lib, "full", "CompositeExplicitAutograd")
def _full_impl(size: List[int], fill_value) -> torch.Tensor:
    return torch.full(size, fill_value)


@impl(lib, "full.out", "CompositeExplicitAutograd")
def _full_out_impl(size: List[int], fill_value, *, out: torch.Tensor) -> torch.Tensor:
    out.fill_(fill_value)
    return out


lib.define("full_like(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None) -> Tensor")
lib.define(
    "full_like.out(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::full_like")
def _full_like_fake(self: torch.Tensor, fill_value, *, memory_format=None) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "full_like", "CompositeExplicitAutograd")
def _full_like_impl(self: torch.Tensor, fill_value, *, memory_format=None) -> torch.Tensor:
    return torch.full_like(self, fill_value)


@impl(lib, "full_like.out", "CompositeExplicitAutograd")
def _full_like_out_impl(self: torch.Tensor, fill_value, *, memory_format=None, out: torch.Tensor) -> torch.Tensor:
    out.fill_(fill_value)
    return out


lib.define("scalar_tensor(Scalar s) -> Tensor")
lib.define("scalar_tensor.out(Scalar s, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::scalar_tensor")
def _scalar_tensor_fake(s) -> torch.Tensor:
    return torch.empty((), dtype=torch.float32)


@impl(lib, "scalar_tensor", "CompositeExplicitAutograd")
def _scalar_tensor_impl(s) -> torch.Tensor:
    return torch.scalar_tensor(s)


@impl(lib, "scalar_tensor.out", "CompositeExplicitAutograd")
def _scalar_tensor_out_impl(s, *, out: torch.Tensor) -> torch.Tensor:
    out.fill_(s)
    return out


lib.define("arange_start_step(Scalar start, Scalar end, Scalar step=1) -> Tensor")
lib.define(
    "arange_start_step.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::arange_start_step")
def _arange_fake(start, end, step=1) -> torch.Tensor:
    return torch.arange(start, end, step)


@impl(lib, "arange_start_step", "CompositeExplicitAutograd")
def _arange_impl(start, end, step=1) -> torch.Tensor:
    return torch.arange(start, end, step)


@impl(lib, "arange_start_step.out", "CompositeExplicitAutograd")
def _arange_out_impl(start, end, step=1, *, out: torch.Tensor) -> torch.Tensor:
    torch.arange(start, end, step, out=out)
    return out


# --- constant_pad_nd -------------------------------------------------------

lib.define("constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor")
lib.define(
    "constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::constant_pad_nd")
def _pad_fake(self: torch.Tensor, pad: List[int], value=0) -> torch.Tensor:
    return torch.nn.functional.pad(self, list(pad), value=value)


@impl(lib, "constant_pad_nd", "CompositeExplicitAutograd")
def _pad_impl(self: torch.Tensor, pad: List[int], value=0) -> torch.Tensor:
    return torch.nn.functional.pad(self, list(pad), value=value)


@impl(lib, "constant_pad_nd.out", "CompositeExplicitAutograd")
def _pad_out_impl(self: torch.Tensor, pad: List[int], value=0, *, out: torch.Tensor) -> torch.Tensor:
    y = torch.nn.functional.pad(self, list(pad), value=value)
    out.copy_(y)
    return out


# --- embedding -------------------------------------------------------------

lib.define(
    "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor"
)
lib.define(
    "embedding.out(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::embedding")
def _embedding_fake(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    return torch.empty(tuple(indices.shape) + (weight.shape[1],), dtype=weight.dtype)


@impl(lib, "embedding", "CompositeExplicitAutograd")
def _embedding_impl(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    return torch.nn.functional.embedding(indices, weight)


@impl(lib, "embedding.out", "CompositeExplicitAutograd")
def _embedding_out_impl(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    y = torch.nn.functional.embedding(indices, weight)
    out.copy_(y)
    return out


# --- any.dim ---------------------------------------------------------------

lib.define("any_dim(Tensor self, int dim, bool keepdim=False) -> Tensor")
lib.define("any_dim.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::any_dim")
def _any_fake(self: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.any(self, dim=dim, keepdim=keepdim)


@impl(lib, "any_dim", "CompositeExplicitAutograd")
def _any_impl(self: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.any(self, dim=dim, keepdim=keepdim)


@impl(lib, "any_dim.out", "CompositeExplicitAutograd")
def _any_out_impl(self: torch.Tensor, dim: int, keepdim: bool = False, *, out: torch.Tensor) -> torch.Tensor:
    torch.any(self, dim=dim, keepdim=keepdim, out=out)
    return out


# --- expand_copy / unsqueeze_copy / slice_copy / cat ----------------------

lib.define("expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor")
lib.define(
    "expand_copy.out(Tensor self, SymInt[] size, *, bool implicit=False, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::expand_copy")
def _expand_copy_fake(self: torch.Tensor, size: List[int], *, implicit: bool = False) -> torch.Tensor:
    return torch.empty(size, dtype=self.dtype)


@impl(lib, "expand_copy", "CompositeExplicitAutograd")
def _expand_copy_impl(self: torch.Tensor, size: List[int], *, implicit: bool = False) -> torch.Tensor:
    return self.expand(size).contiguous()


@impl(lib, "expand_copy.out", "CompositeExplicitAutograd")
def _expand_copy_out_impl(self: torch.Tensor, size: List[int], *, implicit: bool = False, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self.expand(size))
    return out


lib.define("unsqueeze_copy(Tensor self, int dim) -> Tensor")
lib.define("unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::unsqueeze_copy")
def _unsqueeze_copy_fake(self: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.empty(self.unsqueeze(dim).shape, dtype=self.dtype)


@impl(lib, "unsqueeze_copy", "CompositeExplicitAutograd")
def _unsqueeze_copy_impl(self: torch.Tensor, dim: int) -> torch.Tensor:
    return self.unsqueeze(dim).clone()


@impl(lib, "unsqueeze_copy.out", "CompositeExplicitAutograd")
def _unsqueeze_copy_out_impl(self: torch.Tensor, dim: int, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self.unsqueeze(dim))
    return out


lib.define(
    "slice_copy_Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"
)
lib.define(
    "slice_copy_Tensor.out(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("riscv::slice_copy_Tensor")
def _slice_copy_fake(
    self: torch.Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> torch.Tensor:
    return torch.empty(self.shape, dtype=self.dtype)  # exact shape recomputed by exir


@impl(lib, "slice_copy_Tensor", "CompositeExplicitAutograd")
def _slice_copy_impl(
    self: torch.Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> torch.Tensor:
    return self.narrow(dim, start or 0, (end or self.shape[dim]) - (start or 0)).clone()


@impl(lib, "slice_copy_Tensor.out", "CompositeExplicitAutograd")
def _slice_copy_out_impl(
    self: torch.Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    out.copy_(torch.slice_copy(self, dim, start, end, step))
    return out


lib.define("cat(Tensor[] tensors, int dim=0) -> Tensor")
lib.define("cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::cat")
def _cat_fake(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


@impl(lib, "cat", "CompositeExplicitAutograd")
def _cat_impl(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


@impl(lib, "cat.out", "CompositeExplicitAutograd")
def _cat_out_impl(tensors: List[torch.Tensor], dim: int = 0, *, out: torch.Tensor) -> torch.Tensor:
    torch.cat(tensors, dim=dim, out=out)
    return out
