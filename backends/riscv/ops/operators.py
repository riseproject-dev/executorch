# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom op definitions for the RISC-V backend.

Defines ``riscv::add`` (functional + out-variant). The functional impl is
forwarded to ``aten::add.Tensor`` so the AOT pass that lowers
``aten.add.Tensor`` to ``riscv.add.Tensor`` doesn't change numerics. The
runtime kernel registered against ``riscv::add.out`` lives in
``ops/op_add.cpp`` and dispatches to the variant chosen by
``riscv_features_detect()``.
"""

import torch
from torch.library import impl, Library, register_fake

lib = Library("riscv", "DEF")

lib.define("add(Tensor self, Tensor other) -> Tensor")
lib.define("add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("riscv::add")
def _add_fake(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(self)


@impl(lib, "add", "CompositeExplicitAutograd")
def _add_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.add(self, other)


@impl(lib, "add.out", "CompositeExplicitAutograd")
def _add_out_impl(
    self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    torch.add(self, other, out=out)
    return out
