# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper grouping the RISC-V AOT passes.

There's only one pass in the PoC (``ConvertToRiscvPass``); this exists so
new passes can be added later without callers having to track each one.
"""

from __future__ import annotations

from executorch.exir.pass_manager import PassManager

from .convert_to_riscv_pass import ConvertToRiscvPass


class RiscvPassManager(PassManager):
    pass_list = [ConvertToRiscvPass]

    def __init__(self) -> None:
        super().__init__(passes=[cls() for cls in self.pass_list])
