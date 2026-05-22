# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placeholder partitioner for the RISC-V kernel-library backend.

The PoC backend is kernel-library style — there is no ``BackendInterface``
delegate yet, so this class exists only to expose ``ConvertToRiscvPass``
through a stable name. When a real delegate-style addition lands, it can
extend or replace this without changing the import surface.
"""

from __future__ import annotations

from executorch.backends.riscv.passes import ConvertToRiscvPass


class RiscvPartitioner:
    passes = [ConvertToRiscvPass]
