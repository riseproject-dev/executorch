# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AOT-side tests for the riscv::add lowering pass.

Exercises the FX graph transformation. The end-to-end runtime check lives
in ``examples/riscv/run.sh --backend=riscv``; the QEMU-only smoke test
that doesn't need ExecuTorch / torch installed is in
``test_runtime_dispatch.py``.
"""

from __future__ import annotations

import torch
from executorch.backends.riscv import ConvertToRiscvPass
from executorch.exir import to_edge
from torch.export import export


def _count_calls(graph: torch.fx.Graph, target) -> int:
    return sum(1 for n in graph.nodes if n.op == "call_function" and n.target is target)


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def _lowered_graph():
    model = AddModule().eval()
    example_inputs = (torch.ones(1, 4), torch.full((1, 4), 2.0))
    exported = export(model, example_inputs)
    edge = to_edge(exported)
    return edge.exported_program().graph_module


def test_pass_rewrites_add() -> None:
    gm = _lowered_graph()
    before = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and "aten.add" in str(n.target)
    )
    assert before >= 1

    result = ConvertToRiscvPass()(gm)
    assert result is not None
    after_gm = result.graph_module
    after_aten_add = sum(
        1
        for n in after_gm.graph.nodes
        if n.op == "call_function" and "aten.add" in str(n.target)
    )
    after_riscv_add = _count_calls(after_gm.graph, torch.ops.riscv.add.default)
    assert after_aten_add == 0
    assert after_riscv_add == before


def test_pass_leaves_mismatched_dtype_alone() -> None:
    class IntAdd(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    model = IntAdd().eval()
    example_inputs = (
        torch.ones(1, 4, dtype=torch.int32),
        torch.full((1, 4), 2, dtype=torch.int32),
    )
    exported = export(model, example_inputs)
    edge = to_edge(exported)
    gm = edge.exported_program().graph_module

    before_riscv = _count_calls(gm.graph, torch.ops.riscv.add.default)
    result = ConvertToRiscvPass()(gm)
    after_gm = result.graph_module
    after_riscv = _count_calls(after_gm.graph, torch.ops.riscv.add.default)
    assert after_riscv == before_riscv, "pass should not rewrite non-fp32 add"
