# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrites edge-dialect ops to their ``riscv::*`` counterparts.

Today this only handles ``aten.add.Tensor`` and only for the
no-broadcast / ``alpha == 1`` / float32 case. Anything else stays on the
portable kernel. The pass is intentionally conservative: it's safer for the
PoC to leave a node alone than to rewrite it and discover at runtime that
the kernel doesn't support the layout.
"""

from __future__ import annotations

import executorch.backends.riscv.ops.operators  # noqa: F401 — registers riscv::add

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


_ATEN_ADD_TARGETS = {
    exir_ops.edge.aten.add.Tensor,
    torch.ops.aten.add.Tensor,
}


def _shape(node: torch.fx.Node):
    val = node.meta.get("val")
    if val is None:
        return None
    return tuple(val.shape)


def _dtype(node: torch.fx.Node):
    val = node.meta.get("val")
    if val is None:
        return None
    return val.dtype


class ConvertToRiscvPass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in _ATEN_ADD_TARGETS:
                continue

            if len(node.args) < 2:
                continue
            self_n, other_n = node.args[0], node.args[1]
            if not isinstance(self_n, torch.fx.Node) or not isinstance(
                other_n, torch.fx.Node
            ):
                continue

            alpha = node.kwargs.get("alpha", 1)
            if alpha != 1:
                continue

            s_shape = _shape(self_n)
            o_shape = _shape(other_n)
            if s_shape is None or o_shape is None or s_shape != o_shape:
                continue

            if _dtype(self_n) != torch.float32 or _dtype(other_n) != torch.float32:
                continue

            with graph.inserting_before(node):
                new_node = graph.call_function(
                    torch.ops.riscv.add.default, (self_n, other_n)
                )
            new_node.meta = dict(node.meta)
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
