# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Edge-dialect -> riscv:: rewrite pass.

Each entry in ``_REWRITES`` declares one rewrite: the edge-dialect op(s)
that match, the destination ``riscv::`` op, and an optional predicate that
gates the rewrite (e.g. dtype, broadcast, transposed=true). When the
predicate fails the original aten node is left alone and runs on the
portable kernel — the riscv backend is additive, not exclusive.

The scalar kernels under ``backends/riscv/kernels/scalar/`` are float32-only
and assume contiguous NCHW for the conv / batch-norm / pool family; the
predicates here filter out anything outside that envelope.
"""

from __future__ import annotations

import executorch.backends.riscv.ops.operators  # noqa: F401 — registers riscv::*

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def _val(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def _shape(node):
    v = _val(node)
    return tuple(v.shape) if v is not None else None


def _dtype(node):
    v = _val(node)
    return v.dtype if v is not None else None


def _is_fp32(*ts) -> bool:
    return all(t is not None and t == torch.float32 for t in ts)


# --- per-op predicates ----------------------------------------------------


def _match_add_or_mul(node: torch.fx.Node) -> bool:
    """Element-wise binary: identical shape + fp32 + alpha=1."""
    if len(node.args) < 2:
        return False
    a, b = node.args[0], node.args[1]
    if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)):
        return False
    if node.kwargs.get("alpha", 1) != 1:
        return False
    sa, sb = _shape(a), _shape(b)
    if sa is None or sb is None or sa != sb:
        return False
    return _is_fp32(_dtype(a), _dtype(b))


def _match_unary_fp32(node: torch.fx.Node) -> bool:
    if len(node.args) < 1 or not isinstance(node.args[0], torch.fx.Node):
        return False
    return _is_fp32(_dtype(node.args[0]))


def _match_conv(node: torch.fx.Node) -> bool:
    # aten.convolution.default args:
    # input, weight, bias, stride, padding, dilation, transposed, output_pad, groups
    if len(node.args) < 9:
        return False
    inp, w = node.args[0], node.args[1]
    transposed = node.args[6]
    if transposed:
        return False
    if not (isinstance(inp, torch.fx.Node) and isinstance(w, torch.fx.Node)):
        return False
    if not _is_fp32(_dtype(inp), _dtype(w)):
        return False
    s_inp = _shape(inp)
    s_w = _shape(w)
    return s_inp is not None and s_w is not None and len(s_inp) == 4 and len(s_w) == 4


def _match_batch_norm(node: torch.fx.Node) -> bool:
    if len(node.args) < 7:
        return False
    inp = node.args[0]
    return (
        isinstance(inp, torch.fx.Node)
        and _is_fp32(_dtype(inp))
        and _shape(inp) is not None
        and len(_shape(inp)) == 4
    )


def _match_max_pool2d(node: torch.fx.Node) -> bool:
    if len(node.args) < 2:
        return False
    inp = node.args[0]
    return isinstance(inp, torch.fx.Node) and _is_fp32(_dtype(inp))


def _match_addmm(node: torch.fx.Node) -> bool:
    if len(node.args) < 3:
        return False
    a, b, c = node.args[0], node.args[1], node.args[2]
    if not all(isinstance(x, torch.fx.Node) for x in (a, b, c)):
        return False
    return _is_fp32(_dtype(a), _dtype(b), _dtype(c))


def _match_mean_dim(node: torch.fx.Node) -> bool:
    """Only handle the contiguous trailing-dim reduction the vision models
    emit (`(B, C, H, W) -> (B, C)` with keepdim either way)."""
    if len(node.args) < 2:
        return False
    inp = node.args[0]
    dims = node.args[1]
    if not isinstance(inp, torch.fx.Node):
        return False
    if not _is_fp32(_dtype(inp)):
        return False
    if not isinstance(dims, (list, tuple)) or not dims:
        return False
    s = _shape(inp)
    if s is None:
        return False
    ndim = len(s)
    norm = sorted([(d % ndim) for d in dims])
    return norm == list(range(ndim - len(norm), ndim))


def _match_passthrough(node: torch.fx.Node) -> bool:
    """view_copy / permute_copy / _clone_dim_order accept any dtype, no
    shape constraints — they only move bytes."""
    if len(node.args) < 1 or not isinstance(node.args[0], torch.fx.Node):
        return False
    return _dtype(node.args[0]) is not None


def _match_bmm(node: torch.fx.Node) -> bool:
    if len(node.args) < 2:
        return False
    a, b = node.args[0], node.args[1]
    return (
        isinstance(a, torch.fx.Node)
        and isinstance(b, torch.fx.Node)
        and _is_fp32(_dtype(a), _dtype(b))
    )


def _match_softmax(node: torch.fx.Node) -> bool:
    if len(node.args) < 2:
        return False
    inp, dim = node.args[0], node.args[1]
    if not isinstance(inp, torch.fx.Node) or not _is_fp32(_dtype(inp)):
        return False
    s = _shape(inp)
    if s is None:
        return False
    d = int(dim)
    if d < 0:
        d += len(s)
    return d == len(s) - 1


def _match_mul_scalar(node: torch.fx.Node) -> bool:
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return False
    return _is_fp32(_dtype(node.args[0]))


def _match_where(node: torch.fx.Node) -> bool:
    if len(node.args) < 3:
        return False
    cond, a, b = node.args[0], node.args[1], node.args[2]
    if not all(isinstance(x, torch.fx.Node) for x in (cond, a, b)):
        return False
    sc, sa, sb = _shape(cond), _shape(a), _shape(b)
    # No-broadcast contract: kernel assumes all three share the output shape.
    return sc is not None and sa is not None and sb is not None and sa == sb and sa == sc


def _match_logical_not(node: torch.fx.Node) -> bool:
    return len(node.args) >= 1 and isinstance(node.args[0], torch.fx.Node)


def _match_cmp_scalar(node: torch.fx.Node) -> bool:
    return len(node.args) >= 2 and isinstance(node.args[0], torch.fx.Node)


def _match_any_dim(node: torch.fx.Node) -> bool:
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return False
    return True  # bool-tensor result; dtype check skipped


def _match_cat(node: torch.fx.Node) -> bool:
    tensors = node.args[0] if node.args else None
    if not isinstance(tensors, (list, tuple)) or not tensors:
        return False
    return all(isinstance(t, torch.fx.Node) for t in tensors)


def _match_embedding(node: torch.fx.Node) -> bool:
    if len(node.args) < 2:
        return False
    w, ix = node.args[0], node.args[1]
    return (
        isinstance(w, torch.fx.Node)
        and isinstance(ix, torch.fx.Node)
        and _is_fp32(_dtype(w))
    )


def _match_pad(node: torch.fx.Node) -> bool:
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return False
    return _is_fp32(_dtype(node.args[0]))


def _match_anything(node: torch.fx.Node) -> bool:
    """No-op predicate for tensor-creation ops (full, scalar_tensor, arange)
    that don't need an input dtype check."""
    return True


# --- rewrite table --------------------------------------------------------


def _edge_or_aten(name: str):
    """Pull both the edge-dialect and aten-dialect overloads of `name`.
    Edge ops live under exir_ops.edge.aten.<op>.<overload>; the raw aten
    overload is what pre-edge transform passes see."""
    edge_obj = exir_ops.edge.aten
    aten_obj = torch.ops.aten
    for part in name.split("."):
        edge_obj = getattr(edge_obj, part)
        aten_obj = getattr(aten_obj, part)
    return {edge_obj, aten_obj}


_REWRITES = [
    (_edge_or_aten("add.Tensor"), torch.ops.riscv.add.default, _match_add_or_mul),
    (_edge_or_aten("mul.Tensor"), torch.ops.riscv.mul.default, _match_add_or_mul),
    (_edge_or_aten("hardtanh.default"), torch.ops.riscv.hardtanh.default, _match_unary_fp32),
    (_edge_or_aten("relu.default"), torch.ops.riscv.relu.default, _match_unary_fp32),
    (_edge_or_aten("mean.dim"), torch.ops.riscv.mean.default, _match_mean_dim),
    (_edge_or_aten("addmm.default"), torch.ops.riscv.addmm.default, _match_addmm),
    (
        _edge_or_aten("_native_batch_norm_legit_no_training.default"),
        torch.ops.riscv._native_batch_norm_legit_no_training.default,
        _match_batch_norm,
    ),
    (_edge_or_aten("convolution.default"), torch.ops.riscv.convolution.default, _match_conv),
    (
        _edge_or_aten("max_pool2d_with_indices.default"),
        torch.ops.riscv.max_pool2d_with_indices.default,
        _match_max_pool2d,
    ),
    (_edge_or_aten("view_copy.default"), torch.ops.riscv.view_copy.default, _match_passthrough),
    (_edge_or_aten("permute_copy.default"), torch.ops.riscv.permute_copy.default, _match_passthrough),
    # expand / unsqueeze / slice copy: the riscv:: fake Python impls don't
    # always reproduce aten's SymInt shape arithmetic exactly (mobilebert hits
    # a mismatch where the cat upstream sees a size-8 vs size-9 tensor).
    # Keep on portable until the riscv:: fakes match.
    # cat: skipped — fake-shape inference picks the original input sizes and
    # asserts dim-mismatch even though the actual runtime would pad-then-cat.
    # Stays on portable until the riscv:: fake reproduces aten's relaxed
    # broadcasting.
    # dim_order_ops live in a different namespace (executorch-specific), no aten counterpart.
    (
        {exir_ops.edge.dim_order_ops._clone_dim_order.default},
        torch.ops.riscv._clone_dim_order.default,
        _match_passthrough,
    ),
    # _to_dim_order_copy has extra dtype/layout/device kwargs the riscv::
    # registration doesn't carry — left on portable until those kwargs are
    # propagated to a riscv equivalent.
    (_edge_or_aten("bmm.default"), torch.ops.riscv.bmm.default, _match_bmm),
    (_edge_or_aten("mm.default"), torch.ops.riscv.mm.default, _match_bmm),
    (_edge_or_aten("sub.Tensor"), torch.ops.riscv.sub.default, _match_add_or_mul),
    (_edge_or_aten("sigmoid.default"), torch.ops.riscv.sigmoid.default, _match_unary_fp32),
    (_edge_or_aten("rsqrt.default"), torch.ops.riscv.rsqrt.default, _match_unary_fp32),
    (_edge_or_aten("_softmax.default"), torch.ops.riscv._softmax.default, _match_softmax),
    (_edge_or_aten("mul.Scalar"), torch.ops.riscv.mul_Scalar.default, _match_mul_scalar),
    (_edge_or_aten("where.self"), torch.ops.riscv.where_self.default, _match_where),
    (_edge_or_aten("logical_not.default"), torch.ops.riscv.logical_not.default, _match_logical_not),
    (_edge_or_aten("eq.Scalar"), torch.ops.riscv.eq_Scalar.default, _match_cmp_scalar),
    (_edge_or_aten("ge.Scalar"), torch.ops.riscv.ge_Scalar.default, _match_cmp_scalar),
    # full / full_like / scalar_tensor / arange take dtype/layout/device/pin_memory
    # kwargs that the riscv:: registrations don't carry. Leave them on portable
    # for now — they're 1-3 nodes per model and contribute nothing to runtime.
    # constant_pad_nd / embedding: kept on portable until the riscv:: fakes
    # match aten shape-inference exactly (their Python impls go through
    # torch.nn.functional which mishandles a few edge cases in mobilebert).
    (_edge_or_aten("any.dim"), torch.ops.riscv.any_dim.default, _match_any_dim),
]


class ConvertToRiscvPass(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            for sources, dest, predicate in _REWRITES:
                if node.target not in sources:
                    continue
                if not predicate(node):
                    break
                with graph.inserting_before(node):
                    new_node = graph.call_function(dest, node.args, node.kwargs)
                new_node.meta = dict(node.meta)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
                modified = True
                break

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
