# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize-side bootstrap for the RISC-V backend.

Imports torch.ao.quantization's quantized_decomposed lib (so the
``quantize_per_tensor`` / ``dequantize_per_tensor`` / ``add`` functional
schemas exist), then re-opens the same namespace with ``Library("FRAGMENT")``
and registers their ``.out`` variants — necessary for executorch's
``ToOutVarPass`` to rewrite them and produce a runnable .pte.

Loads ``executorch.kernels.quantized``'s shared library on the side: the AOT
shape-inference call sites for the new ``.out`` ops resolve against the
quantized_decomposed kernels packaged inside that .so.

Idempotent — re-import is a no-op via the ``_registered`` guard.
"""

from __future__ import annotations

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 — registers functional schemas
import executorch.kernels.quantized  # noqa: F401 — loads libquantized_ops_aot_lib

_registered = False
_lib = None  # keep alive — torch.library.Library is destructor-sensitive: when
             # the Python wrapper is GC'd the schemas it owns are unregistered.


def register_out_variants() -> None:
    """Add .out overloads to torch.library so ToOutVarPass can rewrite them.

    Schemas mirror executorch/kernels/quantized/quantized.yaml byte for byte
    so the C++ runtime kernels (torch::executor::*_out) match the dispatch
    keys exactly.
    """
    global _registered, _lib
    if _registered:
        return
    _lib = torch.library.Library("quantized_decomposed", "FRAGMENT")
    lib = _lib

    # All schemas copied from kernels/quantized/quantized.yaml. Keep in sync.
    schemas = [
        (
            "quantize_per_tensor.out(Tensor input, float scale, int zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "quantize_per_tensor.Tensor_out(Tensor input, Tensor scale, Tensor zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None, "
            "Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "dequantize_per_tensor.Tensor_out(Tensor input, Tensor scale, Tensor zero_point, "
            "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None, "
            "Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "choose_qparams.Tensor_out(Tensor input, int quant_min, int quant_max, "
            "float eps, ScalarType dtype, *, Tensor(a!) scale_out, Tensor(b!) zero_point_out)"
            " -> (Tensor(a!), Tensor(b!))"
        ),
        (
            "choose_qparams_symmetric.Tensor_out(Tensor input, int quant_min, int quant_max, "
            "float eps, ScalarType dtype, *, Tensor(a!) scale_out, Tensor(b!) zero_point_out)"
            " -> (Tensor(a!), Tensor(b!))"
        ),
        (
            "quantize_per_channel.out(Tensor input, Tensor scales, Tensor zero_points, "
            "int axis, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out)"
            " -> Tensor(a!)"
        ),
        (
            "dequantize_per_channel.out(Tensor input, Tensor scales, Tensor? zero_points, "
            "int axis, int quant_min, int quant_max, ScalarType dtype, *, "
            "ScalarType? out_dtype=None, Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "add.out(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, "
            "Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, "
            "float out_scale, int out_zero_point, int out_quant_min, int out_quant_max, *, "
            "Tensor(a!) out) -> Tensor(a!)"
        ),
        (
            "mul.out(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, "
            "Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, "
            "float out_scale, int out_zero_point, int out_quant_min, int out_quant_max, *, "
            "Tensor(a!) out) -> Tensor(a!)"
        ),
    ]

    for schema in schemas:
        try:
            lib.define(schema)
        except RuntimeError as e:
            # Some schemas may already be registered by other modules
            # (e.g. cortex_m backend's bootstrap); torch.library raises on
            # duplicate define which is harmless for our purposes.
            import sys
            print(f"[riscv.quantize] skip define {schema[:50]}...: {e}", file=sys.stderr)
            if "already" not in str(e).lower():
                raise

    # CompositeExplicitAutograd impls: route the .out variant back through the
    # functional op so meta-tracing during ToOutVarPass succeeds. The native
    # C++ kernels in libquantized_kernels.a (linked at runtime) take over for
    # actual execution.
    def _qpt_out(input, scale, zero_point, quant_min, quant_max, dtype, *, out):
        out.copy_(
            torch.ops.quantized_decomposed.quantize_per_tensor.default(
                input, scale, zero_point, quant_min, quant_max, dtype
            )
        )
        return out

    def _dqpt_out(input, scale, zero_point, quant_min, quant_max, dtype, *, out_dtype=None, out):
        out.copy_(
            torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype
            )
        )
        return out

    def _qadd_out(
        a, a_scale, a_zero_point, a_quant_min, a_quant_max,
        b, b_scale, b_zero_point, b_quant_min, b_quant_max,
        out_scale, out_zero_point, out_quant_min, out_quant_max, *, out,
    ):
        out.copy_(
            torch.ops.quantized_decomposed.add.default(
                a, a_scale, a_zero_point, a_quant_min, a_quant_max,
                b, b_scale, b_zero_point, b_quant_min, b_quant_max,
                out_scale, out_zero_point, out_quant_min, out_quant_max,
            )
        )
        return out

    lib.impl("quantize_per_tensor.out", _qpt_out, "CompositeExplicitAutograd")
    lib.impl("dequantize_per_tensor.out", _dqpt_out, "CompositeExplicitAutograd")
    lib.impl("add.out", _qadd_out, "CompositeExplicitAutograd")

    _registered = True


register_out_variants()
