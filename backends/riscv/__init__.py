# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy re-exports.

The top-level imports are deliberately lazy: the runtime-only pieces of
this backend (kernels, dispatcher, feature detection) don't need torch or
executorch to be importable. Tests that exercise only the runtime dispatch
should be able to collect without an ExecuTorch installation.
"""

_LAZY_IMPORTS = {
    "ConvertToRiscvPass": ("executorch.backends.riscv.passes", "ConvertToRiscvPass"),
    "RiscvPassManager": ("executorch.backends.riscv.passes", "RiscvPassManager"),
    "RiscvPartitioner": (
        "executorch.backends.riscv.partitioner",
        "RiscvPartitioner",
    ),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS)
