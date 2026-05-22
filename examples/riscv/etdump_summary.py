# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render a per-XNNPACK-op summary from an ETDump file.

Reads the per-op microkernel symbol + microsecond time stashed in
`event.raw_delegate_debug_metadatas` by XNNProfiler. With --run-log the
XNNPACK registration log is parsed to list the *available* microkernels
alongside the dispatched ones.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from executorch.devtools import Inspector


# "Convolution (NHWC, F32) IGEMM #3" -> ("Convolution (NHWC, F32) IGEMM", 3)
_SEQ_RE = re.compile(r"^(.*?)\s+#(\d+)$")

# Wrappers around per-op events; kept separate to avoid double-counting children.
FRAMEWORK_EVENTS = frozenset(
    {
        "Method::execute",
        "Method::init",
        "Program::load_method",
        "DELEGATE_CALL",
        "OPERATOR_CALL",
    }
)

_REG_LOG_RE = re.compile(r"Note \(XNNPACK\):.*microkernel '([^']+)'")


def parse_run_log(path: Path):
    syms = set()
    with open(path, errors="ignore") as f:
        for line in f:
            m = _REG_LOG_RE.search(line)
            if m:
                syms.add(m.group(1))
    return sorted(syms)


def _extract_microkernel_record(ev):
    """Unpack the <symbol>\\0<us> blob XNNProfiler wrote. Returns (symbol, us)."""
    raw = getattr(ev, "raw_delegate_debug_metadatas", None) or []
    if not raw:
        return None, None
    item = raw[0]
    if isinstance(item, (bytes, bytearray)):
        try:
            text = bytes(item).decode("utf-8", errors="replace")
        except Exception:
            return None, None
    else:
        text = str(item)
    fields = text.split("\0")
    symbol = fields[0].strip() if fields and fields[0].strip() else None
    us = None
    if len(fields) > 1 and fields[1].strip():
        try:
            us_int = int(fields[1].strip())
            us = us_int if us_int > 0 else None
        except ValueError:
            us = None
    return symbol, us


def aggregate(etdump_path: Path):
    insp = Inspector(etdump_path=str(etdump_path))
    per_op = defaultdict(
        lambda: {"count": 0, "raw": [], "kernels": {}, "ukernel_us_total": 0}
    )
    framework = defaultdict(lambda: {"count": 0, "raw": []})
    for block in insp.event_blocks:
        for ev in block.events:
            m = _SEQ_RE.match(ev.name or "")
            base = m.group(1) if m else (ev.name or "<unnamed>")
            if base in FRAMEWORK_EVENTS:
                framework[base]["count"] += 1
                framework[base]["raw"].extend(
                    ev.perf_data.raw if ev.perf_data else []
                )
            else:
                bucket = per_op[base]
                bucket["count"] += 1
                bucket["raw"].extend(ev.perf_data.raw if ev.perf_data else [])
                symbol, us = _extract_microkernel_record(ev)
                if symbol:
                    bucket["kernels"][symbol] = bucket["kernels"].get(symbol, 0) + (
                        us or 0
                    )
                    if us:
                        bucket["ukernel_us_total"] += us
    return per_op, framework


def render(per_op, framework, etdump_path, registered_kernels):
    def rows_of(d, with_kernels):
        rows = []
        for name, v in d.items():
            raw = v["raw"]
            s = sum(raw)
            row = {
                "op": name,
                "count": v["count"],
                "sum_ms": s,
                "avg_ms": (s / len(raw)) if raw else 0.0,
                "max_ms": max(raw) if raw else 0.0,
            }
            if with_kernels:
                kernels_dict = v.get("kernels", {})
                # Hottest symbol first; ties / zero-timing entries by name.
                ordered = sorted(
                    kernels_dict.items(),
                    key=lambda kv: (-(kv[1] or 0), kv[0]),
                )
                row["kernels"] = [
                    {"name": sym, "ukernel_ms": (us / 1000.0) if us else 0.0}
                    for sym, us in ordered
                ]
                row["ukernel_sum_ms"] = (
                    v.get("ukernel_us_total", 0) / 1000.0
                )
            rows.append(row)
        rows.sort(key=lambda r: r["sum_ms"], reverse=True)
        return rows

    op_rows = rows_of(per_op, with_kernels=True)
    fw_rows = rows_of(framework, with_kernels=False)
    ops_total = sum(r["sum_ms"] for r in op_rows)
    fw_total = sum(r["sum_ms"] for r in fw_rows)

    def fmt_table(label, rows, total):
        print(f"\n[etdump_summary] {label}  total={total:.3f} ms")
        print(
            f"{'%':>5}  {'sum_ms':>10}  {'count':>6}  {'avg_ms':>10}  "
            f"{'max_ms':>10}  {'ukernel_ms':>10}  op"
        )
        for r in rows:
            pct = (r["sum_ms"] / total * 100.0) if total else 0.0
            uk_ms = r.get("ukernel_sum_ms", 0.0)
            print(
                f"{pct:5.1f}  {r['sum_ms']:10.3f}  {r['count']:6d}  "
                f"{r['avg_ms']:10.3f}  {r['max_ms']:10.3f}  {uk_ms:10.3f}  "
                f"{r['op']}"
            )
            for k in r.get("kernels", ()):
                tag = (
                    f"({k['ukernel_ms']:.3f} ms)" if k["ukernel_ms"] else "(n/a)"
                )
                print(f"{'':>61}    {k['name']}  {tag}")

    print(f"[etdump_summary] {etdump_path}")
    fmt_table(f"XNNPACK ops ({len(op_rows)} unique)", op_rows, ops_total)
    fmt_table(f"Framework wrappers ({len(fw_rows)})", fw_rows, fw_total)
    if registered_kernels:
        print(
            f"\n[etdump_summary] Registered XNNPACK microkernels "
            f"({len(registered_kernels)}):"
        )
        for sym in registered_kernels:
            print(f"  {sym}")

    return op_rows, fw_rows, ops_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etdump", type=Path)
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help=(
            "Optional XNNPACK registration log (Note (XNNPACK): ...) used to "
            "report which microkernels were *available* on this build. "
            "Per-op `kernels` rows come from the etdump metadata."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.etdump.exists():
        print(f"[etdump_summary] missing {args.etdump}", file=sys.stderr)
        sys.exit(1)

    registered_kernels = []
    if args.run_log is not None:
        if not args.run_log.exists():
            print(f"[etdump_summary] missing run log {args.run_log}", file=sys.stderr)
            sys.exit(1)
        registered_kernels = parse_run_log(args.run_log)

    per_op, framework = aggregate(args.etdump)
    op_rows, fw_rows, ops_total = render(
        per_op, framework, args.etdump, registered_kernels
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "etdump": str(args.etdump),
                    "run_log": str(args.run_log) if args.run_log else None,
                    "ops_total_ms": ops_total,
                    "registered_kernels": registered_kernels,
                    "ops": op_rows,
                    "framework": fw_rows,
                },
                indent=2,
            )
        )
        print(f"[etdump_summary] wrote {args.json}")


if __name__ == "__main__":
    main()
