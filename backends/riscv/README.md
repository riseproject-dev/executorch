# RISC-V Backend (PoC)

Same-process CPU backend for RISC-V. Kernels are hand-written in C using
`<riscv_vector.h>` intrinsics for the RVV path and plain C for the scalar
path. Variants are selected at **runtime** from `riscv_features_detect()`,
so a single binary can run on RV64GC baseline hardware and light up RVV
when present.

Enable with `-DEXECUTORCH_BUILD_RISCV=ON`.

## Status

| Variant | `-march=`     | Op `add` (fp32) | Tested CPUs                    |
| ------- | ------------- | --------------- | ------------------------------ |
| scalar  | `rv64gc`      | ✓               | `qemu -cpu rv64`               |
| RVV 1.0 | `rv64gcv`     | ✓               | `qemu -cpu rv64,v=true,vlen=…` |
| P       | `rv64gc`      | stub → scalar   | —                              |
| VME     | `rv64gc`      | stub → scalar   | —                              |
| IME     | `rv64gc`      | stub → scalar   | —                              |
| AME     | `rv64gc`      | stub → scalar   | —                              |

Op coverage in the PoC: `aten::add.Tensor` (fp32, no broadcast, alpha=1).
Any other shape / dtype / alpha falls back to the portable kernel. The
"stub" variants exist so the dispatch table and CMake plumbing stay
complete; they forward to scalar.

## Dispatch design

Two compile-time paths, both with a runtime pick:

* **Path 1 — compile everything in (default).** Every implemented variant
  is built into its own static archive with its own `-march=`. The
  dispatcher (`kernels/dispatch/op_add_dispatch.c`, compiled with the
  baseline `rv64gc`) walks a priority-ordered table on first call and
  caches the chosen function pointer. Highest priority first: AME → IME →
  VME → P → RVV → scalar.

* **Path 2 — compile-time subset.** Set
  `-DEXECUTORCH_RISCV_KERNELS=scalar` (or any semicolon list). Variants
  not on the list aren't built and aren't linked; the dispatcher only
  considers what's present. With a single variant the dispatcher
  effectively collapses to a direct call.

The dispatcher itself never uses extension-specific intrinsics, so the
*single binary* property holds: a Path 1 build runs on any RV64GC CPU and
upgrades automatically when V is detected.

## Feature detection

`runtime/riscv_features.{h,c}` exposes:

```c
const riscv_features_t* riscv_features_detect(void);
```

The probe path depends on `CMAKE_SYSTEM_NAME`:

* **Linux** (`runtime/riscv_features_linux.c`): `__riscv_hwprobe(2)` syscall
  first (kernel ≥ 6.5), then `getauxval(AT_HWCAP)` fallback for older
  kernels. The syscall is invoked directly via `syscall(258, …)` to avoid
  depending on the glibc wrapper (only present from 2.40 onward).
* **Baremetal** (`runtime/riscv_features_baremetal.c`,
  `CMAKE_SYSTEM_NAME=Generic`): returns a compile-time-constant struct
  populated from `-DEXECUTORCH_RISCV_BAREMETAL_FEATURES` and
  `-DEXECUTORCH_RISCV_BAREMETAL_VLEN`, which should mirror the `-march=`
  and `-mrvv-vector-bits=` the kernel TUs were compiled with.

## Building & running

```bash
examples/riscv/setup.sh         # cross toolchain + qemu-user-static
examples/riscv/run.sh --backend=riscv

# Run on a specific CPU config
QEMU_CPU=rv64,v=true,vlen=256 examples/riscv/run.sh --backend=riscv

# Path 2: compile only the scalar variant
examples/riscv/run.sh --backend=riscv --riscv-kernels=scalar
```

`Test_result: PASS` from the bundled-IO comparison path is the pass
criterion.

## Adding an op

1. Define the schema in `ops/operators.yaml` and the `torch.library`
   binding in `ops/operators.py` (use the existing `add` as a template).
2. Write the C++ glue in `ops/op_<name>.cpp` calling the dispatcher entry
   point declared in `ops/riscv_ops_common.h`.
3. Implement at least the scalar variant in `kernels/scalar/op_<name>.c`.
   Add RVV/etc. in their respective directories as you go.
4. Add the dispatcher TU under `kernels/dispatch/` following
   `op_add_dispatch.c` — copy the priority table and adjust the symbols.
5. Extend `ConvertToRiscvPass` to recognize the aten op and rewrite it.

## Adding an ISA variant

1. Create `kernels/<new>/op_<name>.c` and define `riscv_<name>_<new>` for
   every op that has a real implementation, or have it forward to scalar
   if you just want the symbol to exist.
2. Add an entry to `_riscv_variants` in `CMakeLists.txt` and a
   `_march_<new>` setting with the right `-march=` flag.
3. Add a feature bit in `runtime/riscv_features.h` and detect it in
   `riscv_features_linux.c` / set it in `riscv_features_baremetal.c`.
4. Add an `#ifdef RISCV_KERNELS_HAVE_<NEW>` entry in each `op_*_dispatch.c`
   selection table at the right priority level.
