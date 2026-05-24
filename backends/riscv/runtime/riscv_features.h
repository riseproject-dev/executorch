/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Feature bits set by riscv_features_detect(). One bit per ISA extension the
 * dispatcher cares about. Keep this stable — values are baked into the
 * dispatch tables in kernels/dispatch/. P / VME / IME / AME are intentionally
 * not represented yet; they'll be added back when their kernels land.
 *
 * Bits 0-7  : base vector
 * Bits 8-15 : vector FP sub-extensions
 * Bits 16-23: vector bitmanip / data-independent timing
 */
#define RISCV_FEATURE_V (1ULL << 0) /* RVV 1.0 (V extension) */

#define RISCV_FEATURE_ZVFHMIN (1ULL << 8) /* Vector FP16 min */
#define RISCV_FEATURE_ZVFH (1ULL << 9) /* Vector FP16 */
#define RISCV_FEATURE_ZVFBFMIN (1ULL << 10) /* Vector BF16 converts */
#define RISCV_FEATURE_ZVFBFWMA (1ULL << 11) /* Vector BF16 widening mul-add */

#define RISCV_FEATURE_ZVBB (1ULL << 16) /* Vector basic bitmanip */
#define RISCV_FEATURE_ZVKT (1ULL << 17) /* Vector data-independent timing */

typedef struct {
  uint64_t bits;
  uint32_t vlen_bytes; /* VLEN/8; 0 when V is absent */
  uint32_t elen_bytes; /* ELEN/8; 0 when V is absent */
} riscv_features_t;

/*
 * Detect once and cache. Safe to call from any TU; the returned pointer is
 * valid for the lifetime of the process and the underlying struct is never
 * mutated after the first call.
 *
 * Detection strategy (Linux): __riscv_hwprobe(2) syscall first, then
 * getauxval(AT_HWCAP) fallback. On baremetal: a constant struct populated at
 * compile time from -DEXECUTORCH_RISCV_BAREMETAL_FEATURES / _VLEN.
 */
const riscv_features_t* riscv_features_detect(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
