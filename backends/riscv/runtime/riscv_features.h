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
 * Feature bits set by riscv_features_detect(). One bit per ISA extension that
 * the dispatcher cares about. Keep this stable; the values are baked into the
 * dispatch tables in kernels/dispatch/.
 */
#define RISCV_FEATURE_V (1ULL << 0) /* RVV 1.0 (V extension) */
#define RISCV_FEATURE_P (1ULL << 1) /* Packed SIMD (P) */
#define RISCV_FEATURE_VME (1ULL << 2) /* Vector Matrix Extension */
#define RISCV_FEATURE_IME (1ULL << 3) /* Integer Matrix Extension */
#define RISCV_FEATURE_AME (1ULL << 4) /* Advanced Matrix Extension */
#define RISCV_FEATURE_ZVFH (1ULL << 5) /* Half-precision vector FP */

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
