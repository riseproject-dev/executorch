/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Shared cache + single-shot init. The actual probe lives in the per-target
 * TU (riscv_features_linux.c or riscv_features_baremetal.c) and is exposed
 * via riscv_features_probe(). The PoC dispatcher is single-threaded, so a
 * plain static flag is sufficient; if multi-threaded dispatch shows up
 * later, swap the flag for __atomic_load_n / call_once.
 */

#include "riscv_features.h"

extern void riscv_features_probe(riscv_features_t* out);

static riscv_features_t g_features;
static int g_initialized = 0;

const riscv_features_t* riscv_features_detect(void) {
  if (!g_initialized) {
    riscv_features_probe(&g_features);
    g_initialized = 1;
  }
  return &g_features;
}
