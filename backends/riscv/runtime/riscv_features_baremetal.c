/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Baremetal feature probe. With no OS there's nothing to query, so the
 * feature set is fixed at build time from the CMake variables
 * EXECUTORCH_RISCV_BAREMETAL_FEATURES (bitmask) and
 * EXECUTORCH_RISCV_BAREMETAL_VLEN (bytes). These mirror the -march= and
 * -mrvv-vector-bits= options the kernel TUs were compiled with.
 */

#include "riscv_features.h"

#include <string.h>

#ifndef EXECUTORCH_RISCV_BAREMETAL_FEATURES
#define EXECUTORCH_RISCV_BAREMETAL_FEATURES 0ULL
#endif

#ifndef EXECUTORCH_RISCV_BAREMETAL_VLEN
#define EXECUTORCH_RISCV_BAREMETAL_VLEN 0U
#endif

void riscv_features_probe(riscv_features_t* out) {
  memset(out, 0, sizeof(*out));
  out->bits = (uint64_t)EXECUTORCH_RISCV_BAREMETAL_FEATURES;
  if (out->bits & RISCV_FEATURE_V) {
    out->vlen_bytes = (uint32_t)EXECUTORCH_RISCV_BAREMETAL_VLEN;
    out->elen_bytes = 8;
  }
}
