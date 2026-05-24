/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Runtime dispatcher for riscv_add_f32. Compiled with -march=rv64gc (the
 * baseline) so this TU can run on any RV64GC CPU. RISCV_KERNELS_HAVE_<V>=1
 * macros are defined per build by CMake — variants not selected at
 * configure time are dropped from the table here so they're not linked.
 *
 * Selection is highest-priority-first; the first variant whose required
 * feature bit is set in riscv_features_detect() wins. With only one
 * variant compiled in, the table degenerates to a single entry and the
 * dispatcher is effectively a direct call (the compiler can constant-fold
 * the table lookup).
 */

#include "../../runtime/riscv_features.h"
#include "../../ops/riscv_ops_common.h"

#include <stddef.h>

typedef struct {
  uint64_t required;
  riscv_add_f32_fn fn;
} riscv_add_f32_variant_t;

static const riscv_add_f32_variant_t kVariants[] = {
#ifdef RISCV_KERNELS_HAVE_RVV
    {RISCV_FEATURE_V, riscv_add_f32_rvv},
#endif
#ifdef RISCV_KERNELS_HAVE_SCALAR
    {0, riscv_add_f32_scalar},
#endif
};

static const size_t kNumVariants =
    sizeof(kVariants) / sizeof(kVariants[0]);

static riscv_add_f32_fn g_chosen = NULL;

static riscv_add_f32_fn choose(void) {
  const riscv_features_t* f = riscv_features_detect();
  for (size_t i = 0; i < kNumVariants; ++i) {
    if (kVariants[i].required == 0 ||
        (f->bits & kVariants[i].required) == kVariants[i].required) {
      return kVariants[i].fn;
    }
  }
  return NULL; /* unreachable when SCALAR is in the build */
}

void riscv_add_f32(const float* a, const float* b, float* out, size_t n) {
  riscv_add_f32_fn fn = g_chosen;
  if (fn == NULL) {
    fn = choose();
    g_chosen = fn;
  }
  fn(a, b, out, n);
}
