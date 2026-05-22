/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Linux feature probe: __riscv_hwprobe(2) (kernel >= 6.5) with a
 * getauxval(AT_HWCAP) fallback for older kernels. We hand-roll the syscall
 * rather than rely on glibc's wrapper because the wrapper appeared in
 * glibc 2.40 and Ubuntu 24.04 still ships 2.39.
 */

#include "riscv_features.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_riscv_hwprobe
#define __NR_riscv_hwprobe 258
#endif

/* From <asm/hwprobe.h>; redeclared so we don't require recent kernel headers
 * at build time. */
struct et_riscv_hwprobe {
  int64_t key;
  uint64_t value;
};

#define ET_RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define ET_RISCV_HWPROBE_IMA_V (1ULL << 2)
#define ET_RISCV_HWPROBE_EXT_ZVFH (1ULL << 30)

/* AT_HWCAP bit for V; matches Linux's COMPAT_HWCAP_ISA_V = 1 << ('V' - 'A'). */
#define ET_HWCAP_ISA_V (1UL << ('V' - 'A'))

static int try_hwprobe(riscv_features_t* out) {
  struct et_riscv_hwprobe pairs[1] = {{ET_RISCV_HWPROBE_KEY_IMA_EXT_0, 0}};
  long rc = syscall(
      __NR_riscv_hwprobe,
      pairs,
      (size_t)1,
      (size_t)0,
      (void*)NULL,
      (unsigned)0);
  if (rc < 0) {
    return -1;
  }
  uint64_t ext = pairs[0].value;
  if (ext & ET_RISCV_HWPROBE_IMA_V) {
    out->bits |= RISCV_FEATURE_V;
    /* hwprobe doesn't expose VLEN directly in older kernels; assume the
     * RVV 1.0 minimum (128 bits) here. Kernels that benefit from a larger
     * VLEN read vlenb themselves once V is known to be present. */
    out->vlen_bytes = 16;
    out->elen_bytes = 8;
  }
  if (ext & ET_RISCV_HWPROBE_EXT_ZVFH) {
    out->bits |= RISCV_FEATURE_ZVFH;
  }
  return 0;
}

static void try_auxval(riscv_features_t* out) {
  unsigned long hwcap = getauxval(AT_HWCAP);
  if (hwcap & ET_HWCAP_ISA_V) {
    out->bits |= RISCV_FEATURE_V;
    out->vlen_bytes = 16;
    out->elen_bytes = 8;
  }
}

void riscv_features_probe(riscv_features_t* out) {
  memset(out, 0, sizeof(*out));
  if (try_hwprobe(out) == 0) {
    return;
  }
  /* hwprobe missing (ENOSYS on kernel < 6.5) or otherwise failed. */
  try_auxval(out);
}
