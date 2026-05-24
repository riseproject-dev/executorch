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

/* From <asm/hwprobe.h>; the IMA_EXT_0 key returns a bitmask of supported
 * extensions. Bit positions match the upstream kernel headers (Linux 6.5+ for
 * V/Zba/Zbb, 6.7+ for Zvf*, 6.10+ for Zvbb/Zvkt). Unknown bits read as 0 on
 * older kernels — features are *optional* by design. */
#define ET_RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define ET_RISCV_HWPROBE_IMA_V (1ULL << 2)
#define ET_RISCV_HWPROBE_EXT_ZVFHMIN (1ULL << 28)
#define ET_RISCV_HWPROBE_EXT_ZVFH (1ULL << 29)
#define ET_RISCV_HWPROBE_EXT_ZVFBFMIN (1ULL << 38)
#define ET_RISCV_HWPROBE_EXT_ZVFBFWMA (1ULL << 39)
#define ET_RISCV_HWPROBE_EXT_ZVBB (1ULL << 17)
#define ET_RISCV_HWPROBE_EXT_ZVKT (1ULL << 26)

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
  /* Optional V sub-extensions. The kernel masks unknown bits to 0, so probing
   * a missing extension just leaves the feature bit clear; kernels must check
   * before using the corresponding intrinsics. */
  if (ext & ET_RISCV_HWPROBE_EXT_ZVFHMIN)
    out->bits |= RISCV_FEATURE_ZVFHMIN;
  if (ext & ET_RISCV_HWPROBE_EXT_ZVFH)
    out->bits |= RISCV_FEATURE_ZVFH;
  if (ext & ET_RISCV_HWPROBE_EXT_ZVFBFMIN)
    out->bits |= RISCV_FEATURE_ZVFBFMIN;
  if (ext & ET_RISCV_HWPROBE_EXT_ZVFBFWMA)
    out->bits |= RISCV_FEATURE_ZVFBFWMA;
  if (ext & ET_RISCV_HWPROBE_EXT_ZVBB)
    out->bits |= RISCV_FEATURE_ZVBB;
  if (ext & ET_RISCV_HWPROBE_EXT_ZVKT)
    out->bits |= RISCV_FEATURE_ZVKT;
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
