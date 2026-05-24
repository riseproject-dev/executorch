# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# RISC-V kernel-library backend. Builds one static archive per ISA variant,
# each compiled with its own -march= flag, plus a baseline-march dispatcher
# that picks at runtime from riscv_features_detect().
#
# Configure:
#   -DEXECUTORCH_BUILD_RISCV=ON
#   -DEXECUTORCH_RISCV_KERNELS=all      # default; same as scalar;rvv
#   -DEXECUTORCH_RISCV_KERNELS=scalar   # compile-time subset (single variant)
#   -DEXECUTORCH_RISCV_KERNELS=scalar;rvv
# P / VME / IME / AME kernels are not part of this PoC and will be reintroduced
# alongside their implementations.

cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

# Resolve the variant list. "all" expands to every variant the PoC ships.
set(EXECUTORCH_RISCV_KERNELS
    "all"
    CACHE STRING "Semicolon list of variants or 'all'"
)
if(EXECUTORCH_RISCV_KERNELS STREQUAL "all")
  set(_riscv_variants scalar rvv)
else()
  set(_riscv_variants ${EXECUTORCH_RISCV_KERNELS})
endif()

# Per-variant -march=. The scalar TU inherits whatever the cross toolchain set
# (rv64gc on linux, rv64iafd / rv32imafdc on baremetal). The rvv TU needs the
# V extension on top, so we re-spell -march per target. Override via
# -DEXECUTORCH_RISCV_RVV_MARCH=... if the defaults don't match your toolchain.
set(_march_scalar "")
if(NOT EXECUTORCH_RISCV_RVV_MARCH)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32")
    set(EXECUTORCH_RISCV_RVV_MARCH "rv32imafdcv")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(EXECUTORCH_RISCV_RVV_MARCH "rv64gcv")
  else()
    set(EXECUTORCH_RISCV_RVV_MARCH "rv64iafdv")
  endif()
endif()
set(_march_rvv "-march=${EXECUTORCH_RISCV_RVV_MARCH}")

# Feature-detection library. Linux path uses hwprobe + getauxval; baremetal
# returns a compile-time-constant struct. The choice is driven by
# CMAKE_SYSTEM_NAME.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_riscv_features_target_src
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/riscv_features_linux.c
  )
else()
  set(_riscv_features_target_src
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/riscv_features_baremetal.c
  )
endif()

add_library(
  riscv_features STATIC ${CMAKE_CURRENT_SOURCE_DIR}/runtime/riscv_features.c
                        ${_riscv_features_target_src}
)
target_include_directories(
  riscv_features
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/riscv/runtime>
)
target_compile_options(riscv_features PRIVATE ${_march_scalar})
# Forward the CMake cache vars (set by examples/riscv/run.sh for baremetal) to
# the C preprocessor so riscv_features_baremetal.c picks them up.
if(DEFINED EXECUTORCH_RISCV_BAREMETAL_FEATURES)
  target_compile_definitions(
    riscv_features PRIVATE
    EXECUTORCH_RISCV_BAREMETAL_FEATURES=${EXECUTORCH_RISCV_BAREMETAL_FEATURES}ULL
  )
endif()
if(DEFINED EXECUTORCH_RISCV_BAREMETAL_VLEN)
  target_compile_definitions(
    riscv_features PRIVATE
    EXECUTORCH_RISCV_BAREMETAL_VLEN=${EXECUTORCH_RISCV_BAREMETAL_VLEN}U
  )
endif()
set_property(TARGET riscv_features PROPERTY POSITION_INDEPENDENT_CODE ON)

# Per-variant kernel archives. Each TU is compiled with its own -march= so
# that a single binary can hold V intrinsics next to plain rv64gc code; the
# dispatcher (compiled with the baseline) is what decides at runtime which
# one to call.
set(_riscv_kernel_libs)
foreach(_v IN LISTS _riscv_variants)
  set(_lib riscv_kernels_${_v})
  add_library(
    ${_lib} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${_v}/op_add.c
  )
  target_compile_options(${_lib} PRIVATE ${_march_${_v}})
  set_property(TARGET ${_lib} PROPERTY POSITION_INDEPENDENT_CODE ON)
  list(APPEND _riscv_kernel_libs ${_lib})
endforeach()

# Dispatcher. RISCV_KERNELS_HAVE_<V>=1 macros let the dispatcher's
# selection table drop unselected variants at preprocessor time.
add_library(
  riscv_dispatch STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/dispatch/op_add_dispatch.c
)
target_compile_options(riscv_dispatch PRIVATE ${_march_scalar})
set_property(TARGET riscv_dispatch PROPERTY POSITION_INDEPENDENT_CODE ON)
foreach(_v IN LISTS _riscv_variants)
  string(TOUPPER ${_v} _vU)
  target_compile_definitions(
    riscv_dispatch PRIVATE RISCV_KERNELS_HAVE_${_vU}=1
  )
endforeach()
target_link_libraries(
  riscv_dispatch PUBLIC riscv_features ${_riscv_kernel_libs}
)

# C++ glue + codegen wiring. gen_selected_ops + generate_bindings_for_kernels
# turn operators.yaml into the static op-registration table that the runtime
# pulls in via the riscv_ops_lib target.
set(_yaml_file ${CMAKE_CURRENT_SOURCE_DIR}/ops/operators.yaml)
gen_selected_ops(LIB_NAME "riscv_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")
generate_bindings_for_kernels(
  LIB_NAME "riscv_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)

add_library(riscv_kernels ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_add.cpp)
target_include_directories(
  riscv_kernels PRIVATE ${EXECUTORCH_ROOT}/..
                        ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
                        ${CMAKE_CURRENT_SOURCE_DIR}/ops
                        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)
target_link_libraries(
  riscv_kernels PRIVATE executorch kernels_util_all_deps riscv_dispatch
)
target_compile_options(riscv_kernels PRIVATE ${_march_scalar})

gen_operators_lib(
  LIB_NAME "riscv_ops_lib" KERNEL_LIBS riscv_kernels DEPS executorch
)

install(
  TARGETS riscv_kernels riscv_ops_lib riscv_dispatch riscv_features
          ${_riscv_kernel_libs}
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/riscv/ops/
)
