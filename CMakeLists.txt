cmake_minimum_required(VERSION 3.18)
project(symla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SYMLA_WITH_SUITESPARSE_BENCH "Link SuiteSparse (CHOLMOD/UMFPACK) for benchmark comparisons" ON)
option(SYMLA_WITH_METIS "Enable METIS nested-dissection ordering" ON)
option(SYMLA_WITH_OPENMP "Enable OpenMP task-DAG parallelism" ON)
option(SYMLA_BUILD_TESTS "Build test suite" ON)
option(SYMLA_BUILD_BENCH "Build benchmark harness" ON)
# `symla` is header-only (an INTERFACE target), so every dense-kernel
# operation is compiled by whatever consumer includes it -- by default that
# means no AVX2/AVX-512/FMA at all, since plain `Release` doesn't imply any
# architecture-specific vectorization target. ON by default because this
# project is developed/benchmarked on a fixed dev machine, and (guarded by a
# compiler-support check below) propagates to *every* consumer of the
# `symla` target -- this repo's own tests/bench, and any downstream project
# that `add_subdirectory`s this repo and doesn't set its own arch flags --
# via `target_compile_options(symla INTERFACE ...)`. A downstream project
# building portable/distributable binaries (i.e. not guaranteed to run on
# this exact build machine's CPU) should set `SYMLA_WITH_ARCH_NATIVE=OFF` and
# choose its own target-architecture flags instead: `-march=native` binaries
# can crash with an illegal-instruction fault when run on a different CPU.
option(SYMLA_WITH_ARCH_NATIVE "Compile symla (and its INTERFACE consumers) with -march=native, if supported" ON)
# MKL or OpenBLAS backing for Eigen's own GEMM-shaped expressions
# (EIGEN_USE_BLAS). Default OFF, unlike the auto-detected-if-found pattern
# used for METIS/OpenMP/CHOLMOD above: empirically (full `ctest` run on this
# machine, MKL's libmkl_rt found and linked) turning this on does not just
# nudge residuals at the margins the way `-march=native`'s FMA contraction
# does above -- it destabilizes numerics across the board (10 of 21 tests
# failed, including previously-comfortable cases like
# `stability/zero_diagonal_test` and `stability/near_singular_test`, not
# just the already-marginal `real_matrix_test` case). Left here, opt-in only
# and clearly flagged, as plumbing for a future investigation (e.g. MKL's
# BLAS may use a different pivoting-irrelevant summation order in ways that
# interact badly with this solver's *own* pivot search rather than being a
# pure GEMM backend swap -- worth a closer look before ever defaulting this
# on), not as a currently-recommended performance win.
option(SYMLA_WITH_BLAS "Route Eigen's dense BLAS-shaped ops through a vendor BLAS if one is found (EXPERIMENTAL: destabilizes results on this codebase as of this writing, see comment above -- default OFF, opt-in only)" OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(PkgConfig QUIET)

include(CheckCXXCompilerFlag)

# --- SuiteSparse (AMD is a required ordering backend; CHOLMOD/UMFPACK optional for bench) ---
find_path(SUITESPARSE_INCLUDE_DIR amd.h PATH_SUFFIXES suitesparse)
find_library(AMD_LIBRARY NAMES amd)
find_library(SUITESPARSECONFIG_LIBRARY NAMES suitesparseconfig)
if(NOT SUITESPARSE_INCLUDE_DIR OR NOT AMD_LIBRARY)
  message(FATAL_ERROR "SuiteSparse AMD headers/library not found; required as an ordering backend")
endif()

set(SYMLA_HAVE_CHOLMOD OFF)
set(SYMLA_HAVE_UMFPACK OFF)
if(SYMLA_WITH_SUITESPARSE_BENCH)
  find_library(CHOLMOD_LIBRARY NAMES cholmod)
  find_library(UMFPACK_LIBRARY NAMES umfpack)
  find_library(CAMD_LIBRARY NAMES camd)
  find_library(CCOLAMD_LIBRARY NAMES ccolamd)
  find_library(COLAMD_LIBRARY NAMES colamd)
  if(CHOLMOD_LIBRARY)
    set(SYMLA_HAVE_CHOLMOD ON)
  endif()
  if(UMFPACK_LIBRARY)
    set(SYMLA_HAVE_UMFPACK ON)
  endif()
endif()

set(SYMLA_HAVE_METIS OFF)
if(SYMLA_WITH_METIS)
  find_path(METIS_INCLUDE_DIR metis.h)
  find_library(METIS_LIBRARY NAMES metis)
  if(METIS_INCLUDE_DIR AND METIS_LIBRARY)
    set(SYMLA_HAVE_METIS ON)
  endif()
endif()

set(SYMLA_HAVE_OPENMP OFF)
if(SYMLA_WITH_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(SYMLA_HAVE_OPENMP ON)
  endif()
endif()

set(SYMLA_HAVE_ARCH_NATIVE OFF)
set(SYMLA_ARCH_NATIVE_FLAGS "")
if(SYMLA_WITH_ARCH_NATIVE)
  check_cxx_compiler_flag("-march=native" SYMLA_COMPILER_SUPPORTS_MARCH_NATIVE)
  if(SYMLA_COMPILER_SUPPORTS_MARCH_NATIVE)
    set(SYMLA_HAVE_ARCH_NATIVE ON)
    set(SYMLA_ARCH_NATIVE_FLAGS -march=native)
    # `-march=native` on an AVX2/AVX-512 host implies `-mfma`, which lets the
    # compiler silently contract `a*b+c`-shaped expressions (extremely
    # common in this codebase's rank-1/rank-2 trailing-update formulas, e.g.
    # `A -= l * d * l^T`) into a single fused-multiply-add with a different,
    # more-accurate-but-*different* rounding than separate multiply+add.
    # That's normally a harmless (even beneficial) accuracy change, but
    # empirically (see `test/correctness/real_matrix_test.cpp`'s
    # `GHS_indef/sit100` case, already documented there as a known
    # borderline Bunch-Kaufman scale-invariance case) it can tip an
    # already-marginal threshold-pivoting decision on a genuinely
    # ill-conditioned real matrix far enough to blow the test's residual
    # bound (observed: residual 0.05-threshold test failing at ~0.15-0.29
    # with FMA contraction on, vs. passing -- differently, as a flagged
    # "singular" case handled explicitly -- without it). `-ffp-contract=off`
    # disables only this silent cross-statement contraction (scalar and
    # vectorized arithmetic still use the full instruction set, i.e. this is
    # not a vectorization-width downgrade), which was confirmed empirically
    # to restore a fully green `ctest` run while keeping `-march=native`'s
    # SIMD-width benefit. Applied whenever supported, unconditionally (not
    # itself a separate option): there is no known upside to leaving FMA
    # contraction on that offsets this risk for a direct symmetric
    # indefinite solver, where pivot *decisions* (not just final residuals)
    # depend on intermediate rounding.
    check_cxx_compiler_flag("-ffp-contract=off" SYMLA_COMPILER_SUPPORTS_FP_CONTRACT_OFF)
    if(SYMLA_COMPILER_SUPPORTS_FP_CONTRACT_OFF)
      list(APPEND SYMLA_ARCH_NATIVE_FLAGS -ffp-contract=off)
    endif()
  endif()
endif()

# --- Optional vendor BLAS backing for Eigen (EIGEN_USE_BLAS) ---
set(SYMLA_HAVE_BLAS OFF)
if(SYMLA_WITH_BLAS)
  # Prefer MKL's single-dylib interface if present, else fall back to any
  # generic BLAS (e.g. OpenBLAS) found under the usual library search path.
  find_library(SYMLA_MKL_RT_LIBRARY NAMES mkl_rt)
  find_library(SYMLA_OPENBLAS_LIBRARY NAMES openblas)
  if(SYMLA_MKL_RT_LIBRARY)
    set(SYMLA_BLAS_LIBRARY ${SYMLA_MKL_RT_LIBRARY})
    set(SYMLA_HAVE_BLAS ON)
  elseif(SYMLA_OPENBLAS_LIBRARY)
    set(SYMLA_BLAS_LIBRARY ${SYMLA_OPENBLAS_LIBRARY})
    set(SYMLA_HAVE_BLAS ON)
  endif()
endif()

add_library(symla INTERFACE)
add_library(symla::symla ALIAS symla)
target_include_directories(symla INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SUITESPARSE_INCLUDE_DIR}>
)
target_link_libraries(symla INTERFACE Eigen3::Eigen ${AMD_LIBRARY})
if(SUITESPARSECONFIG_LIBRARY)
  target_link_libraries(symla INTERFACE ${SUITESPARSECONFIG_LIBRARY})
endif()
target_compile_definitions(symla INTERFACE SYMLA_HAVE_AMD=1)

if(SYMLA_HAVE_METIS)
  target_include_directories(symla INTERFACE ${METIS_INCLUDE_DIR})
  target_link_libraries(symla INTERFACE ${METIS_LIBRARY})
  target_compile_definitions(symla INTERFACE SYMLA_HAVE_METIS=1)
endif()

if(SYMLA_HAVE_OPENMP)
  target_link_libraries(symla INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(symla INTERFACE SYMLA_HAVE_OPENMP=1)
endif()

if(SYMLA_HAVE_ARCH_NATIVE)
  target_compile_options(symla INTERFACE ${SYMLA_ARCH_NATIVE_FLAGS})
endif()

if(SYMLA_HAVE_BLAS)
  target_compile_definitions(symla INTERFACE EIGEN_USE_BLAS=1)
  target_link_libraries(symla INTERFACE ${SYMLA_BLAS_LIBRARY})
endif()

message(STATUS "symla: METIS ordering:        ${SYMLA_HAVE_METIS}")
message(STATUS "symla: OpenMP parallelism:    ${SYMLA_HAVE_OPENMP}")
message(STATUS "symla: CHOLMOD bench compare: ${SYMLA_HAVE_CHOLMOD}")
message(STATUS "symla: UMFPACK bench compare: ${SYMLA_HAVE_UMFPACK}")
message(STATUS "symla: -march=native:         ${SYMLA_HAVE_ARCH_NATIVE}")
message(STATUS "symla: vendor BLAS (Eigen):   ${SYMLA_HAVE_BLAS}")

if(SYMLA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(SYMLA_BUILD_BENCH)
  add_subdirectory(bench)
endif()
