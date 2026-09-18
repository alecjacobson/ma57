# Phase 5: performance benchmark harness. Times symla::SymLDLT against
# Eigen::SimplicialLDLT / Eigen::SparseLU baselines, plus SuiteSparse
# CHOLMOD / UMFPACK when available (guarded by the same SYMLA_HAVE_CHOLMOD /
# SYMLA_HAVE_UMFPACK plumbing the top-level CMakeLists.txt already sets up).
add_executable(symla_bench_main bench_main.cpp)
target_link_libraries(symla_bench_main PRIVATE symla::symla)

if(SYMLA_HAVE_CHOLMOD)
  target_link_libraries(symla_bench_main PRIVATE ${CHOLMOD_LIBRARY})
  if(CAMD_LIBRARY)
    target_link_libraries(symla_bench_main PRIVATE ${CAMD_LIBRARY})
  endif()
  if(CCOLAMD_LIBRARY)
    target_link_libraries(symla_bench_main PRIVATE ${CCOLAMD_LIBRARY})
  endif()
  if(COLAMD_LIBRARY)
    target_link_libraries(symla_bench_main PRIVATE ${COLAMD_LIBRARY})
  endif()
  target_compile_definitions(symla_bench_main PRIVATE SYMLA_HAVE_CHOLMOD=1)
endif()

if(SYMLA_HAVE_UMFPACK)
  target_link_libraries(symla_bench_main PRIVATE ${UMFPACK_LIBRARY})
  target_compile_definitions(symla_bench_main PRIVATE SYMLA_HAVE_UMFPACK=1)
endif()

# Ordering diagnostic: AMD vs METIS structural/timing comparison on a single
# Matrix Market file. See bench/ordering_diag_main.cpp for the real-world
# regression this was written to diagnose.
add_executable(symla_ordering_diag ordering_diag_main.cpp)
target_link_libraries(symla_ordering_diag PRIVATE symla::symla)


# Serial-vs-parallel factorize() timing diagnostic: used to tune the
# task-DAG scheduling-granularity fix in include/symla/multifrontal.hpp
# (subtreeWeight cost estimate, task_cutoff, adaptive thread-count capping).
add_executable(symla_parallel_diag parallel_diag_main.cpp)
target_link_libraries(symla_parallel_diag PRIVATE symla::symla)

# Serial-vs-parallel solve() timing diagnostic: used to calibrate the
# task-DAG scheduling cost model in include/symla/solve.hpp (separately
# from factorize()'s, since solve()'s per-front cost is much smaller).
add_executable(symla_solve_parallel_diag solve_parallel_diag_main.cpp)
target_link_libraries(symla_solve_parallel_diag PRIVATE symla::symla)

# Throwaway before/after timing driver for the Tier-1 performance fixes
# (compact block-diagonal D storage, -march=native, cached masked top block
# in solve()): analyzePattern()+factorize()+solve() end-to-end timing.
add_executable(symla_tier1_timing tier1_timing_main.cpp)
target_link_libraries(symla_tier1_timing PRIVATE symla::symla)
