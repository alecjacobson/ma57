add_executable(symla_smoke_test unit/smoke_test.cpp)
target_link_libraries(symla_smoke_test PRIVATE symla::symla)
add_test(NAME smoke_test COMMAND symla_smoke_test)

add_executable(symla_ordering_test unit/ordering_test.cpp)
target_link_libraries(symla_ordering_test PRIVATE symla::symla)
target_include_directories(symla_ordering_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME ordering_test COMMAND symla_ordering_test)

add_executable(symla_elimination_tree_test unit/elimination_tree_test.cpp)
target_link_libraries(symla_elimination_tree_test PRIVATE symla::symla)
target_include_directories(symla_elimination_tree_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME elimination_tree_test COMMAND symla_elimination_tree_test)

add_executable(symla_symbolic_test unit/symbolic_test.cpp)
target_link_libraries(symla_symbolic_test PRIVATE symla::symla)
target_include_directories(symla_symbolic_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME symbolic_test COMMAND symla_symbolic_test)

# Dense frontal LDL^T kernel test: additionally links against the system
# LAPACK (used purely as a black-box numerical oracle via dsytrf_/dsytrs_
# for cross-checking residuals/inertia; no LAPACK source is used).
find_library(LAPACK_LIBRARY NAMES lapack)
if(NOT LAPACK_LIBRARY)
  message(FATAL_ERROR "system liblapack not found; required by symla_dense_kernel_test as a numerical oracle")
endif()

add_executable(symla_dense_kernel_test unit/dense_kernel_test.cpp)
target_link_libraries(symla_dense_kernel_test PRIVATE symla::symla ${LAPACK_LIBRARY})
target_include_directories(symla_dense_kernel_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME dense_kernel_test COMMAND symla_dense_kernel_test)

# Phase 3: multifrontal numeric factorization.
add_executable(symla_multifrontal_test unit/multifrontal_test.cpp)
target_link_libraries(symla_multifrontal_test PRIVATE symla::symla)
target_include_directories(symla_multifrontal_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME multifrontal_test COMMAND symla_multifrontal_test)

add_executable(symla_multifrontal_correctness_test correctness/multifrontal_correctness_test.cpp)
target_link_libraries(symla_multifrontal_correctness_test PRIVATE symla::symla)
target_include_directories(symla_multifrontal_correctness_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME multifrontal_correctness_test COMMAND symla_multifrontal_correctness_test)

# Phase 4: multifrontal solve (forward/diag/back substitution over the
# per-front factorization).
add_executable(symla_solve_test correctness/solve_test.cpp)
target_link_libraries(symla_solve_test PRIVATE symla::symla)
target_include_directories(symla_solve_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME solve_test COMMAND symla_solve_test)

# --- Phase 5: regression/benchmark suite ---

# Synthetic block-arrow / KKT-shaped correctness tests.
add_executable(symla_kkt_test correctness/kkt_test.cpp)
target_link_libraries(symla_kkt_test PRIVATE symla::symla)
target_include_directories(symla_kkt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME kkt_test COMMAND symla_kkt_test)

# Controlled-conditioning correctness fuzzing.
add_executable(symla_conditioning_fuzz_test correctness/conditioning_fuzz_test.cpp)
target_link_libraries(symla_conditioning_fuzz_test PRIVATE symla::symla)
target_include_directories(symla_conditioning_fuzz_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME conditioning_fuzz_test COMMAND symla_conditioning_fuzz_test)

# Real downloaded matrices (bench/fetch_matrices.py); skips gracefully if
# bench/matrices/ hasn't been populated.
add_executable(symla_real_matrix_test correctness/real_matrix_test.cpp)
target_link_libraries(symla_real_matrix_test PRIVATE symla::symla)
target_include_directories(symla_real_matrix_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME real_matrix_test COMMAND symla_real_matrix_test)

# Stability stress tests.
add_executable(symla_near_singular_test stability/near_singular_test.cpp)
target_link_libraries(symla_near_singular_test PRIVATE symla::symla)
target_include_directories(symla_near_singular_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME near_singular_test COMMAND symla_near_singular_test)

add_executable(symla_zero_diagonal_test stability/zero_diagonal_test.cpp)
target_link_libraries(symla_zero_diagonal_test PRIVATE symla::symla)
target_include_directories(symla_zero_diagonal_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME zero_diagonal_test COMMAND symla_zero_diagonal_test)

add_executable(symla_pivot_threshold_sweep_test stability/pivot_threshold_sweep_test.cpp)
target_link_libraries(symla_pivot_threshold_sweep_test PRIVATE symla::symla)
target_include_directories(symla_pivot_threshold_sweep_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME pivot_threshold_sweep_test COMMAND symla_pivot_threshold_sweep_test)
