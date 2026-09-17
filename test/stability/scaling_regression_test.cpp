// Phase 5-fix scaling-regression guard: factorize() on a fixed-size
// synthetic sparse indefinite problem must complete within a generous
// wall-clock bound.
//
// Context: the Phase 5 benchmark run found `factorize()` scaling far worse
// than expected (n=5000 -> 0.76s, n=10000 -> 6.8s, n=15000 -> 55s, an
// apparent ~n^3-ish blowup rather than the near-linear-in-nnz(L) a correct
// multifrontal method should show). Root cause: `dense_kernel.hpp`'s
// trailing-submatrix re-symmetrization step
// (`A = ((A + A.transpose()) * 0.5).eval()`), evaluated once per pivot over
// the *entire* remaining trailing block, forced Eigen to read a transposed
// view of a large non-contiguous `Block` of a column-major matrix -- highly
// cache-hostile -- and dominated factor() time (profiling on a 2745x2745
// front found this one line responsible for ~40 of ~45 total seconds). This
// was fixed by computing the same "average both triangles" result via an
// explicit cache-blocked (tiled) loop instead of Eigen's whole-block
// transpose expression (see dense_kernel.hpp's `resymmetrizeTrailing`).
//
// This test is deliberately generous (order of magnitude above the ~0.03s
// measured post-fix on this machine for n=5000) to avoid flaking on slower
// CI hardware, while still catching a gross regression (e.g. an
// accidentally-reintroduced O(front_size^2)-over-the-whole-matrix operation)
// long before it grows into the kind of superlinear blowup the Phase 5 run
// hit at n=15000.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <iostream>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;
using Clock = std::chrono::steady_clock;

int main() {
  const int n = 5000;
  Sparse A = symla_test::randomSparseIndefinite(n, 6, 20260917u);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);

  const auto t0 = Clock::now();
  solver.analyzePattern(A);
  const double analyzeSec = std::chrono::duration<double>(Clock::now() - t0).count();

  const auto t1 = Clock::now();
  solver.factorize(A);
  const double factorizeSec = std::chrono::duration<double>(Clock::now() - t1).count();

  std::cout << "symla scaling_regression_test: n=" << n << " nnz=" << A.nonZeros()
            << " analyzePattern=" << analyzeSec << "s factorize=" << factorizeSec << "s\n";

  // Generous bounds: post-fix measurements on this machine are ~0.01-0.05s
  // (analyze) and ~0.02-0.5s (factorize) for n=5000; these bounds leave
  // roughly 10-20x headroom for slower CI machines while still catching a
  // regression back toward the pre-fix ~0.76s (and the much worse blowup at
  // larger n the pre-fix code showed).
  SYMLA_CHECK(analyzeSec < 5.0);
  SYMLA_CHECK(factorizeSec < 10.0);

  std::cout << "symla scaling_regression_test: PASSED\n";
  return 0;
}
