// Phase 6: repeated-run determinism as the primary defense against data
// races in the task-DAG parallel factorize(). A race on shared state (e.g.
// a scratch buffer stomped by two concurrently-running fronts, or a
// non-atomic accumulation) would typically manifest as nondeterministic
// results across repeated runs of the *same* parallel solve at a *fixed*
// thread count, even though each individual run might still happen to look
// "plausible" -- this is a stronger and more targeted check than the
// serial-vs-parallel comparison in thread_equivalence_test.cpp, since here
// even the reference is itself a parallel run (i.e. any run-to-run
// divergence at all, for any reason, is suspicious).
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

namespace {

void checkDeterministic(const std::string& label, const Sparse& A, int numThreads, int nRuns) {
  const int n = static_cast<int>(A.rows());
  std::mt19937 rng(777u ^ static_cast<unsigned>(n) ^ static_cast<unsigned>(numThreads));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Eigen::MatrixXd Xtrue(n, 3);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 3; ++j) Xtrue(i, j) = dist(rng);
  Eigen::MatrixXd B = A.selfadjointView<Eigen::Lower>() * Xtrue;

  Eigen::MatrixXd reference;
  symla::Inertia refInertia;
  for (int run = 0; run < nRuns; ++run) {
    SymLDLT<double> solver;
    solver.setOrdering(OrderingType::AMD);
    solver.setParallel(true);
    solver.setNumThreads(numThreads);
    solver.compute(A);
    SYMLA_CHECK(!solver.isSingular());
    Eigen::MatrixXd X = solver.solve(B);
    const auto& inertia = solver.inertia();

    if (run == 0) {
      reference = X;
      refInertia = inertia;
    } else {
      SYMLA_CHECK(inertia.n_pos == refInertia.n_pos);
      SYMLA_CHECK(inertia.n_neg == refInertia.n_neg);
      SYMLA_CHECK(inertia.n_zero == refInertia.n_zero);
      const double diff = (X - reference).norm();
      const double scale = std::max(1.0, reference.norm());
      SYMLA_CHECK(diff / scale < 1e-12);
    }
  }
  std::cout << "symla repeated_run_determinism_test: " << label << " threads=" << numThreads << " (" << nRuns
            << " runs) -- all identical\n";
}

}  // namespace

int main() {
  Sparse A500 = symla_test::randomSparseIndefinite(500, 6, 99u);
  Sparse A2000 = symla_test::randomSparseIndefinite(2000, 8, 100u);

  checkDeterministic("synthetic n=500", A500, 16, 10);
  checkDeterministic("synthetic n=500", A500, 64, 10);
  checkDeterministic("synthetic n=2000", A2000, 32, 10);

  std::cout << "symla repeated_run_determinism_test: PASSED\n";
  return 0;
}
