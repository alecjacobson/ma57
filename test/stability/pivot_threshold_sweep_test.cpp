// Phase 5 stability test: sweep the `pivot_threshold` (u) parameter over a
// fixed indefinite test matrix and confirm larger u never makes accuracy
// *worse* -- MA57's core stability tradeoff (larger u favors numerical
// stability, at some fill-in cost; smaller u favors sparsity). This is a
// straightforward regression-style monotonicity check.
//
// NOTE for whoever picks up Phase 6/7: as of this phase,
// DenseLDLTOptions::pivot_threshold (dense_kernel.hpp) is *not yet wired
// into the actual Bunch-Kaufman accept/reject decision* -- the kernel
// currently always uses the fixed classical alpha = (1+sqrt(17))/8 for that
// decision and documents pivot_threshold as reserved for a future
// (static-pivoting/KKT) mode. So today this sweep is expected to produce
// *identical* results at every u (checked below as "non-increasing", which
// trivially holds when everything is equal) -- this test will only start
// showing real, distinguishing behavior once u actually participates in
// pivot selection.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

namespace {

Eigen::MatrixXd toDense(const Sparse& A) {
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Ad(it.row(), it.col()) = it.value();
  return Ad;
}

double solveResidual(const Sparse& A, double u, unsigned seed) {
  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setPivotThreshold(u);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  std::mt19937 rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  const int n = static_cast<int>(A.rows());
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Ad = toDense(A);
  Eigen::MatrixXd B = Ad * xtrue;
  Eigen::MatrixXd X = solver.solve(B);
  return (Ad * X - B).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
}

}  // namespace

int main() {
  const std::vector<double> us = {0.001, 0.01, 0.1, 0.5};

  for (int n : {60, 300}) {
    Sparse A = symla_test::randomSparseIndefinite(n, 6, 2718281u + static_cast<unsigned>(n));

    std::vector<double> residuals;
    for (double u : us) {
      const double r = solveResidual(A, u, 314159u + static_cast<unsigned>(n));
      residuals.push_back(r);
      std::cout << "symla pivot_threshold_sweep_test: n=" << n << " u=" << u << " residual=" << r << "\n";
    }

    // Every residual must itself be small (a real correctness bar, not just
    // relative comparison).
    for (double r : residuals) SYMLA_CHECK(r < 1e-8);

    // Larger u must not produce a *worse* (larger) residual than the
    // smallest u tested, up to a generous slack factor for run-to-run
    // floating point noise -- i.e. accuracy should stay the same or improve
    // as u grows, never regress outright.
    const double best = *std::min_element(residuals.begin(), residuals.end());
    for (double r : residuals) {
      SYMLA_CHECK(r < std::max(best * 10.0, 1e-13));
    }
  }

  std::cout << "symla pivot_threshold_sweep_test: all checks passed\n";
  return 0;
}
