// Phase 5 stability test: near-singular / ill-conditioned matrices (a
// deliberately small eigenvalue gap -- one diagonal block engineered to be
// close to numerically singular relative to the rest of the matrix). Checks
// that the solver doesn't produce silent garbage: either the residual stays
// bounded appropriately (given the achievable conditioning), or
// isSingular()/delayed-pivot machinery kicks in sensibly (and if it does,
// we must not get NaN/Inf anywhere).
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <random>

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

bool allFinite(const Eigen::MatrixXd& M) { return M.allFinite(); }

// Builds a diagonally-dominant sparse indefinite matrix (as in
// symla_test::randomSparseIndefinite) but then rescales one diagonal entry
// down to `gap` (e.g. 1e-8, 1e-12, ...) relative to the rest, deliberately
// creating a near-singular direction while keeping the matrix's structural
// nonsingularity intact (the entry is small but nonzero).
Sparse withTinyDiagonal(int n, int avgNnzPerRow, unsigned seed, double gap, int tinyRow) {
  Sparse A = symla_test::randomSparseIndefinite(n, avgNnzPerRow, seed);
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Sparse::InnerIterator it(A, k); it; ++it) {
      if (static_cast<int>(it.row()) == tinyRow && static_cast<int>(it.col()) == tinyRow) {
        it.valueRef() = (it.value() > 0 ? 1.0 : -1.0) * gap;
      }
    }
  }
  return A;
}

void checkNearSingular(int n, double gap, unsigned seed) {
  Sparse A = withTinyDiagonal(n, 5, seed, gap, n / 2);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());

  const auto& inertia = solver.inertia();
  SYMLA_CHECK(inertia.n_pos + inertia.n_neg + inertia.n_zero == n);

  if (solver.isSingular()) {
    // Acceptable outcome: the delayed-pivot machinery correctly identified
    // this as numerically singular rather than silently producing garbage.
    // solve() must then refuse (throw), not return anything.
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, 1);
    bool threw = false;
    try {
      solver.solve(B);
    } catch (const std::logic_error&) {
      threw = true;
    }
    SYMLA_CHECK(threw);
    return;
  }

  // Otherwise: not flagged singular, so the solve path must produce
  // finite, sane output for a known-solution problem, with a residual that
  // is at worst proportional to the (large but finite) conditioning induced
  // by `gap` -- generous, but must never be NaN/Inf or wildly divergent.
  std::mt19937 rng(seed + 321u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Ad = toDense(A);
  Eigen::MatrixXd B = Ad * xtrue;

  Eigen::MatrixXd X = solver.solve(B);
  SYMLA_CHECK(allFinite(X));
  const double resid = (Ad * X - B).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
  SYMLA_CHECK(std::isfinite(resid));
  // Residual (backward error) should stay near machine epsilon regardless
  // of conditioning for a backward-stable factorization; allow a generous
  // multiple to account for the deliberately adversarial construction.
  SYMLA_CHECK(resid < 1e-4);
}

}  // namespace

int main() {
  for (int n : {20, 50, 200}) {
    for (double gap : {1e-4, 1e-8, 1e-12, 1e-16}) {
      for (unsigned trial = 0; trial < 2; ++trial) {
        const unsigned seed = trial * 7919u + static_cast<unsigned>(n) + static_cast<unsigned>(gap * 1e6);
        checkNearSingular(n, gap, seed);
      }
    }
  }

  // Exactly zero on one diagonal entry that also has no off-diagonal mass in
  // its row/col isn't constructed here (that's genuinely structurally
  // singular unless a 2x2 pivot bails it out -- covered by
  // zero_diagonal_test.cpp), but gap=0 exercises the boundary: a truly zero
  // diagonal with off-diagonal support forcing 2x2 pivoting.
  {
    const int n = 40;
    Sparse A = withTinyDiagonal(n, 5, 999u, 0.0, n / 2);
    SymLDLT<double> solver;
    solver.compute(A);
    SYMLA_CHECK(solver.factorized());
    SYMLA_CHECK(solver.inertia().n_pos + solver.inertia().n_neg + solver.inertia().n_zero == n);
  }

  std::cout << "symla near_singular_test: all checks passed\n";
  return 0;
}
