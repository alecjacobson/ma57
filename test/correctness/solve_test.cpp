// Phase 4 correctness tests: the real public solve() path (analyzePattern ->
// factorize -> solve), checked against:
//   (a) a known solution: B = A * Xtrue for random Xtrue, then
//       solve(B) should recover Xtrue and the direct residual
//       ||A*X - B|| / (||A||*||X|| + ||B||) should be tiny;
//   (b) multi-RHS: solving [b1|b2|b3] as one call must match three separate
//       single-column solves, exactly (same code path, same arithmetic);
//   (c) for SPD inputs, cross-checked against Eigen::SimplicialLDLT's own
//       solve on the same system;
//   (d) a larger (n=2000+) sparse case, to sanity check this stays fast and
//       doesn't secretly materialize anything dense.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <Eigen/SparseCholesky>

#include <chrono>
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

double residual(const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
  return (Ad * X - B).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
}

void checkSolveIndefinite(int n, int avgNnzPerRow, unsigned seed, int nrhs, OrderingType ordering) {
  Sparse A = symla_test::randomSparseIndefinite(n, avgNnzPerRow, seed);

  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  std::mt19937 rng(seed + 12345u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd Xtrue(n, nrhs);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);

  Eigen::MatrixXd Ad = toDense(A);
  Eigen::MatrixXd B = Ad * Xtrue;

  Eigen::MatrixXd X = solver.solve(B);
  SYMLA_CHECK(X.rows() == n);
  SYMLA_CHECK(X.cols() == nrhs);

  const double recErr = (X - Xtrue).norm() / std::max(1.0, Xtrue.norm());
  SYMLA_CHECK(recErr < 1e-6);

  const double resid = residual(Ad, X, B);
  SYMLA_CHECK(resid < 1e-8);

  // Multi-RHS consistency: solving all columns together must match solving
  // each column separately.
  if (nrhs > 1) {
    for (int j = 0; j < nrhs; ++j) {
      Eigen::MatrixXd bj = B.col(j);
      Eigen::MatrixXd xj = solver.solve(bj);
      SYMLA_CHECK((xj.col(0) - X.col(j)).norm() < 1e-12 * std::max(1.0, X.col(j).norm()));
    }
  }
}

void checkSolveSPD(int n, int avgNnzPerRow, unsigned seed, int nrhs) {
  Sparse A = symla_test::randomSparseSPD(n, avgNnzPerRow, seed);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  std::mt19937 rng(seed + 999u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd B(n, nrhs);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < nrhs; ++j) B(i, j) = dist(rng);

  Eigen::MatrixXd X = solver.solve(B);
  Eigen::MatrixXd Ad = toDense(A);
  SYMLA_CHECK(residual(Ad, X, B) < 1e-8);

  // Cross-check against Eigen's own SPD sparse solver on the identical
  // system.
  Eigen::SimplicialLDLT<Sparse> eig;
  eig.compute(A);
  SYMLA_CHECK(eig.info() == Eigen::Success);
  Eigen::MatrixXd Xeig = eig.solve(B);
  const double diff = (X - Xeig).norm() / std::max(1.0, Xeig.norm());
  SYMLA_CHECK(diff < 1e-6);
}

void checkSolveThrowsBeforeFactorize() {
  SymLDLT<double> solver;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 1);
  bool threw = false;
  try {
    solver.solve(B);
  } catch (const std::logic_error&) {
    threw = true;
  }
  SYMLA_CHECK(threw);
}

}  // namespace

int main() {
  checkSolveThrowsBeforeFactorize();

  for (int n : {20, 100, 500}) {
    for (unsigned trial = 0; trial < 3; ++trial) {
      const unsigned seed = trial * 104729u + static_cast<unsigned>(n);
      checkSolveIndefinite(n, 5, seed, 1, OrderingType::AMD);
      checkSolveIndefinite(n, 5, seed, 5, OrderingType::AMD);
      checkSolveIndefinite(n, 5, seed + 3u, 1, OrderingType::Natural);
      checkSolveSPD(n, 5, seed + 7u, 5);
#ifdef SYMLA_HAVE_METIS
      checkSolveIndefinite(n, 5, seed + 11u, 3, OrderingType::Metis);
#endif
    }
  }

  // Larger sparse case: sanity check on time/memory (stays comfortably
  // under a minute if this is genuinely sparse-structure-driven and not
  // secretly materializing anything dense).
  {
    const int n = 2500;
    const auto t0 = std::chrono::steady_clock::now();
    Sparse A = symla_test::randomSparseIndefinite(n, 6, 4242u);
    SymLDLT<double> solver;
    solver.setOrdering(OrderingType::AMD);
    solver.compute(A);
    SYMLA_CHECK(!solver.isSingular());

    std::mt19937 rng(4242u + 1);
    std::normal_distribution<double> dist(0.0, 1.0);
    const int nrhs = 5;
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd Ad = toDense(A);
    Eigen::MatrixXd B = Ad * Xtrue;

    Eigen::MatrixXd X = solver.solve(B);
    const auto t1 = std::chrono::steady_clock::now();
    const double resid = residual(Ad, X, B);
    SYMLA_CHECK(resid < 1e-8);

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "symla solve_test: n=" << n << " analyze+factorize+solve took " << seconds << "s, residual="
              << resid << "\n";
    SYMLA_CHECK(seconds < 60.0);
  }

  std::cout << "symla solve_test: all checks passed\n";
  return 0;
}
