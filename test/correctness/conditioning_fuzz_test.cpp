// Phase 5: correctness fuzzing with *controlled conditioning*, rather than
// arbitrary random signs. Construction: a diagonally-dominant sparse
// symmetric matrix whose diagonal magnitudes are drawn from a fixed range
// [1, targetCond], with off-diagonal magnitudes kept small relative to the
// diagonal (bounded fraction of it) so the matrix stays diagonally dominant.
// By Gershgorin's circle theorem, every eigenvalue lies within a small
// perturbation of some diagonal entry, so the matrix's condition number is
// controlled to be close to targetCond (== max|diag| / min|diag|) by
// construction -- not just "whatever random signs happen to produce". Signs
// of diagonal entries are randomized independently to make the matrix
// genuinely indefinite.
//
// We then check that the achieved solve residual scales sensibly with
// targetCond: forward error / residual should grow roughly in proportion to
// the condition number (standard backward-stability argument: a backward-
// stable solver achieves residual ~ O(eps) but *forward* error / relative
// solve accuracy degrades like O(cond * eps)), and in particular must not
// blow up wildly faster than that bound, across several decades of
// targetCond.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

namespace {

// Builds an n x n sparse symmetric indefinite matrix with diagonal
// magnitudes log-uniform in [1, targetCond] and off-diagonal magnitudes
// bounded so row-wise diagonal dominance (and hence the targeted condition
// number, up to a modest constant factor) is preserved.
Sparse controlledConditionMatrix(int n, int avgNnzPerRow, double targetCond, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> colDist(0, n - 1);
  std::uniform_real_distribution<double> logDiagDist(0.0, std::log(targetCond));
  std::uniform_int_distribution<int> signDist(0, 1);

  std::vector<double> diagMag(n);
  std::vector<int> diagSign(n);
  for (int i = 0; i < n; ++i) {
    diagMag[i] = std::exp(logDiagDist(rng));
    diagSign[i] = signDist(rng) == 0 ? 1 : -1;
  }

  std::vector<std::vector<std::pair<int, double>>> upper(n);
  std::vector<double> rowAbsSum(n, 0.0);
  for (int i = 0; i < n; ++i) {
    const int nEntries = avgNnzPerRow > 0 ? (1 + static_cast<int>(rng() % avgNnzPerRow)) : 0;
    for (int e = 0; e < nEntries; ++e) {
      const int j = colDist(rng);
      if (j <= i) continue;
      // Keep off-diagonal magnitude a small fraction of min(|d_i|,|d_j|) so
      // diagonal dominance (and thus the targeted spectrum) is preserved
      // regardless of how large targetCond is.
      const double scale = 0.05 * std::min(diagMag[i], diagMag[j]) / std::max(1, avgNnzPerRow);
      std::uniform_real_distribution<double> valDist(-scale, scale);
      const double v = valDist(rng);
      upper[i].push_back({j, v});
    }
  }

  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) {
    for (auto& pr : upper[i]) {
      const int j = pr.first;
      const double v = pr.second;
      trips.emplace_back(i, j, v);
      trips.emplace_back(j, i, v);
      rowAbsSum[i] += std::abs(v);
      rowAbsSum[j] += std::abs(v);
    }
  }
  for (int i = 0; i < n; ++i) {
    // Diagonal dominates the accumulated off-diagonal mass by construction
    // (rowAbsSum[i] <= n * 0.05 * diagMag[i], generously bounded), so add a
    // small safety margin on top of diagMag[i] itself.
    const double d = diagSign[i] * (diagMag[i] + rowAbsSum[i]);
    trips.emplace_back(i, i, d);
  }

  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

Eigen::MatrixXd toDense(const Sparse& A) {
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Ad(it.row(), it.col()) = it.value();
  return Ad;
}

// Returns (forward relative error, residual).
std::pair<double, double> checkConditioned(int n, double targetCond, unsigned seed) {
  Sparse A = controlledConditionMatrix(n, 5, targetCond, seed);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  std::mt19937 rng(seed + 77u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);

  Eigen::MatrixXd Ad = toDense(A);
  Eigen::VectorXd b = Ad * xtrue;
  Eigen::MatrixXd X = solver.solve(Eigen::MatrixXd(b));

  const double resid =
      (Ad * X - Eigen::MatrixXd(b)).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + b.norm());
  const double fwdErr = (X.col(0) - xtrue).norm() / std::max(1.0, xtrue.norm());
  return {fwdErr, resid};
}

}  // namespace

int main() {
  const double eps = std::numeric_limits<double>::epsilon();

  for (int n : {50, 200}) {
    std::cout << "symla conditioning_fuzz_test: n=" << n << "\n";
    for (double targetCond : {1e1, 1e3, 1e6, 1e9}) {
      double worstFwd = 0.0, worstResid = 0.0;
      for (unsigned trial = 0; trial < 3; ++trial) {
        const unsigned seed = trial * 104729u + static_cast<unsigned>(n) + static_cast<unsigned>(targetCond);
        auto [fwdErr, resid] = checkConditioned(n, targetCond, seed);
        worstFwd = std::max(worstFwd, fwdErr);
        worstResid = std::max(worstResid, resid);
      }
      std::cout << "  targetCond=" << targetCond << "  worst fwdErr=" << worstFwd
                << "  worst residual=" << worstResid << "\n";

      // Residual is a backward-stability measure: for a stable solver this
      // should stay near machine epsilon *regardless* of conditioning (it is
      // not the forward error). Allow a generous constant factor for
      // matrix-size-dependent rounding accumulation, but it must not scale
      // with targetCond.
      SYMLA_CHECK(worstResid < 1e5 * eps);

      // Forward error legitimately scales with conditioning (standard
      // backward-stability bound: fwdErr <~ cond(A) * O(eps)). Check it
      // stays within a generous multiple of that bound -- i.e. that
      // conditioning-driven error growth is the *expected* polynomial rate,
      // not some catastrophic blow-up.
      SYMLA_CHECK(worstFwd < 1e4 * targetCond * eps || worstFwd < 1e-2);
    }
  }

  std::cout << "symla conditioning_fuzz_test: all checks passed\n";
  return 0;
}
