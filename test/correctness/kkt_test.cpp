// Phase 5: synthetic block-arrow / KKT-shaped correctness tests.
//
// Constructs the canonical saddle-point / SQD (symmetric quasi-definite)
// system
//
//     K = [ -E   A^T ]
//         [  A    F  ]
//
// with E (n1 x n1) SPD, F (n2 x n2) SPD, and A (n2 x n1) a random sparse
// rectangular block. This is exactly the structure Phase 7's KKT/static-
// pivoting mode will target (Vanderbei's SQD theory / Gill-Saunders-Shinnerl
// existence guarantees), so getting symla's ordinary threshold-pivoted path
// to handle it correctly now is useful groundwork.
//
// Expected inertia, worked out by block elimination: eliminate the -E block
// first (it is invertible and negative definite, contributing inertia
// (0, n1, 0)). The Schur complement is
//     S = F - A (-E)^{-1} A^T = F + A E^{-1} A^T,
// which is SPD whenever F is SPD (F is already PD, and A E^{-1} A^T is PSD,
// so the sum is PD) -- contributing inertia (n2, 0, 0). By Sylvester's law
// of inertia (basis/pivot-order independent), the *total* inertia of K must
// be (n_pos = n2, n_neg = n1, n_zero = 0), regardless of what order symla's
// threshold pivoting actually processes rows/columns in. That is the
// concrete, worked-out assertion below -- not just "no crash".
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

namespace {

// Random sparse SPD block of size n (independent RNG stream from `seed`),
// stored lower-triangle-authoritative (row >= col), matching symla's input
// convention.
Sparse spdBlock(int n, int avgNnzPerRow, unsigned seed) {
  return symla_test::randomSparseSPD(n, avgNnzPerRow, seed);
}

// Random sparse n2 x n1 rectangular block, moderate density, modest
// magnitudes (kept smaller than the diagonal blocks so the KKT system isn't
// absurdly ill-conditioned).
Eigen::SparseMatrix<double, Eigen::ColMajor, int> randomRect(int n2, int n1, int avgNnzPerRow, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> colDist(0, n1 - 1);
  std::uniform_real_distribution<double> valDist(-0.5, 0.5);
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n2; ++i) {
    const int nEntries = avgNnzPerRow > 0 ? (1 + static_cast<int>(rng() % avgNnzPerRow)) : 0;
    for (int e = 0; e < nEntries; ++e) {
      const int j = colDist(rng);
      trips.emplace_back(i, j, valDist(rng));
    }
  }
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n2, n1);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// Builds the full (lower-triangle-authoritative) KKT matrix K, n = n1 + n2.
Sparse buildKKT(const Sparse& E, const Sparse& F, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A) {
  const int n1 = static_cast<int>(E.rows());
  const int n2 = static_cast<int>(F.rows());
  const int n = n1 + n2;
  std::vector<Eigen::Triplet<double>> trips;

  // Top-left: -E, only lower triangle (row >= col) of E is authoritative.
  for (int c = 0; c < E.outerSize(); ++c)
    for (Sparse::InnerIterator it(E, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(it.row(), it.col(), -it.value());

  // Bottom-right: F, offset by n1, only lower triangle authoritative.
  for (int c = 0; c < F.outerSize(); ++c)
    for (Sparse::InnerIterator it(F, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(n1 + it.row(), n1 + it.col(), it.value());

  // Bottom-left: A (n2 x n1), placed at block (n1.., 0..) -- this is already
  // the lower-left block, so every entry is in the lower triangle of K.
  for (int c = 0; c < A.outerSize(); ++c)
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, c); it; ++it)
      trips.emplace_back(n1 + it.row(), it.col(), it.value());

  Sparse K(n, n);
  K.setFromTriplets(trips.begin(), trips.end());
  return K;
}

Eigen::MatrixXd toDense(const Sparse& A) {
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) {
      Ad(it.row(), it.col()) = it.value();
      Ad(it.col(), it.row()) = it.value();
    }
  return Ad;
}

void checkKKT(int n1, int n2, unsigned seed, OrderingType ordering) {
  Sparse E = spdBlock(n1, 4, seed);
  Sparse F = spdBlock(n2, 4, seed + 17u);
  auto A = randomRect(n2, n1, 3, seed + 31u);
  Sparse K = buildKKT(E, F, A);
  const int n = n1 + n2;

  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.compute(K);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  const auto& inertia = solver.inertia();
  SYMLA_CHECK(inertia.n_pos == n2);
  SYMLA_CHECK(inertia.n_neg == n1);
  SYMLA_CHECK(inertia.n_zero == 0);
  SYMLA_CHECK(inertia.n_pos + inertia.n_neg + inertia.n_zero == n);

  // Residual check against a known solution.
  std::mt19937 rng(seed + 999u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);

  Eigen::MatrixXd Kd = toDense(K);
  Eigen::VectorXd b = Kd * xtrue;
  Eigen::MatrixXd B = b;  // n x 1
  Eigen::MatrixXd X = solver.solve(B);
  const double resid = (Kd * X - B).norm() / (Kd.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
  SYMLA_CHECK(resid < 1e-8);
  const double recErr = (X.col(0) - xtrue).norm() / std::max(1.0, xtrue.norm());
  SYMLA_CHECK(recErr < 1e-6);
}

}  // namespace

int main() {
  for (unsigned trial = 0; trial < 4; ++trial) {
    const unsigned seed = trial * 6151u + 17;
    checkKKT(10, 6, seed, OrderingType::AMD);
    checkKKT(30, 15, seed + 1u, OrderingType::AMD);
    checkKKT(80, 40, seed + 2u, OrderingType::AMD);
    checkKKT(30, 15, seed + 3u, OrderingType::Natural);
#ifdef SYMLA_HAVE_METIS
    checkKKT(60, 25, seed + 4u, OrderingType::Metis);
#endif
  }
  // Rectangular A block, n1 != n2 (over/under-determined constraint counts).
  checkKKT(50, 5, 4242u, OrderingType::AMD);
  checkKKT(5, 50, 4243u, OrderingType::AMD);

  std::cout << "symla kkt_test: all checks passed\n";
  return 0;
}
