// Unit tests for symla::MultifrontalFactorizer / symla::SymLDLT (Phase 3):
// small hand-traceable cases, a case exercising the delayed-pivot mechanism
// end to end, and a genuinely (numerically) singular case.
#include "symla/solver.hpp"
#include "multifrontal_test_helpers.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymbolicFactor;
using symla::SymLDLT;

namespace {

Sparse fromDense(const Eigen::MatrixXd& A) {
  const int n = static_cast<int>(A.rows());
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (A(i, j) != 0.0) trips.emplace_back(i, j, A(i, j));
  Sparse S(n, n);
  S.setFromTriplets(trips.begin(), trips.end());
  return S;
}

void checkFactorizationCorrectness(const Sparse& A, OrderingType ordering = OrderingType::AMD) {
  const int n = static_cast<int>(A.rows());
  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.analyzePattern(A);
  solver.factorize(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  auto flat = symla_test::flatten(solver.symbolicFactor(), solver.numericFactor());
  Eigen::MatrixXd rec = flat.L * flat.D * flat.L.transpose();
  Eigen::MatrixXd Ap = symla_test::permuteDense(A, flat.origIndex);
  const double denom = std::max(1.0, Ap.cwiseAbs().maxCoeff());
  SYMLA_CHECK((rec - Ap).norm() / denom < 1e-8);

  // Residual check against a random RHS.
  std::mt19937 rng(12345u + static_cast<unsigned>(n));
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) b(i) = dist(rng);
  Eigen::VectorXd x = symla_test::solveFlattened(solver.numericFactor(), flat, b);
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Ad(it.row(), it.col()) = it.value();
  double resid = (Ad * x - b).norm() / (Ad.cwiseAbs().maxCoeff() * x.norm() + b.norm() + 1e-300);
  SYMLA_CHECK(resid < 1e-8);
}

void testHandVerifiedSPD4x4() {
  Eigen::MatrixXd A(4, 4);
  A << 10, 1, 0, 1, 1, 8, 1, 0, 0, 1, 6, 1, 1, 0, 1, 9;
  checkFactorizationCorrectness(fromDense(A));
  checkFactorizationCorrectness(fromDense(A), OrderingType::Natural);
}

void testHandVerifiedIndefinite5x5() {
  // Diagonal signs mixed; still diagonally dominant per row so well
  // conditioned (exercises negative-inertia bookkeeping, not near-
  // singularity).
  Eigen::MatrixXd A(5, 5);
  A << 10, 1, 0, 0, 1, 1, -8, 1, 0, 0, 0, 1, 6, 1, 0, 0, 0, 1, -9, 1, 1, 0, 0, 1, 7;
  checkFactorizationCorrectness(fromDense(A));

  SymLDLT<double> solver;
  solver.compute(fromDense(A));
  SYMLA_CHECK(solver.inertia().n_pos + solver.inertia().n_neg + solver.inertia().n_zero == 5);
  SYMLA_CHECK(solver.inertia().n_zero == 0);
}

// Forces the delayed-pivot mechanism: an "arrow" matrix (hub column last)
// with the two leaf diagonals set to exactly zero. Under natural ordering
// this is the textbook arrow etree (elimination_tree_test.cpp): leaves 0
// and 1 are single-column supernodes each with the hub as their only
// off-diagonal entry, so their own fronts (front size 2: own column + hub
// row, n_eligible == 1) can never pivot a zero diagonal with no eligible
// partner -- both delay their entire column to the hub's supernode. The
// hub's own two strong pivots are processed first (well-conditioned), and
// the resulting rank-1 Schur-complement updates give the delayed leaf
// diagonals nonzero values, so they successfully finalize *at the hub's
// front* rather than their own -- i.e. this specifically exercises
// cross-supernode delayed-pivot bookkeeping end to end.
void testDelayedPivotArrow() {
  const int n = 4;  // leaves 0,1,2 ; hub 3 (leaf 2 fundamentally merges into hub's supernode)
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  A(0, 0) = 0.0;
  A(1, 1) = 0.0;
  A(2, 2) = 5.0;
  A(3, 3) = 20.0;
  A(0, 3) = A(3, 0) = 2.0;
  A(1, 3) = A(3, 1) = 3.0;
  A(2, 3) = A(3, 2) = 1.7;

  Sparse S = fromDense(A);
  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::Natural);  // keep the hand-analyzed arrow etree shape
  solver.analyzePattern(S);

  const SymbolicFactor& sf = solver.symbolicFactor();
  // Sanity-check the assumed supernode shape before trusting the rest of
  // the test's reasoning in the comment above.
  SYMLA_CHECK(sf.supernodes.size() == 3);  // {0}, {1}, {2,3}
  SYMLA_CHECK(sf.supernodes[0].ncols == 1 && sf.supernodes[0].firstCol == 0);
  SYMLA_CHECK(sf.supernodes[1].ncols == 1 && sf.supernodes[1].firstCol == 1);
  SYMLA_CHECK(sf.supernodes[2].ncols == 2 && sf.supernodes[2].firstCol == 2);

  solver.factorize(S);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  const auto& nf = solver.numericFactor();
  SYMLA_CHECK(nf.fronts.size() == 3);
  // Leaves each delay their sole column entirely at their own front...
  SYMLA_CHECK(nf.fronts[0].nPivots == 0);
  SYMLA_CHECK(nf.fronts[1].nPivots == 0);
  // ...and the hub's front (with both leaves' delayed columns arriving as
  // extra eligible columns) successfully finalizes all 4 pivots.
  SYMLA_CHECK(nf.fronts[2].nPivots == 4);
  SYMLA_CHECK(nf.inertia.n_pos + nf.inertia.n_neg + nf.inertia.n_zero == n);
  SYMLA_CHECK(nf.inertia.n_zero == 0);

  auto flat = symla_test::flatten(sf, nf);
  Eigen::MatrixXd rec = flat.L * flat.D * flat.L.transpose();
  Eigen::MatrixXd Ap = symla_test::permuteDense(S, flat.origIndex);
  SYMLA_CHECK((rec - Ap).norm() / Ap.cwiseAbs().maxCoeff() < 1e-10);

  Eigen::VectorXd b(n);
  b << 1, 2, 3, 4;
  Eigen::VectorXd x = symla_test::solveFlattened(nf, flat, b);
  double resid = (A * x - b).norm() / (A.cwiseAbs().maxCoeff() * x.norm() + b.norm());
  SYMLA_CHECK(resid < 1e-10);
}

// A genuinely (structurally + numerically) singular system: column 1 has no
// off-diagonal entries at all and a zero diagonal, i.e. an isolated
// all-zero row/column -- no front anywhere in the tree can ever find a
// pivot partner for it. Must be reported via isSingular()/singularColumns(),
// not crash or silently corrupt the rest of the factorization.
void testGenuinelySingular() {
  const int n = 3;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  A(0, 0) = 4.0;
  A(2, 2) = 3.0;
  A(0, 2) = A(2, 0) = 0.5;
  // column/row 1 stays all-zero.

  Sparse S = fromDense(A);
  SymLDLT<double> solver;
  solver.compute(S);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(solver.isSingular());
  std::vector<int> cols = solver.singularColumns();
  SYMLA_CHECK(cols.size() == 1);
  SYMLA_CHECK(cols[0] == 1);

  // The other two (genuinely nonsingular) pivots must still have finalized
  // correctly -- singularity of one column must not corrupt the rest.
  int totalPivots = 0;
  for (const auto& f : solver.numericFactor().fronts) totalPivots += f.nPivots;
  SYMLA_CHECK(totalPivots == n - 1);
  SYMLA_CHECK(solver.numericFactor().inertia.n_pos + solver.numericFactor().inertia.n_neg +
                  solver.numericFactor().inertia.n_zero ==
              n - 1);
}

void testRandomSmallIndefinite() {
  for (int n : {4, 6, 8}) {
    for (unsigned seed : {1u, 2u, 3u, 4u}) {
      Sparse A = symla_test::randomSparseIndefinite(n, 3, seed * 100u + static_cast<unsigned>(n));
      checkFactorizationCorrectness(A);
      checkFactorizationCorrectness(A, OrderingType::Natural);
    }
  }
}

}  // namespace

int main() {
  testHandVerifiedSPD4x4();
  testHandVerifiedIndefinite5x5();
  testDelayedPivotArrow();
  testGenuinelySingular();
  testRandomSmallIndefinite();

  std::cout << "symla multifrontal_test: all checks passed\n";
  return 0;
}
