// Phase 5 stability test: exact zero diagonal entries forcing 2x2 pivots,
// exercised at the *sparse multifrontal* level (through the real public
// analyzePattern -> factorize -> solve path), not just Phase 2's dense
// kernel unit test (dense_kernel_test.cpp already covers the dense-only
// case standalone).
//
// Construction: n/2 independent 2x2 "hinge" blocks
//     [ 0  c ]
//     [ c  0 ]
// (c != 0) placed at (2i,2i)/(2i,2i+1)/(2i+1,2i+1), each of which is
// structurally forced into a 2x2 Bunch-Kaufman pivot (lambda = |c| > 0,
// A(k,k) = 0 so the 1x1-at-k test fails outright, and since this is a 2x2
// isolated block sigma = 0 too, so the 1x1-swap test also fails, leaving
// only the 2x2 pivot as an acceptable choice) -- plus light sparse coupling
// between hinges so this is genuinely routed through the elimination
// tree/supernode assembly, not one big block-diagonal trivial case.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::PivotKind;
using symla::SymLDLT;

namespace {

Eigen::MatrixXd toDense(const Sparse& A) {
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Ad(it.row(), it.col()) = it.value();
  return Ad;
}

Sparse buildHingeChain(int nHinges, unsigned seed) {
  const int n = 2 * nHinges;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> cDist(0.5, 2.0);
  std::uniform_real_distribution<double> couplingDist(-0.05, 0.05);

  std::vector<Eigen::Triplet<double>> trips;
  for (int h = 0; h < nHinges; ++h) {
    const int a = 2 * h, b = 2 * h + 1;
    const double c = cDist(rng);
    trips.emplace_back(a, a, 0.0);  // exact zero diagonal
    trips.emplace_back(b, b, 0.0);  // exact zero diagonal
    trips.emplace_back(b, a, c);
    trips.emplace_back(a, b, c);
  }
  // Light coupling between consecutive hinges (kept small so it doesn't
  // change the requirement of a 2x2 pivot within each hinge -- lambda from
  // the hinge's own off-diagonal partner still dominates).
  for (int h = 0; h + 1 < nHinges; ++h) {
    const int a = 2 * h, bnext = 2 * (h + 1);
    const double v = couplingDist(rng);
    trips.emplace_back(bnext, a, v);
    trips.emplace_back(a, bnext, v);
  }

  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

int countTwoByTwoPivots(const symla::NumericFactor& nf) {
  int count = 0;
  for (const auto& f : nf.fronts)
    for (const auto& pb : f.pivotBlocks)
      if (pb.kind == PivotKind::TwoByTwo) ++count;
  return count;
}

void checkHingeChain(int nHinges, unsigned seed, OrderingType ordering) {
  Sparse A = buildHingeChain(nHinges, seed);
  const int n = 2 * nHinges;

  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  const auto& inertia = solver.inertia();
  SYMLA_CHECK(inertia.n_pos + inertia.n_neg + inertia.n_zero == n);
  // Each hinge contributes exactly one + and one - eigenvalue sign (2x2
  // block with det = 0*0 - c^2 = -c^2 < 0), so with only light coupling
  // perturbing things, the total should be balanced.
  SYMLA_CHECK(inertia.n_pos == nHinges);
  SYMLA_CHECK(inertia.n_neg == nHinges);

  // The key structural assertion: this really did route through 2x2 pivots
  // at the sparse/multifrontal level, not just get silently delayed to a
  // dense fallback or produce 1x1 pivots via some other mechanism.
  SYMLA_CHECK(countTwoByTwoPivots(solver.numericFactor()) >= nHinges);

  std::mt19937 rng(seed + 55u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Ad = toDense(A);
  Eigen::MatrixXd B = Ad * xtrue;
  Eigen::MatrixXd X = solver.solve(B);
  const double resid = (Ad * X - B).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
  SYMLA_CHECK(resid < 1e-8);
}

}  // namespace

int main() {
  for (int nHinges : {2, 10, 50, 200}) {
    checkHingeChain(nHinges, 1000u + nHinges, OrderingType::AMD);
    checkHingeChain(nHinges, 2000u + nHinges, OrderingType::Natural);
#ifdef SYMLA_HAVE_METIS
    checkHingeChain(nHinges, 3000u + nHinges, OrderingType::Metis);
#endif
  }

  std::cout << "symla zero_diagonal_test: all checks passed\n";
  return 0;
}
