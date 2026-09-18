// Phase 3 correctness fuzzing: full analyzePattern -> factorize pipeline on
// random sparse symmetric indefinite matrices at several sizes, cross-
// checked against:
//   (a) reconstruction of P A P^T from the flattened multifrontal L/D/perm
//       (test/unit/multifrontal_test_helpers.hpp), and a forward/diag/back
//       solve residual against a random RHS;
//   (b) a fully independent dense oracle: convert the same matrix to dense
//       and run Phase 2's DenseLDLT::factor on the *whole* matrix with no
//       sparsity/tree structure at all, then compare inertia. Since inertia
//       is basis-independent (Sylvester's law of inertia), the multifrontal
//       result and the dense brute-force result must report the *same*
//       (n_pos, n_neg, n_zero) even though the two factorizations use
//       entirely different pivot orders/algorithms -- this is a strong,
//       implementation-independent correctness check that doesn't rely on
//       matching pivot sequences.
#include "symla/dense_kernel.hpp"
#include "symla/solver.hpp"
#include "multifrontal_test_helpers.hpp"
#include "test_helpers.hpp"

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

void fuzzOne(int n, int avgNnzPerRow, unsigned seed, OrderingType ordering) {
  Sparse A = symla_test::randomSparseIndefinite(n, avgNnzPerRow, seed);

  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.analyzePattern(A);
  solver.factorize(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  // (a) Reconstruction + residual.
  auto flat = symla_test::flatten(solver.symbolicFactor(), solver.numericFactor());
  Eigen::MatrixXd rec = flat.L * flat.D * flat.L.transpose();
  Eigen::MatrixXd Ap = symla_test::permuteDense(A, flat.origIndex);
  const double relErr = (rec - Ap).norm() / std::max(1.0, Ap.cwiseAbs().maxCoeff());
  SYMLA_CHECK(relErr < 1e-6);

  std::mt19937 rng(seed + 555u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) b(i) = dist(rng);
  Eigen::VectorXd x = symla_test::solveFlattened(solver.numericFactor(), flat, b);
  Eigen::MatrixXd Ad = toDense(A);
  const double resid = (Ad * x - b).norm() / (Ad.cwiseAbs().maxCoeff() * std::max(1.0, x.norm()) + b.norm());
  SYMLA_CHECK(resid < 1e-6);

  // (b) Inertia cross-check against an independent dense oracle.
  Eigen::MatrixXd Aoracle = Ad;
  symla::BlockDiagonalD<double> Doracle;
  auto oracleRes = symla::DenseLDLT<double>::factor(Aoracle, Doracle);
  SYMLA_CHECK(oracleRes.n_factored == n);  // the dense oracle should never need to delay on these matrices

  const auto& si = solver.inertia();
  SYMLA_CHECK(si.n_pos == oracleRes.inertia.n_pos);
  SYMLA_CHECK(si.n_neg == oracleRes.inertia.n_neg);
  SYMLA_CHECK(si.n_zero == oracleRes.inertia.n_zero);
  SYMLA_CHECK(si.n_pos + si.n_neg + si.n_zero == n);
}

void fuzzSPD(int n, int avgNnzPerRow, unsigned seed, OrderingType ordering) {
  Sparse A = symla_test::randomSparseSPD(n, avgNnzPerRow, seed);
  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());
  // SPD -> all-positive inertia, a strong cross-check on its own (no oracle
  // needed: this is a structural guarantee of a correct LDL^T of an SPD
  // matrix, independent of pivot order).
  SYMLA_CHECK(solver.inertia().n_pos == n);
  SYMLA_CHECK(solver.inertia().n_neg == 0);
  SYMLA_CHECK(solver.inertia().n_zero == 0);

  auto flat = symla_test::flatten(solver.symbolicFactor(), solver.numericFactor());
  Eigen::MatrixXd rec = flat.L * flat.D * flat.L.transpose();
  Eigen::MatrixXd Ap = symla_test::permuteDense(A, flat.origIndex);
  SYMLA_CHECK((rec - Ap).norm() / std::max(1.0, Ap.cwiseAbs().maxCoeff()) < 1e-6);
}

}  // namespace

int main() {
  for (int n : {20, 100, 500}) {
    for (unsigned trial = 0; trial < 4; ++trial) {
      const unsigned seed = trial * 7919u + static_cast<unsigned>(n);
      fuzzOne(n, 5, seed, OrderingType::AMD);
      fuzzOne(n, 5, seed, OrderingType::Natural);
      fuzzSPD(n, 5, seed + 31u, OrderingType::AMD);
#ifdef SYMLA_HAVE_METIS
      fuzzOne(n, 5, seed + 61u, OrderingType::Metis);
#endif
    }
  }

  std::cout << "symla multifrontal_correctness_test: all checks passed\n";
  return 0;
}
