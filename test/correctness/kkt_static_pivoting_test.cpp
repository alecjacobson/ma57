// Phase 7: KKT-aware static pivoting + inertia-controlled regularization +
// iterative refinement (Mode::StaticRegularized).
//
// Reuses the same SQD/KKT construction as test/correctness/kkt_test.cpp
// (K = [[-E, A^T], [A, F]], E/F SPD -> expected inertia (n2, n1, 0) by
// Sylvester's law of inertia / Vanderbei's SQD theory), but exercises the
// Phase 7 static-pivoting path instead of the Phase 2-6 Bunch-Kaufman
// threshold-pivoted path:
//
//   1. Genuine SQD systems: static pivoting with setKKTBlockSizes(n1, n2)
//      should succeed with zero or minimal regularization retries (this is
//      the concrete test of "pure diagonal pivoting in a fixed order is
//      provably safe for SQD matrices").
//   2. Near-SQD (not quite quasidefinite) systems: naive static pivoting
//      with too-small an initial delta produces wrong inertia; the retry
//      loop must detect it, escalate, and either recover the correct
//      inertia or throw cleanly within the retry budget (never silently
//      report a wrong-inertia factorization as successful).
//   3. Iterative refinement: for a case with non-trivial regularization
//      perturbation, solveWithRefinement() must substantially reduce the
//      residual compared to the raw (unrefined) static-pivoted solve().
//   4. Static-regularized vs. dynamic threshold-pivoted mode agree (after
//      refinement) on the same KKT-shaped systems, with a rough timing
//      comparison logged for context.
#include "symla/refine.hpp"
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::Mode;
using symla::OrderingType;
using symla::RefinementOptions;
using symla::SymLDLT;

namespace {

Sparse spdBlock(int n, int avgNnzPerRow, unsigned seed) { return symla_test::randomSparseSPD(n, avgNnzPerRow, seed); }

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

// Builds K = [[-E, A^T], [A, F]] (n = n1 + n2), lower-triangle-authoritative.
// Returns a copy of `M` with M(idx,idx) overridden to `newDiag`, leaving
// every off-diagonal entry (including M's row/col `idx` couplings to other
// rows/columns) untouched. Used to build the "near-SQD, not quite
// quasidefinite" test case realistically: a genuinely near-zero/wrong-sign
// diagonal entry at one position, but with the row/column's real
// off-diagonal support left intact -- exactly the "isolated bad diagonal,
// real coupling elsewhere" situation static (diagonal-only, no
// off-diagonal awareness) pivoting is meant to handle via regularization,
// as opposed to simply replacing the whole entry and potentially making the
// *true* system itself ill-conditioned (which iterative refinement cannot
// be expected to fix -- no factorization approach can recover accuracy an
// ill-conditioned true system doesn't have).
Sparse withDiagonalOverride(const Sparse& M, int idx, double newDiag) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int c = 0; c < M.outerSize(); ++c) {
    for (Sparse::InnerIterator it(M, c); it; ++it) {
      if (static_cast<int>(it.row()) == idx && static_cast<int>(it.col()) == idx) continue;
      trips.emplace_back(it.row(), it.col(), it.value());
    }
  }
  trips.emplace_back(idx, idx, newDiag);
  Sparse out(M.rows(), M.cols());
  out.setFromTriplets(trips.begin(), trips.end());
  return out;
}

Sparse buildKKT(const Sparse& E, const Sparse& F, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A) {
  const int n1 = static_cast<int>(E.rows());
  const int n2 = static_cast<int>(F.rows());
  const int n = n1 + n2;
  std::vector<Eigen::Triplet<double>> trips;

  for (int c = 0; c < E.outerSize(); ++c)
    for (Sparse::InnerIterator it(E, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(it.row(), it.col(), -it.value());

  for (int c = 0; c < F.outerSize(); ++c)
    for (Sparse::InnerIterator it(F, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(n1 + it.row(), n1 + it.col(), it.value());

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

double directResidual(const Eigen::MatrixXd& Kd, const Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
  return (Kd * X - B).norm() / (Kd.cwiseAbs().maxCoeff() * std::max(1.0, X.norm()) + B.norm());
}

// --- Test 1: genuine SQD systems, minimal/no regularization. ---
void checkGenuineSQD(int n1, int n2, unsigned seed, OrderingType ordering) {
  Sparse E = spdBlock(n1, 4, seed);
  Sparse F = spdBlock(n2, 4, seed + 17u);
  auto A = randomRect(n2, n1, 3, seed + 31u);
  Sparse K = buildKKT(E, F, A);
  const int n = n1 + n2;

  SymLDLT<double> solver;
  solver.setOrdering(ordering);
  solver.setMode(Mode::StaticRegularized);
  solver.setKKTBlockSizes(n1, n2);
  solver.analyzePattern(K);
  solver.factorize(K);

  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());
  const auto& inertia = solver.inertia();
  SYMLA_CHECK(inertia.n_pos == n2);
  SYMLA_CHECK(inertia.n_neg == n1);
  SYMLA_CHECK(inertia.n_zero == 0);

  // The theoretical claim under test: a genuinely SQD system needs no (or
  // essentially no) regularization retries -- pure diagonal pivoting in a
  // fixed sparsity-driven order already gives the right inertia.
  SYMLA_CHECK(solver.regularizationRetriesUsed() <= 1);

  std::mt19937 rng(seed + 999u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Kd = toDense(K);
  Eigen::MatrixXd B = Kd * xtrue;
  Eigen::MatrixXd X = solver.solve(B);
  const double resid = directResidual(Kd, X, B);
  SYMLA_CHECK(resid < 1e-6);

  std::cout << "  genuineSQD n1=" << n1 << " n2=" << n2 << ": retries=" << solver.regularizationRetriesUsed()
            << " totalPerturbation=" << solver.totalPerturbation() << " numPerturbed=" << solver.numPerturbedPivots()
            << " residual=" << resid << "\n";
}

// --- Test 2: near-SQD (E driven slightly indefinite), regularization must
// trigger and either recover the right inertia or throw cleanly. ---
void checkNearSQDRegularizationTriggers(int n1, int n2, unsigned seed) {
  Sparse E = spdBlock(n1, 4, seed);
  Sparse F = spdBlock(n2, 4, seed + 17u);
  auto A = randomRect(n2, n1, 3, seed + 31u);
  // Drive E's (0,0) diagonal entry negative (E locally indefinite at that
  // one position) while leaving all of E's off-diagonal couplings for row/
  // column 0 untouched -- realistic "near-SQD" construction: the isolated
  // diagonal is bad, but the true off-diagonal support (and the rest of E,
  // F, A) is exactly as good as the genuinely-SQD case. The magnitude
  // (0.05) is large enough that a single small default-scale perturbation
  // will not fix the resulting wrong-signed K(0,0) = -E(0,0), forcing the
  // retry loop to actually escalate `delta` across multiple attempts.
  E = withDiagonalOverride(E, 0, -0.05);
  Sparse K = buildKKT(E, F, A);
  const int n = n1 + n2;

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setMode(Mode::StaticRegularized);
  solver.setKKTBlockSizes(n1, n2);
  solver.setInitialRegularizationDelta(1e-6);
  solver.setRegularizationGrowth(10.0);
  solver.setMaxRegularizationRetries(10);
  solver.analyzePattern(K);
  solver.factorize(K);  // must not throw: 10 retries * 10x growth from 1e-6 easily covers a ~5-magnitude fix

  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());
  const auto& inertia = solver.inertia();
  SYMLA_CHECK(inertia.n_pos == n2);
  SYMLA_CHECK(inertia.n_neg == n1);
  SYMLA_CHECK(inertia.n_zero == 0);
  // Confirms escalation actually happened (not a lucky first pass).
  SYMLA_CHECK(solver.regularizationRetriesUsed() >= 1);

  std::cout << "  nearSQD-regularized n1=" << n1 << " n2=" << n2 << ": retries=" << solver.regularizationRetriesUsed()
            << " finalDelta=" << solver.lastRegularizationDelta() << " totalPerturbation=" << solver.totalPerturbation()
            << "\n";

  // Clean-failure path: an absurdly small retry budget for the same
  // pathological matrix must throw rather than silently report a
  // wrong-inertia factorization as successful.
  {
    SymLDLT<double> failSolver;
    failSolver.setOrdering(OrderingType::AMD);
    failSolver.setMode(Mode::StaticRegularized);
    failSolver.setKKTBlockSizes(n1, n2);
    failSolver.setInitialRegularizationDelta(1e-6);
    failSolver.setRegularizationGrowth(10.0);
    failSolver.setMaxRegularizationRetries(0);  // only the first (non-escalated) attempt is allowed
    failSolver.analyzePattern(K);
    bool threw = false;
    try {
      failSolver.factorize(K);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    SYMLA_CHECK(threw);
    SYMLA_CHECK(!failSolver.factorized());
  }
}

// --- Test 3: iterative refinement recovers accuracy lost to regularization. ---
//
// Uses a *genuinely* quasidefinite (well-conditioned, true inertia exactly
// (n2, n1, 0)) system -- like Test 1 -- but forces meaningful regularization
// anyway via `setStaticPivotFloor()`: an artificially large floor makes
// every pivot look "too small" and get perturbed, even though none of them
// actually needed it. Because the sign expectation is correct everywhere
// (genuine SQD), the perturbation direction is always right, so the
// resulting factorization stays a close, useful preconditioner (just
// uniformly diagonally shifted) for the true system -- exactly the
// "regularization introduced real but recoverable error" scenario iterative
// refinement is meant to fix, isolated from any question of whether the
// true system itself is well-conditioned (it is, by construction, same as
// Test 1's SQD systems).
void checkIterativeRefinement(int n1, int n2, unsigned seed) {
  Sparse E = spdBlock(n1, 4, seed);
  Sparse F = spdBlock(n2, 4, seed + 17u);
  auto A = randomRect(n2, n1, 3, seed + 31u);
  Sparse K = buildKKT(E, F, A);
  const int n = n1 + n2;

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setMode(Mode::StaticRegularized);
  solver.setKKTBlockSizes(n1, n2);
  solver.setStaticPivotFloor(100.0);  // force every pivot to be judged "too small", regardless of true magnitude
  solver.setInitialRegularizationDelta(0.005);
  solver.setMaxRegularizationRetries(10);
  solver.analyzePattern(K);
  solver.factorize(K);
  SYMLA_CHECK(!solver.isSingular());
  SYMLA_CHECK(solver.inertia().n_pos == n2 && solver.inertia().n_neg == n1 && solver.inertia().n_zero == 0);

  std::mt19937 rng(seed + 555u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Kd = toDense(K);
  Eigen::MatrixXd B = Kd * xtrue;

  Eigen::MatrixXd Xraw = solver.solve(B);
  const double residRaw = directResidual(Kd, Xraw, B);

  RefinementOptions ropts;
  ropts.max_iterations = 10;
  ropts.tolerance = 1e-14;
  auto refined = solver.solveWithRefinement(K, B, ropts);
  const double residRefined = directResidual(Kd, refined.x, B);

  std::cout << "  iterativeRefinement n1=" << n1 << " n2=" << n2 << ": raw_residual=" << residRaw
            << " refined_residual=" << residRefined << " iterations=" << refined.diagnostics.iterations
            << " diag.final_relative_residual=" << refined.diagnostics.final_relative_residual << "\n";

  // The whole point of this test: refinement must substantially improve
  // accuracy whenever the raw solve had a non-trivial residual to begin
  // with.
  if (residRaw > 1e-10) {
    SYMLA_CHECK(residRefined <= residRaw * 1e-2);
  }
  SYMLA_CHECK(residRefined < 1e-8);
}

// --- Test 4: static-regularized vs. dynamic threshold-pivoted mode agree,
// plus a rough timing comparison. ---
void checkStaticVsDynamic(int n1, int n2, unsigned seed) {
  Sparse E = spdBlock(n1, 4, seed);
  Sparse F = spdBlock(n2, 4, seed + 17u);
  auto A = randomRect(n2, n1, 3, seed + 31u);
  Sparse K = buildKKT(E, F, A);
  const int n = n1 + n2;

  std::mt19937 rng(seed + 777u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd xtrue(n);
  for (int i = 0; i < n; ++i) xtrue(i) = dist(rng);
  Eigen::MatrixXd Kd = toDense(K);
  Eigen::MatrixXd B = Kd * xtrue;

  SymLDLT<double> staticSolver;
  staticSolver.setOrdering(OrderingType::AMD);
  staticSolver.setMode(Mode::StaticRegularized);
  staticSolver.setKKTBlockSizes(n1, n2);
  auto t0 = std::chrono::steady_clock::now();
  staticSolver.analyzePattern(K);
  staticSolver.factorize(K);
  auto t1 = std::chrono::steady_clock::now();
  auto staticRefined = staticSolver.solveWithRefinement(K, B);
  const double staticResid = directResidual(Kd, staticRefined.x, B);

  SymLDLT<double> dynamicSolver;
  dynamicSolver.setOrdering(OrderingType::AMD);
  dynamicSolver.setMode(Mode::ThresholdPivot);
  auto t2 = std::chrono::steady_clock::now();
  dynamicSolver.analyzePattern(K);
  dynamicSolver.factorize(K);
  auto t3 = std::chrono::steady_clock::now();
  Eigen::MatrixXd dynamicX = dynamicSolver.solve(B);
  const double dynamicResid = directResidual(Kd, dynamicX, B);

  const double staticSec = std::chrono::duration<double>(t1 - t0).count();
  const double dynamicSec = std::chrono::duration<double>(t3 - t2).count();

  std::cout << "  staticVsDynamic n1=" << n1 << " n2=" << n2 << ": static_residual(refined)=" << staticResid
            << " dynamic_residual=" << dynamicResid << " static_factorize_sec=" << staticSec
            << " dynamic_factorize_sec=" << dynamicSec << "\n";

  SYMLA_CHECK(staticSolver.inertia().n_pos == dynamicSolver.inertia().n_pos);
  SYMLA_CHECK(staticSolver.inertia().n_neg == dynamicSolver.inertia().n_neg);
  SYMLA_CHECK(staticResid < 1e-8);
  SYMLA_CHECK(dynamicResid < 1e-8);
}

}  // namespace

int main() {
  std::cout << "Test 1: genuine SQD systems, minimal regularization\n";
  for (unsigned trial = 0; trial < 3; ++trial) {
    const unsigned seed = trial * 4133u + 11;
    checkGenuineSQD(10, 6, seed, OrderingType::AMD);
    checkGenuineSQD(40, 20, seed + 1u, OrderingType::AMD);
    checkGenuineSQD(30, 15, seed + 2u, OrderingType::Natural);
#ifdef SYMLA_HAVE_METIS
    checkGenuineSQD(60, 25, seed + 3u, OrderingType::Metis);
#endif
  }
  checkGenuineSQD(50, 5, 9001u, OrderingType::AMD);
  checkGenuineSQD(5, 50, 9002u, OrderingType::AMD);

  std::cout << "Test 2: near-SQD systems trigger regularization retries\n";
  checkNearSQDRegularizationTriggers(20, 12, 2024u);
  checkNearSQDRegularizationTriggers(40, 20, 2025u);

  std::cout << "Test 3: iterative refinement recovers accuracy\n";
  checkIterativeRefinement(20, 12, 3033u);
  checkIterativeRefinement(40, 20, 3034u);

  std::cout << "Test 4: static-regularized vs. dynamic threshold-pivoted mode\n";
  checkStaticVsDynamic(60, 30, 4044u);
  checkStaticVsDynamic(200, 100, 4045u);

  std::cout << "symla kkt_static_pivoting_test: all checks passed\n";
  return 0;
}
