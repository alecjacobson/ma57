// Task-DAG parallel solve() correctness: single- vs multi-thread numerical
// equivalence, isolated from factorize()'s own parallelism.
//
// thread_equivalence_test.cpp and repeated_run_determinism_test.cpp already
// exercise solve() end-to-end (via SymLDLT::solve()), but always with
// factorize() and solve() sharing the same setParallel()/setNumThreads()
// setting, and only nrhs=2/3. This file isolates solve()'s own
// parallelism specifically: factorize *once* (serially, so the
// factorization itself is a fixed, deterministic reference), then call
// solve() repeatedly against that single NumericFactor with varying
// thread counts and both nrhs=1 (the case the plan called out as the one
// most at risk of a regression, since there's no multi-column throughput
// to amortize task overhead against) and nrhs>1, checking every run agrees
// with the serial solve() to a tight tolerance -- this directly targets
// the forward-solve extend-add restructuring in solve.hpp (the part that
// needed real restructuring vs. factorize()'s existing pattern), since a
// race or a wrong merge (e.g. overwrite instead of sum for a shared
// ancestor row touched by multiple sibling subtrees -- the actual bug
// found during development of this feature) would show up here as a
// numerical divergence from the serial reference, not just a crash.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;
namespace fs = std::filesystem;

namespace {

void checkSolveEquivalence(const std::string& label, const Sparse& A, const std::vector<int>& nrhsValues,
                            const std::vector<int>& threadCounts) {
  const int n = static_cast<int>(A.rows());

  // Factorize once, serially -- a fixed reference NumericFactor that every
  // solve() call below (serial and parallel alike) reuses unchanged, so any
  // divergence is attributable to solve() alone, not to factorize()'s own
  // (separately tested) parallelism.
  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setParallel(false);
  solver.compute(A);
  SYMLA_CHECK(solver.factorized());
  SYMLA_CHECK(!solver.isSingular());

  for (int nrhs : nrhsValues) {
    std::mt19937 rng(2026091801u ^ static_cast<unsigned>(n) ^ static_cast<unsigned>(nrhs));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = A.selfadjointView<Eigen::Lower>() * Xtrue;

    solver.setParallel(false);
    Eigen::MatrixXd Xserial = solver.solve(B);
    const double serialErr = (Xserial - Xtrue).norm() / std::max(1.0, Xtrue.norm());
    std::cout << "symla solve_thread_equivalence_test: " << label << " nrhs=" << nrhs
              << " serial recErr=" << serialErr << "\n";
    SYMLA_CHECK(serialErr < 1e-8);

    for (int t : threadCounts) {
      solver.setParallel(true);
      solver.setNumThreads(t);

      // Repeated calls at a fixed thread count: any run-to-run divergence
      // at all would indicate a race (the forward-solve extend-add merge
      // touches per-thread scratch buffers and a handful of shared
      // pre-sized arrays -- see solve.hpp's header comment).
      Eigen::MatrixXd first;
      for (int run = 0; run < 3; ++run) {
        Eigen::MatrixXd Xpar = solver.solve(B);
        SYMLA_CHECK(Xpar.rows() == n);
        SYMLA_CHECK(Xpar.cols() == nrhs);
        if (run == 0) {
          first = Xpar;
          const double diff = (Xpar - Xserial).norm() / std::max(1.0, Xserial.norm());
          std::cout << "symla solve_thread_equivalence_test: " << label << " nrhs=" << nrhs << " threads=" << t
                     << " relDiffVsSerial=" << diff << "\n";
          SYMLA_CHECK(diff < 1e-10);
        } else {
          const double runDiff = (Xpar - first).norm() / std::max(1.0, first.norm());
          SYMLA_CHECK(runDiff < 1e-12);
        }
      }
    }
  }
}

}  // namespace

int main() {
  const std::vector<int> nrhsValues = {1, 5};
  const std::vector<int> threadCounts = {1, 4, 16, 32, 64};

  checkSolveEquivalence("synthetic n=500", symla_test::randomSparseIndefinite(500, 6, 142u), nrhsValues,
                         threadCounts);
  checkSolveEquivalence("synthetic n=2000", symla_test::randomSparseIndefinite(2000, 8, 143u), nrhsValues,
                         threadCounts);
  // Denser/more-fill matrix: more supernode-tree "fan-in" (multiple sibling
  // subtrees sharing a common ancestor row) than the sparser cases above,
  // specifically exercising the forward-solve extend-add merge's summation
  // (not overwrite) requirement across siblings.
  checkSolveEquivalence("synthetic dense n=800", symla_test::randomSparseIndefinite(800, 20, 144u), nrhsValues,
                         threadCounts);

  const fs::path mtxPath = fs::path(__FILE__).parent_path().parent_path().parent_path() / "bench" / "matrices" /
                            "GHS_indef" / "qpband" / "qpband.mtx";
  if (fs::exists(mtxPath)) {
    Sparse A;
    if (Eigen::loadMarket(A, mtxPath.string()) && A.rows() == A.cols() && A.rows() > 0) {
      checkSolveEquivalence("real GHS_indef/qpband", A, nrhsValues, threadCounts);
    } else {
      std::cout << "symla solve_thread_equivalence_test: FAILED to load " << mtxPath << "\n";
    }
  } else {
    std::cout << "symla solve_thread_equivalence_test: skipping real matrix (bench/matrices not fetched)\n";
  }

  std::cout << "symla solve_thread_equivalence_test: PASSED\n";
  return 0;
}
