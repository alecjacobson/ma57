// Phase 6: single- vs multi-thread numerical equivalence.
//
// The task-DAG parallel factorize() (multifrontal.hpp) is designed so that
// each front's extend-add assembly always iterates its children in a fixed
// order determined once, serially, before the parallel region starts
// (`childrenSN[psn].push_back(si)`, built by a single-threaded forward pass
// over supernode index order) -- *not* "whichever child's task happens to
// finish first". Combined with pinning `Eigen::setNbThreads(1)` for the
// duration of the parallel region (so no front's own dense kernel work is
// itself further parallelized in a schedule-dependent way), this means
// floating-point summation order inside every front is identical between
// the serial driver and the parallel one, and identical across different
// thread counts -- so we expect not just "close" but bit-identical (or
// extremely close) factorization and solve results. The test still uses a
// tolerance rather than a hard `==` to avoid being fragile to legitimate
// future changes (e.g. adaptive nested Eigen threading for huge fronts,
// noted as a Phase 6 future-tuning idea) that could plausibly reorder
// some sums.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;
namespace fs = std::filesystem;

namespace {

struct RunResult {
  bool ok = false;
  int n_pos = 0, n_neg = 0, n_zero = 0;
  double factorResidual = 0.0;  // ||PAP^T - flattened LDL^T|| style check via solve residual
  Eigen::MatrixXd X;
};

double residualSelfAdjoint(const Sparse& A, const Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
  Eigen::MatrixXd AX = A.selfadjointView<Eigen::Lower>() * X;
  double Amax = 0.0;
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Amax = std::max(Amax, std::abs(it.value()));
  if (Amax == 0.0) Amax = 1.0;
  return (AX - B).norm() / (Amax * std::max(1.0, X.norm()) + B.norm());
}

RunResult runOnce(const Sparse& A, const Eigen::MatrixXd& B, bool parallel, int numThreads) {
  RunResult r;
  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setParallel(parallel);
  solver.setNumThreads(numThreads);
  solver.compute(A);
  if (solver.isSingular()) return r;
  r.X = solver.solve(B);
  r.factorResidual = residualSelfAdjoint(A, r.X, B);
  const auto& inertia = solver.inertia();
  r.n_pos = inertia.n_pos;
  r.n_neg = inertia.n_neg;
  r.n_zero = inertia.n_zero;
  r.ok = true;
  return r;
}

void checkEquivalence(const std::string& label, const Sparse& A, const std::vector<int>& threadCounts) {
  std::mt19937 rng(2026091701u ^ static_cast<unsigned>(A.rows()));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const int n = static_cast<int>(A.rows());
  Eigen::MatrixXd Xtrue(n, 2);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 2; ++j) Xtrue(i, j) = dist(rng);
  Eigen::MatrixXd B = A.selfadjointView<Eigen::Lower>() * Xtrue;

  RunResult serial = runOnce(A, B, /*parallel=*/false, /*numThreads=*/1);
  SYMLA_CHECK(serial.ok);
  std::cout << "symla thread_equivalence_test: " << label << " serial: residual=" << serial.factorResidual
            << " inertia=(" << serial.n_pos << "," << serial.n_neg << "," << serial.n_zero << ")\n";
  SYMLA_CHECK(serial.factorResidual < 1e-8);

  for (int t : threadCounts) {
    RunResult par = runOnce(A, B, /*parallel=*/true, /*numThreads=*/t);
    SYMLA_CHECK(par.ok);
    SYMLA_CHECK(par.n_pos == serial.n_pos);
    SYMLA_CHECK(par.n_neg == serial.n_neg);
    SYMLA_CHECK(par.n_zero == serial.n_zero);
    SYMLA_CHECK(par.factorResidual < 1e-8);

    const double diff = (par.X - serial.X).norm();
    const double scale = std::max(1.0, serial.X.norm());
    const double relDiff = diff / scale;
    std::cout << "symla thread_equivalence_test: " << label << " threads=" << t
              << " residual=" << par.factorResidual << " relDiffVsSerial=" << relDiff << "\n";
    SYMLA_CHECK(relDiff < 1e-10);
  }
}

}  // namespace

int main() {
  const std::vector<int> threadCounts = {1, 4, 16, 32, 64};

  checkEquivalence("synthetic n=500", symla_test::randomSparseIndefinite(500, 6, 42u), threadCounts);
  checkEquivalence("synthetic n=2000", symla_test::randomSparseIndefinite(2000, 8, 43u), threadCounts);

  // Real matrix, if fetched (bench/fetch_matrices.py): qpband, n=20000, is
  // the largest GHS_indef matrix the ctest gate's runtime budget comfortably
  // allows given how many (size x thread-count) combinations this test
  // already runs -- see real_matrix_test.cpp's kMaxNForCtest discussion for
  // why bratu3d (n=27792) is excluded from the ctest gate specifically.
  const fs::path mtxPath =
      fs::path(__FILE__).parent_path().parent_path().parent_path() / "bench" / "matrices" / "GHS_indef" / "qpband" / "qpband.mtx";
  if (fs::exists(mtxPath)) {
    Sparse A;
    if (Eigen::loadMarket(A, mtxPath.string()) && A.rows() == A.cols() && A.rows() > 0) {
      checkEquivalence("real GHS_indef/qpband", A, threadCounts);
    } else {
      std::cout << "symla thread_equivalence_test: FAILED to load " << mtxPath << "\n";
    }
  } else {
    std::cout << "symla thread_equivalence_test: skipping real matrix (bench/matrices not fetched)\n";
  }

  std::cout << "symla thread_equivalence_test: PASSED\n";
  return 0;
}
