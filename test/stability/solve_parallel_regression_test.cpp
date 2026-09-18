// Regression guard for the task-DAG parallel solve() scheduling cost model
// (solve.hpp): parallel solve() at nrhs=1 on a many-small-fronts tree must
// not regress far past serial solve() time.
//
// Context: while adding task-DAG parallelism to solve() (mirroring
// factorize()'s existing Phase 6 task-DAG, see multifrontal.hpp), an early
// version reused too-permissive cost-model constants and regressed parallel
// solve() by ~150x on GHS_indef/qpband (a real KKT matrix with ~1442
// mostly-tiny supernodes -- the same shape that caused the original
// factorize() Task 13 scheduling-granularity bug) at nrhs=1: nearly every
// front's subtree cleared a too-low task-spawn cutoff, so OpenMP
// task-creation/taskwait overhead dominated the ~1ms of genuine work. This
// test locks in a fixed-size synthetic reproduction of that shape (many
// tiny fronts via disabled/near-disabled amalgamation on a mesh-like sparse
// pattern) so a future change to solve.hpp's cost model that reintroduces
// over-eager task spawning at nrhs=1 is caught here rather than only in
// ad hoc benchmarking. See bench/solve_parallel_diag_main.cpp for the fuller
// (non-ctest-gated) benchmark this constant was calibrated against.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <iostream>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;
using Clock = std::chrono::steady_clock;

namespace {

double timeSolveOnce(SymLDLT<double>& solver, const Eigen::MatrixXd& B, bool parallel, int repeats) {
  solver.setParallel(parallel);
  solver.solve(B);  // warm-up
  const auto t0 = Clock::now();
  for (int r = 0; r < repeats; ++r) solver.solve(B);
  return std::chrono::duration<double>(Clock::now() - t0).count() / repeats;
}

}  // namespace

int main() {
  // Many-small-fronts tree: a fairly sparse, large-ish random indefinite
  // matrix under AMD ordering typically yields a tree of many small
  // supernodes (few thousand of them for n in the low tens of thousands).
  const int n = 20000;
  Sparse A = symla_test::randomSparseIndefinite(n, 4, 20260918u);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setNumThreads(16);
  solver.compute(A);
  SYMLA_CHECK(!solver.isSingular());

  Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, 1);

  const double serialTime = timeSolveOnce(solver, B, /*parallel=*/false, /*repeats=*/20);
  const double parallelTime = timeSolveOnce(solver, B, /*parallel=*/true, /*repeats=*/20);

  std::cout << "symla solve_parallel_regression_test: n=" << n << " nrhs=1 serial=" << serialTime * 1e3
            << "ms parallel=" << parallelTime * 1e3 << "ms ratio=" << (parallelTime / serialTime) << "\n";

  // Generous bound: post-calibration measurements on this machine show
  // parallel solve() within ~1.0-1.1x of serial at nrhs=1 on shapes like
  // this (see bench/solve_parallel_diag_main.cpp); this leaves large
  // headroom for slower/noisier CI machines while still catching a
  // regression back toward the ~150x blowup the original miscalibration
  // caused.
  SYMLA_CHECK(parallelTime < serialTime * 10.0 + 0.01);

  std::cout << "symla solve_parallel_regression_test: PASSED\n";
  return 0;
}
