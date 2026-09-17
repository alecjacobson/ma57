// Phase 6: ThreadSanitizer smoke test for the task-DAG parallel
// factorize()/solve() path. Deliberately small (TSan instrumentation makes
// everything much slower) but non-trivial: enough columns/nnz to produce a
// real supernode tree with several levels and multiple leaves so there is
// genuine concurrent front processing to race-check, plus a couple of
// delayed pivots (indefinite matrix) to exercise the cross-front
// dependency paths. Not part of the default `ctest` gate (see
// SYMLA_WITH_TSAN_TEST in test/CMakeLists.txt) -- build with
// `-DSYMLA_WITH_TSAN_TEST=ON` in a separate build directory and run this
// target directly.
//
// KNOWN LIMITATION (investigated during Phase 6, this machine's toolchain
// is GCC 11 + libgomp, no clang available): running this target under GCC's
// libgomp reports several hundred "data race" warnings, but every single
// one is between (a) a read of `SymbolicFactor`/`NumericFactor` state
// performed *inside* a completed `#pragma omp task` during factorize(), and
// (b) a *later* write that only happens after `factorize()`/`solve()` have
// both already returned (e.g. in `~SymLDLT()`'s destructor teardown, well
// after `main` regains control) -- i.e. TSan is failing to recognize
// libgomp's task/taskwait/parallel-end barriers as synchronization points
// at all, rather than finding any genuine happens-before violation between
// concurrently-running fronts. This is a documented gap in GCC libgomp's
// ThreadSanitizer interoperability (libgomp implements its barriers with
// raw futex syscalls rather than the pthread mutex/condvar primitives
// TSan's interceptors track, so *any* OpenMP task-parallel program built
// with GCC produces this same class of false positive under TSan -- it is
// not specific to this code). LLVM's libomp is reported to have much better
// TSan support; this machine only has libomp headers/runtime installed
// (libomp-14-dev/libomp5-14), not a `clang` compiler binary, so
// cross-checking with clang+libomp was not possible in this environment --
// flagged as a Phase 6 follow-up if a clang toolchain becomes available.
// Correctness/no-races confidence for this phase instead rests on
// thread_equivalence_test.cpp (bit-identical results, serial vs. every
// tested thread count) and repeated_run_determinism_test.cpp (bit-identical
// results across 10 repeated parallel runs) -- both of which would be
// expected to show run-to-run or thread-count-to-thread-count divergence if
// a real race existed, and neither does.
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <iostream>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

int main() {
  const int n = 400;
  Sparse A = symla_test::randomSparseIndefinite(n, 8, 20260917u);

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.setParallel(true);
  solver.setNumThreads(8);
  solver.compute(A);
  SYMLA_CHECK(!solver.isSingular());

  Eigen::MatrixXd Xtrue = Eigen::MatrixXd::Ones(n, 2);
  Eigen::MatrixXd B = A.selfadjointView<Eigen::Lower>() * Xtrue;
  Eigen::MatrixXd X = solver.solve(B);

  Eigen::MatrixXd AX = A.selfadjointView<Eigen::Lower>() * X;
  const double residual = (AX - B).norm() / std::max(1.0, B.norm());
  SYMLA_CHECK(residual < 1e-6);

  std::cout << "symla tsan_concurrency_test: n=" << n << " residual=" << residual << " PASSED\n";
  return 0;
}
