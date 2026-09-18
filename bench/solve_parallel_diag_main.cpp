// Timing driver for solve.hpp's task-DAG parallel solve() cost model: (a)
// nrhs=1 parallel-vs-serial on many-small-fronts trees (the shape that
// caused the original Task 13 factorize() scheduling-granularity
// regression, see multifrontal.hpp's MultifrontalOptions::task_cutoff doc
// comment) must not regress, (b) a large-front tree at larger nrhs should
// show real speedup. Mirrors parallel_diag_main.cpp's role for factorize().
// Not part of the regular test suite (`ctest`); see
// test/concurrency/solve_thread_equivalence_test.cpp for the correctness
// tests. Opportunistically uses /tmp/symla_small/k{1..5}_Q.mtx (the
// many-tiny-supernode meshes from the Task 13 diagnosis) and
// bench/matrices/GHS_indef/{qpband,bratu3d} (bench/fetch_matrices.py) if
// present; skips gracefully otherwise.
#include "symla/solver.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
namespace fs = std::filesystem;

namespace {

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Synthetic block-arrow / KKT-shaped matrix: `numBlocks` independent dense
// blocks of size `blockSize`, each coupled to a shared set of `numCoupling`
// "arrow" variables. Under AMD/elimination-tree ordering this naturally
// produces a supernode tree with `numBlocks` large, expensive, mutually
// independent sibling subtrees feeding into a root coupling front --
// exactly the shape that should benefit from task-DAG sibling parallelism
// in solve() (real per-front cost, genuine tree-level concurrency) --
// without GHS_indef/bratu3d's unrelated, separately-documented slow
// symbolic-analysis cost at this scale (see real_matrix_test.cpp's
// kMaxNForCtest comment).
Sparse blockArrow(int numBlocks, int blockSize, int numCoupling, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> valDist(0.05, 0.5);
  const int n = numBlocks * blockSize + numCoupling;
  std::vector<Eigen::Triplet<double>> trips;
  std::vector<double> rowAbsSum(n, 0.0);
  auto addSym = [&](int i, int j, double v) {
    trips.emplace_back(i, j, v);
    trips.emplace_back(j, i, v);
    rowAbsSum[i] += std::abs(v);
    rowAbsSum[j] += std::abs(v);
  };
  for (int b = 0; b < numBlocks; ++b) {
    const int base = b * blockSize;
    for (int i = 0; i < blockSize; ++i) {
      for (int j = i + 1; j < blockSize; ++j) addSym(base + i, base + j, valDist(rng));
      // Couple every block variable to a couple of the shared arrow vars.
      for (int c = 0; c < std::min(numCoupling, 2); ++c) addSym(base + i, numBlocks * blockSize + c, valDist(rng));
    }
  }
  for (int c = 0; c < numCoupling; ++c)
    for (int c2 = c + 1; c2 < numCoupling; ++c2) addSym(numBlocks * blockSize + c, numBlocks * blockSize + c2, valDist(rng));
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, rowAbsSum[i] + 1.0);
  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

void timeSolve(const std::string& label, const Sparse& A, int nrhs, int threads, int repeats) {
  const int n = static_cast<int>(A.rows());
  symla::SymLDLT<double> solver;
  solver.setOrdering(symla::OrderingType::AMD);
  solver.setParallel(true);
  solver.setNumThreads(threads);
  solver.compute(A);
  if (solver.isSingular()) {
    std::cout << label << ": singular, skipping\n";
    return;
  }

  std::mt19937 rng(777u);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd Xtrue(n, nrhs);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
  Eigen::MatrixXd B = A.selfadjointView<Eigen::Lower>() * Xtrue;

  solver.setParallel(false);
  Eigen::MatrixXd Xs = solver.solve(B);  // warm-up + correctness reference
  double t0 = now();
  for (int r = 0; r < repeats; ++r) solver.solve(B);
  double serialTime = (now() - t0) / repeats;

  solver.setParallel(true);
  Eigen::MatrixXd Xp = solver.solve(B);
  double err = (Xp - Xs).norm() / std::max(1.0, Xs.norm());
  t0 = now();
  for (int r = 0; r < repeats; ++r) solver.solve(B);
  double parTime = (now() - t0) / repeats;

  std::cout << label << ": n=" << n << " nrhs=" << nrhs << " threads=" << threads << " serial=" << serialTime * 1e3
            << "ms parallel=" << parTime * 1e3 << "ms speedup=" << (serialTime / parTime) << "x relDiff=" << err
            << "\n";
}

}  // namespace

int main() {
  // Many-small-fronts trees (the Task 13 regression shape): nrhs=1
  // parallel solve() should track serial within noise.
  for (const char* name : {"k1", "k2", "k3", "k4", "k5"}) {
    fs::path p = fs::path("/tmp/symla_small") / (std::string(name) + "_Q.mtx");
    if (!fs::exists(p)) continue;
    Sparse A;
    if (!Eigen::loadMarket(A, p.string())) continue;
    timeSolve(std::string("small-front ") + name, A, 1, 16, 20);
  }

  // Real KKT matrix with ~1442 mostly-tiny supernodes -- the exact matrix
  // that caused the original factorize() 18-30x slowdown (see
  // multifrontal.hpp). At nrhs=1, thread count should not matter (should
  // stay near the serial time across the whole range).
  const fs::path qpbandPath =
      fs::path(__FILE__).parent_path() / "matrices" / "GHS_indef" / "qpband" / "qpband.mtx";
  if (fs::exists(qpbandPath)) {
    Sparse A;
    if (Eigen::loadMarket(A, qpbandPath.string())) {
      for (int t : {1, 2, 4, 8, 16}) timeSolve("qpband", A, 1, t, 20);
      timeSolve("qpband", A, 10, 16, 20);
    }
  } else {
    std::cout << "qpband not found, skipping (see bench/fetch_matrices.py)\n";
  }

  // Synthetic block-arrow tree: several large, expensive, mutually
  // independent sibling subtrees -- should show a genuine speedup at
  // larger nrhs, where per-front work is substantial enough to amortize
  // task-spawn overhead. Fast to build/factorize (unlike bratu3d at this
  // scale, see real_matrix_test.cpp's documented symbolic-analysis cost).
  {
    Sparse A = blockArrow(/*numBlocks=*/16, /*blockSize=*/220, /*numCoupling=*/24, 2026091801u);
    timeSolve("block-arrow", A, 1, 16, 10);
    timeSolve("block-arrow", A, 10, 16, 10);
    timeSolve("block-arrow", A, 50, 16, 5);
  }

  // Large-front tree (real 3D PDE KKT system): should show a genuine
  // speedup at larger nrhs, where per-front work is substantial enough to
  // amortize task-spawn overhead. Gated behind an env var: symbolic
  // analysis alone is documented to take 90+ seconds at this scale (see
  // real_matrix_test.cpp's kMaxNForCtest comment, an unrelated pre-existing
  // limitation), so this is opt-in rather than part of the default run.
  if (std::getenv("SYMLA_BENCH_INCLUDE_BRATU3D")) {
    const fs::path bratuPath =
        fs::path(__FILE__).parent_path() / "matrices" / "GHS_indef" / "bratu3d" / "bratu3d.mtx";
    if (fs::exists(bratuPath)) {
      Sparse A;
      if (Eigen::loadMarket(A, bratuPath.string())) {
        timeSolve("bratu3d", A, 1, 16, 5);
        timeSolve("bratu3d", A, 10, 16, 5);
        timeSolve("bratu3d", A, 50, 16, 3);
      }
    } else {
      std::cout << "bratu3d not found, skipping (see bench/fetch_matrices.py)\n";
    }
  }
}
