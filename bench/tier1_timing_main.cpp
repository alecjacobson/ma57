// Throwaway timing driver for the Tier-1 performance fixes (see the task
// writeup): times analyzePattern()+factorize()+solve() on a .mtx matrix,
// serial and parallel, with several repetitions and several RHS columns
// (to stress solve()'s per-front cost, including the masked-top-block
// caching fix). Not part of the regular test suite.
#include "symla/solver.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <matrix.mtx> [reps] [nrhs] [serial|parallel|both]\n", argv[0]);
    return 1;
  }
  Sparse A;
  if (!Eigen::loadMarket(A, argv[1])) {
    std::fprintf(stderr, "failed to load %s\n", argv[1]);
    return 1;
  }
  A.makeCompressed();
  const int n = static_cast<int>(A.rows());
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  const int nrhs = argc > 3 ? std::atoi(argv[3]) : 4;
  const std::string mode = argc > 4 ? argv[4] : "both";
  std::printf("matrix: %s n=%d nnz=%lld reps=%d nrhs=%d mode=%s\n", argv[1], n, (long long)A.nonZeros(), reps, nrhs,
              mode.c_str());
  std::fflush(stdout);

  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd B(n, nrhs);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < nrhs; ++j) B(i, j) = dist(rng);

  std::vector<bool> parallelModes;
  if (mode == "serial" || mode == "both") parallelModes.push_back(false);
  if (mode == "parallel" || mode == "both") parallelModes.push_back(true);

  for (bool parallel : parallelModes) {
    symla::SymLDLT<double> solver;
    solver.setParallel(parallel);

    auto t0 = Clock::now();
    solver.analyzePattern(A);
    double analyzeTime = std::chrono::duration<double>(Clock::now() - t0).count();

    double bestFactor = 1e300, bestSolve = 1e300;
    for (int r = 0; r < reps; ++r) {
      auto tf0 = Clock::now();
      solver.factorize(A);
      double dtf = std::chrono::duration<double>(Clock::now() - tf0).count();
      bestFactor = std::min(bestFactor, dtf);

      auto ts0 = Clock::now();
      Eigen::MatrixXd X = solver.solve(B);
      double dts = std::chrono::duration<double>(Clock::now() - ts0).count();
      bestSolve = std::min(bestSolve, dts);
      std::printf("  parallel=%d rep=%d factorize=%.4fs solve=%.4fs\n", (int)parallel, r, dtf, dts);
      std::fflush(stdout);
    }
    std::printf("  parallel=%d analyze=%.4fs best_factorize=%.4fs best_solve=%.4fs total=%.4fs singular=%d\n",
                (int)parallel, analyzeTime, bestFactor, bestSolve, analyzeTime + bestFactor + bestSolve,
                solver.isSingular() ? 1 : 0);
    std::fflush(stdout);
  }
  return 0;
}
