// Throwaway diagnostic driver: times serial vs. parallel factorize() on a
// .mtx matrix, for tuning the task-DAG scheduling-granularity fix in
// multifrontal.hpp (see the project's performance-diagnosis notes).
#include "symla/solver.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <matrix.mtx> [reps] [serial|parallel|both]\n", argv[0]);
    return 1;
  }
  Sparse A;
  if (!Eigen::loadMarket(A, argv[1])) {
    std::fprintf(stderr, "failed to load %s\n", argv[1]);
    return 1;
  }
  A.makeCompressed();
  const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
  const std::string mode = argc > 3 ? argv[3] : "both";
  std::printf("matrix: %s n=%d nnz=%lld reps=%d mode=%s\n", argv[1], (int)A.rows(), (long long)A.nonZeros(), reps,
              mode.c_str());
  std::fflush(stdout);

  std::vector<bool> parallelModes;
  if (mode == "serial" || mode == "both") parallelModes.push_back(false);
  if (mode == "parallel" || mode == "both") parallelModes.push_back(true);

  for (bool parallel : parallelModes) {
    symla::SymLDLT<double> solver;
    solver.setParallel(parallel);
    solver.analyzePattern(A);
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
      auto t0 = Clock::now();
      solver.factorize(A);
      double dt = std::chrono::duration<double>(Clock::now() - t0).count();
      best = std::min(best, dt);
      std::printf("  parallel=%d rep=%d time=%.4fs\n", (int)parallel, r, dt);
      std::fflush(stdout);
    }
    std::printf("  parallel=%d best_factorize_time=%.4fs singular=%d\n", (int)parallel, best,
                solver.isSingular() ? 1 : 0);
    std::fflush(stdout);
  }
  return 0;
}
