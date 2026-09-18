// Ordering diagnostic: compares AMD vs METIS nested-dissection fill-reducing
// orderings on real Matrix Market systems, reporting purely structural
// SymbolicFactor::analyze() numbers (analysis time, supernode count,
// average/max supernode size, predicted total storage nnz) and, optionally,
// end-to-end SymLDLT<double>::compute() factorize wall-clock time under each
// ordering.
//
// This was written to diagnose a large real-world regression: symla was
// 5x-280x slower than CHOLMOD/other solvers on a 720K-vertex mesh benchmark
// (sparse-solver-benchmark), with the gap widening sharply as the matrix
// densifies (k=1 Laplacian -> k=3 triharmonic at the same n). The leading
// hypothesis was that AMD (symla's then-default ordering) produces a badly
// unbalanced elimination tree / one or a few huge dominant fronts on this
// large, mesh-like (near-3D at higher k) sparsity pattern, and that METIS
// nested dissection does much better. See include/symla/solver.hpp for the
// resulting default-ordering change and its rationale.
//
// Usage: symla_ordering_diag <matrix.mtx> [--factorize]
//   --factorize also runs full SymLDLT<double>::compute() under each
//   ordering that completes in reasonable time (skips AMD if its structural
//   numbers already look pathological and a full factorize would be very
//   slow -- pass --factorize-amd-anyway to force it).
#include "symla/solver.hpp"
#include "symla/symbolic.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void reportStructural(const std::string& label, const symla::SymbolicFactor& sf) {
  const auto& sns = sf.supernodes;
  std::int64_t totalStorage = 0;
  std::size_t maxFront = 0;
  double sumCols = 0;
  for (const auto& sn : sns) {
    totalStorage += sn.storageNnz();
    maxFront = std::max(maxFront, sn.rowPattern.size());
    sumCols += sn.ncols;
  }
  const double avgCols = sns.empty() ? 0.0 : sumCols / static_cast<double>(sns.size());
  std::printf("  [%s] supernodes=%zu avgCols=%.2f maxFrontRows=%zu totalStorageNnz=%lld\n", label.c_str(),
              sns.size(), avgCols, maxFront, static_cast<long long>(totalStorage));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <matrix.mtx> [--factorize] [--factorize-amd-anyway]\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  bool doFactorize = false;
  bool factorizeAmdAnyway = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--factorize") doFactorize = true;
    if (a == "--factorize-amd-anyway") factorizeAmdAnyway = true;
  }

  Sparse A;
  if (!Eigen::loadMarket(A, path)) {
    std::fprintf(stderr, "failed to load %s\n", path.c_str());
    return 1;
  }
  A.makeCompressed();
  std::printf("matrix: %s  n=%d nnz=%lld\n", path.c_str(), static_cast<int>(A.rows()),
              static_cast<long long>(A.nonZeros()));

  for (auto [label, type] :
       std::vector<std::pair<std::string, symla::OrderingType>>{
           {"AMD", symla::OrderingType::AMD},
#ifdef SYMLA_HAVE_METIS
           {"Metis", symla::OrderingType::Metis},
#endif
       }) {
    auto t0 = Clock::now();
    symla::SymbolicFactor sf = symla::SymbolicFactor::analyze(A, type);
    const double tAnalyze = seconds_since(t0);
    std::printf("ordering=%s analyzeTime=%.3fs\n", label.c_str(), tAnalyze);
    reportStructural(label, sf);

    if (doFactorize) {
      if (label == "AMD" && !factorizeAmdAnyway) {
        std::printf("  (skipping full factorize under AMD; pass --factorize-amd-anyway to force)\n");
        continue;
      }
      symla::SymLDLT<double> solver;
      solver.setOrdering(type);
      auto tf0 = Clock::now();
      solver.compute(A);
      const double tFactor = seconds_since(tf0);
      std::printf("  [%s] full compute() wall time = %.3fs singular=%d\n", label.c_str(), tFactor,
                  solver.isSingular() ? 1 : 0);
    }
  }
  return 0;
}
