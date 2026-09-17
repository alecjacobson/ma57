#pragma once

// Shared test-only helpers: random sparse SPD matrix generation and a
// brute-force dense symbolic-Cholesky oracle, used as an independent
// cross-check for elimination_tree.hpp / symbolic.hpp (never derived from
// any library's internals -- just literal dense symmetric Gaussian
// elimination fill tracking).

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// A check macro that behaves like assert() but is NOT compiled out under
// NDEBUG (the default CMAKE_BUILD_TYPE here is Release, which defines
// NDEBUG, which would silently turn every bare assert() in a test into a
// no-op). Unit tests in this project must use SYMLA_CHECK, not assert(),
// so that `ctest` actually exercises the assertions in the default build.
#define SYMLA_CHECK(cond)                                                           \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::cerr << "SYMLA_CHECK failed: " #cond " at " __FILE__ ":" << __LINE__     \
                << "\n";                                                            \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

namespace symla_test {

// Builds a random sparse SPD matrix of size n with roughly `avgNnzPerRow`
// off-diagonal entries per row (symmetrized), plus a diagonal strong enough
// to guarantee SPD via diagonal dominance. Returns the full symmetric
// pattern/values (both triangles stored) as an Eigen::SparseMatrix.
inline Eigen::SparseMatrix<double, Eigen::ColMajor, int> randomSparseSPD(int n, int avgNnzPerRow,
                                                                          unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> colDist(0, n - 1);
  std::uniform_real_distribution<double> valDist(0.1, 1.0);

  std::vector<Eigen::Triplet<double>> trips;
  std::vector<double> rowAbsSum(n, 0.0);
  std::vector<std::vector<std::pair<int, double>>> upper(n);  // upper[i] = {(j>i, val)}

  for (int i = 0; i < n; ++i) {
    const int nEntries = avgNnzPerRow > 0 ? (1 + static_cast<int>(rng() % avgNnzPerRow)) : 0;
    for (int e = 0; e < nEntries; ++e) {
      int j = colDist(rng);
      if (j <= i) continue;
      double v = valDist(rng);
      upper[i].push_back({j, v});
    }
  }

  for (int i = 0; i < n; ++i) {
    for (auto& pr : upper[i]) {
      const int j = pr.first;
      const double v = pr.second;
      trips.emplace_back(i, j, v);
      trips.emplace_back(j, i, v);
      rowAbsSum[i] += std::abs(v);
      rowAbsSum[j] += std::abs(v);
    }
  }

  for (int i = 0; i < n; ++i) {
    const double diag = rowAbsSum[i] + 1.0 + static_cast<double>(i % 5) * 0.1;
    trips.emplace_back(i, i, diag);
  }

  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// Independent oracle: performs literal dense symmetric Gaussian elimination
// (no pivoting -- SPD input guarantees it is well-defined) on a dense copy
// of the *already permuted* symmetric pattern (nonzero = true, treating any
// stored entry, including explicit zeros, as structurally nonzero), and
// returns the resulting fill pattern's per-column nonzero counts
// (colCount[j] = number of rows i >= j with L(i,j) structurally nonzero).
// O(n^3) -- intended for small-to-moderate n in unit tests only.
template <typename SparseMatrix>
inline std::vector<int> denseSymbolicCholeskyColCounts(const SparseMatrix& permutedPattern) {
  const int n = static_cast<int>(permutedPattern.rows());
  std::vector<std::vector<bool>> nz(n, std::vector<bool>(n, false));
  for (int k = 0; k < permutedPattern.outerSize(); ++k) {
    for (typename SparseMatrix::InnerIterator it(permutedPattern, k); it; ++it) {
      nz[it.row()][it.col()] = true;
      nz[it.col()][it.row()] = true;
    }
  }
  for (int i = 0; i < n; ++i) nz[i][i] = true;

  // Symbolic symmetric Gaussian elimination: for pivot column j, any two
  // rows i, k > j both nonzero in column j become (symbolically) nonzero
  // in column j (already true) and fill in at (i,k)/(k,i) once eliminated.
  for (int j = 0; j < n; ++j) {
    std::vector<int> below;
    for (int i = j + 1; i < n; ++i) {
      if (nz[i][j]) below.push_back(i);
    }
    for (std::size_t a = 0; a < below.size(); ++a) {
      for (std::size_t b = a + 1; b < below.size(); ++b) {
        const int r = below[a];
        const int c = below[b];
        nz[r][c] = true;
        nz[c][r] = true;
      }
    }
  }

  std::vector<int> colCount(n, 0);
  for (int j = 0; j < n; ++j) {
    int cnt = 0;
    for (int i = j; i < n; ++i) {
      if (nz[i][j]) ++cnt;
    }
    colCount[j] = cnt;
  }
  return colCount;
}

// Random sparse *indefinite* symmetric matrix: starts from randomSparseSPD
// (diagonally dominant, so well-conditioned) and then flips the sign of a
// random subset of diagonal entries. Off-diagonal magnitudes are untouched,
// so each flipped row remains diagonally dominant (now around a negative
// center), giving a matrix with a generic mix of positive and negative
// eigenvalues (via Gershgorin) while staying safely away from exact
// singularity -- suitable for full-pipeline residual/inertia correctness
// tests (Phase 3) where the point is exercising indefinite threshold
// pivoting/delayed pivots end to end, not stress-testing near-singularity
// (that is test/stability's job, Phase 5).
inline Eigen::SparseMatrix<double, Eigen::ColMajor, int> randomSparseIndefinite(int n, int avgNnzPerRow,
                                                                                 unsigned seed) {
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A = randomSparseSPD(n, avgNnzPerRow, seed);
  std::mt19937 rng(seed ^ 0x9e3779b9u);
  std::uniform_int_distribution<int> coin(0, 1);
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, k); it; ++it) {
      if (it.row() == it.col() && coin(rng) == 0) {
        it.valueRef() = -it.value();
      }
    }
  }
  return A;
}

}  // namespace symla_test
