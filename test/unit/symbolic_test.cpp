// Unit tests for symla::SymbolicFactor (symbolic.hpp): supernode partition
// correctness (permutation validity, fundamental supernode nnz exactly
// equal to the sum of etree column counts, relaxed-amalgamated supernode
// storage only ever >= the fundamental total), on hand-built, random small,
// and one larger (200-1000 node) random sparse SPD matrix.
#include "symla/symbolic.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymbolicFactor;
using symla::SymbolicFactorOptions;

namespace {

Sparse tridiagonal(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, 4.0);
  for (int i = 0; i + 1 < n; ++i) {
    trips.emplace_back(i, i + 1, -1.0);
    trips.emplace_back(i + 1, i, -1.0);
  }
  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

bool isValidPermutation(const Eigen::VectorXi& perm, int n) {
  if (perm.size() != n) return false;
  std::vector<bool> seen(n, false);
  for (int k = 0; k < n; ++k) {
    const int v = perm[k];
    if (v < 0 || v >= n) return false;
    if (seen[v]) return false;
    seen[v] = true;
  }
  return true;
}

void checkSymbolicFactor(const Sparse& A, OrderingType type, const SymbolicFactorOptions& opts) {
  const int n = static_cast<int>(A.rows());
  SymbolicFactor sf = SymbolicFactor::analyze(A, type, opts);

  SYMLA_CHECK(isValidPermutation(sf.perm, n));
  SYMLA_CHECK(sf.etree.n == n);

  const std::int64_t totalColCount = sf.etree.colCount.template cast<std::int64_t>().sum();

  // Fundamental supernodes: cover 0..n-1 exactly once, contiguous, and the
  // sum of their (exact, zero-extra-fill) storageNnz() must equal the total
  // column-count sum exactly.
  std::int64_t fundaTotal = 0;
  int expectedNext = 0;
  for (const auto& sn : sf.fundamentalSupernodes) {
    SYMLA_CHECK(sn.firstCol == expectedNext);
    SYMLA_CHECK(sn.ncols > 0);
    SYMLA_CHECK(static_cast<int>(sn.rowPattern.size()) >= sn.ncols);
    for (int k = 0; k < sn.ncols; ++k) SYMLA_CHECK(sn.rowPattern[k] == sn.firstCol + k);
    fundaTotal += sn.storageNnz();
    expectedNext = sn.firstCol + sn.ncols;
  }
  SYMLA_CHECK(expectedNext == n);
  SYMLA_CHECK(fundaTotal == totalColCount);

  std::vector<int> fundaFirstCols;
  for (const auto& sn : sf.fundamentalSupernodes) fundaFirstCols.push_back(sn.firstCol);

  // Relaxed-amalgamated supernodes: also cover 0..n-1 exactly once,
  // contiguous, and total storage only ever >= fundamental. Note:
  // max_relax_size only bounds *relaxed-amalgamation merges themselves*
  // (cur.ncols + next.ncols <= max_relax_size at merge time) -- a
  // fundamental supernode can legitimately already be larger than
  // max_relax_size on its own (it has zero extra fill either way, so there
  // is no reason to cap it), so we must not assert sn.ncols <=
  // max_relax_size as a blanket invariant here.
  std::int64_t relaxedTotal = 0;
  expectedNext = 0;
  for (const auto& sn : sf.supernodes) {
    SYMLA_CHECK(sn.firstCol == expectedNext);
    SYMLA_CHECK(sn.ncols > 0);
    SYMLA_CHECK(static_cast<int>(sn.rowPattern.size()) >= sn.ncols);
    for (int k = 0; k < sn.ncols; ++k) SYMLA_CHECK(sn.rowPattern[k] == sn.firstCol + k);
    relaxedTotal += sn.storageNnz();
    expectedNext = sn.firstCol + sn.ncols;

    // If this supernode spans more than one fundamental supernode (i.e. it
    // is the product of one or more relaxed-amalgamation merges), the
    // merge-time size cap must actually have been respected.
    const int nFundaCovered = static_cast<int>(
        std::count_if(fundaFirstCols.begin(), fundaFirstCols.end(),
                      [&](int fc) { return fc >= sn.firstCol && fc < sn.firstCol + sn.ncols; }));
    if (nFundaCovered > 1) {
      SYMLA_CHECK(sn.ncols <= opts.max_relax_size);
    }
  }
  SYMLA_CHECK(expectedNext == n);
  SYMLA_CHECK(relaxedTotal >= fundaTotal);
}

}  // namespace

int main() {
  SymbolicFactorOptions defaultOpts;

  // --- Hand-built: tridiagonal (no fill; every supernode should end up
  // as single columns or small chains with zero extra fill either way). ---
  for (int n : {1, 2, 5, 20}) {
    checkSymbolicFactor(tridiagonal(n), OrderingType::Natural, defaultOpts);
    checkSymbolicFactor(tridiagonal(n), OrderingType::AMD, defaultOpts);
  }

  // --- Random small SPD matrices, AMD + Natural ordering. ---
  {
    unsigned seed = 777;
    for (int n : {5, 10, 25, 60}) {
      for (int trial = 0; trial < 3; ++trial, ++seed) {
        Sparse A = symla_test::randomSparseSPD(n, /*avgNnzPerRow=*/4, seed);
        checkSymbolicFactor(A, OrderingType::Natural, defaultOpts);
        checkSymbolicFactor(A, OrderingType::AMD, defaultOpts);
#ifdef SYMLA_HAVE_METIS
        checkSymbolicFactor(A, OrderingType::Metis, defaultOpts);
#endif
      }
    }
  }

  // --- Small relaxation-parameter sweep to exercise both very tight (no
  // extra merges beyond fundamental) and generous relaxation. ---
  {
    Sparse A = symla_test::randomSparseSPD(80, 5, 999);
    SymbolicFactorOptions tight;
    tight.max_relax_size = 1;  // effectively disables relaxed amalgamation
    tight.max_relax_fill_fraction = 0.0;
    checkSymbolicFactor(A, OrderingType::AMD, tight);

    SymbolicFactorOptions loose;
    loose.max_relax_size = 128;
    loose.max_relax_fill_fraction = 1.0;
    checkSymbolicFactor(A, OrderingType::AMD, loose);
  }

  // --- One larger (200-1000 node) random sparse SPD matrix. ---
  {
    Sparse A = symla_test::randomSparseSPD(500, /*avgNnzPerRow=*/6, /*seed=*/2024);
    checkSymbolicFactor(A, OrderingType::AMD, defaultOpts);
#ifdef SYMLA_HAVE_METIS
    checkSymbolicFactor(A, OrderingType::Metis, defaultOpts);
#endif
  }

  std::cout << "symla symbolic_test OK\n";
  return 0;
}
