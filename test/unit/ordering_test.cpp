// Unit tests for symla::computeOrdering (ordering.hpp): checks the result is
// a valid permutation for AMD / METIS (if available) / Natural, and does a
// rough fill-reduction sanity check (AMD fill should be dramatically lower
// than a deliberately bad natural order on a "star" graph).
#include "symla/ordering.hpp"
#include "symla/elimination_tree.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <numeric>
#include <vector>

using symla::OrderingType;

namespace {

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

Eigen::SparseMatrix<double, Eigen::ColMajor, int> tridiagonal(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, 4.0);
  for (int i = 0; i + 1 < n; ++i) {
    trips.emplace_back(i, i + 1, -1.0);
    trips.emplace_back(i + 1, i, -1.0);
  }
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

Eigen::SparseMatrix<double, Eigen::ColMajor, int> arrow(int n) {
  // Hub node 0 connected to every other node; only upper triangle stored,
  // to also exercise the "single triangle stored" input assumption.
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, static_cast<double>(n + 1));
  for (int i = 1; i < n; ++i) trips.emplace_back(0, i, 1.0);  // upper triangle only
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// hub-first star graph: node 0 connected to all others, stored as full
// symmetric pattern (both triangles).
Eigen::SparseMatrix<double, Eigen::ColMajor, int> starHubFirst(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, static_cast<double>(n + 1));
  for (int i = 1; i < n; ++i) {
    trips.emplace_back(0, i, 1.0);
    trips.emplace_back(i, 0, 1.0);
  }
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

long totalFill(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A, const Eigen::VectorXi& perm) {
  // Build permuted pattern P A P^T (structurally) and compute etree colCounts.
  const int n = static_cast<int>(A.rows());
  Eigen::VectorXi newIndexOf(n);
  for (int k = 0; k < n; ++k) newIndexOf[perm[k]] = k;
  std::vector<Eigen::Triplet<double>> trips;
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, k); it; ++it) {
      trips.emplace_back(newIndexOf[it.row()], newIndexOf[it.col()], 1.0);
    }
  }
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> P(n, n);
  P.setFromTriplets(trips.begin(), trips.end());
  auto tree = symla::EliminationTree::build(P);
  return tree.colCount.sum();
}

}  // namespace

int main() {
  // --- Validity checks on small hand-built matrices ---
  for (int n : {1, 2, 5, 10}) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> T = tridiagonal(n);
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> S = arrow(n);

    for (auto type : {OrderingType::Natural, OrderingType::AMD}) {
      SYMLA_CHECK(isValidPermutation(symla::computeOrdering(T, type), n));
      SYMLA_CHECK(isValidPermutation(symla::computeOrdering(S, type), n));
    }
#ifdef SYMLA_HAVE_METIS
    if (n >= 2) {
      SYMLA_CHECK(isValidPermutation(symla::computeOrdering(T, OrderingType::Metis), n));
      SYMLA_CHECK(isValidPermutation(symla::computeOrdering(S, OrderingType::Metis), n));
    }
#endif
  }

  // Natural ordering must be the identity.
  {
    Eigen::VectorXi p = symla::computeOrdering(tridiagonal(7), OrderingType::Natural);
    for (int i = 0; i < 7; ++i) SYMLA_CHECK(p[i] == i);
  }

  // --- Fill-reduction sanity check on a star graph ---
  {
    const int n = 60;
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> star = starHubFirst(n);

    Eigen::VectorXi natural(n);
    std::iota(natural.begin(), natural.end(), 0);
    const long naturalFill = totalFill(star, natural);

    Eigen::VectorXi amdPerm = symla::computeOrdering(star, OrderingType::AMD);
    SYMLA_CHECK(isValidPermutation(amdPerm, n));
    const long amdFill = totalFill(star, amdPerm);

    // Natural (hub-first) order fully fills the (n-1)x(n-1) trailing block;
    // AMD should recognize the hub should go last, giving near-zero fill.
    SYMLA_CHECK(naturalFill > amdFill * 4);

#ifdef SYMLA_HAVE_METIS
    Eigen::VectorXi metisPerm = symla::computeOrdering(star, OrderingType::Metis);
    SYMLA_CHECK(isValidPermutation(metisPerm, n));
    const long metisFill = totalFill(star, metisPerm);
    SYMLA_CHECK(naturalFill > metisFill * 4);
#endif
  }

  std::cout << "symla ordering_test OK\n";
  return 0;
}
