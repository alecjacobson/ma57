// Unit tests for symla::EliminationTree (elimination_tree.hpp):
//  - hand-verified etree shape on a tridiagonal matrix (path/chain etree)
//    and an arrow matrix with the dense row/col last (star/broom etree),
//    both under the natural (identity) ordering.
//  - column counts cross-checked against an independent brute-force dense
//    symbolic-Cholesky oracle (test_helpers.hpp) on random small SPD
//    matrices.
#include "symla/elimination_tree.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

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

// Hub node n-1 connected to every other node 0..n-2 (dense last row/col);
// with the natural (identity) elimination order this is the textbook
// example that produces a star/"broom" elimination tree: every leaf's
// parent is the hub, and eliminating any leaf causes no fill at all.
Sparse arrowHubLast(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, static_cast<double>(n + 1));
  for (int i = 0; i + 1 < n; ++i) {
    trips.emplace_back(i, n - 1, 1.0);
    trips.emplace_back(n - 1, i, 1.0);
  }
  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

}  // namespace

int main() {
  // --- Tridiagonal -> path/chain etree: parent[j] == j+1 for j < n-1, root n-1. ---
  {
    const int n = 5;
    auto tree = symla::EliminationTree::build(tridiagonal(n));
    SYMLA_CHECK(tree.n == n);
    for (int j = 0; j < n - 1; ++j) SYMLA_CHECK(tree.parent[j] == j + 1);
    SYMLA_CHECK(tree.parent[n - 1] == -1);
    // Tridiagonal has no fill at all: colCount[j] should be 2 for interior
    // columns (self + one subdiagonal neighbor) and 1 for the last column.
    for (int j = 0; j < n - 1; ++j) SYMLA_CHECK(tree.colCount[j] == 2);
    SYMLA_CHECK(tree.colCount[n - 1] == 1);
  }

  // --- Arrow (hub last) -> star/broom etree: parent[j] == n-1 for all j < n-1. ---
  {
    const int n = 5;
    auto tree = symla::EliminationTree::build(arrowHubLast(n));
    SYMLA_CHECK(tree.n == n);
    for (int j = 0; j < n - 1; ++j) SYMLA_CHECK(tree.parent[j] == n - 1);
    SYMLA_CHECK(tree.parent[n - 1] == -1);
    // No fill: each leaf column j<n-1 touches only {j, n-1} => colCount 2.
    // The hub is the *last* pivot, so Struct(L*(n-1)) has nothing below the
    // diagonal left to eliminate => colCount 1 (LDL^T column counts only
    // count rows *below* the diagonal entry, not the whole dense row/col).
    for (int j = 0; j < n - 1; ++j) SYMLA_CHECK(tree.colCount[j] == 2);
    SYMLA_CHECK(tree.colCount[n - 1] == 1);
  }

  // --- Cross-check colCount against a brute-force dense oracle on random SPD matrices. ---
  {
    unsigned seed = 12345;
    for (int n : {5, 10, 25, 40}) {
      for (int trial = 0; trial < 3; ++trial, ++seed) {
        Sparse A = symla_test::randomSparseSPD(n, /*avgNnzPerRow=*/3, seed);
        auto tree = symla::EliminationTree::build(A);
        std::vector<int> oracle = symla_test::denseSymbolicCholeskyColCounts(A);
        SYMLA_CHECK(static_cast<int>(oracle.size()) == n);
        for (int j = 0; j < n; ++j) {
          SYMLA_CHECK(tree.colCount[j] == oracle[j]);
        }
      }
    }
  }

  // --- n == 0 and n == 1 edge cases. ---
  {
    Sparse empty(0, 0);
    auto tree0 = symla::EliminationTree::build(empty);
    SYMLA_CHECK(tree0.n == 0);

    Sparse single(1, 1);
    single.insert(0, 0) = 1.0;
    single.makeCompressed();
    auto tree1 = symla::EliminationTree::build(single);
    SYMLA_CHECK(tree1.n == 1);
    SYMLA_CHECK(tree1.parent[0] == -1);
    SYMLA_CHECK(tree1.colCount[0] == 1);
  }

  std::cout << "symla elimination_tree_test OK\n";
  return 0;
}
