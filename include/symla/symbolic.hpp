#pragma once

// Supernode partition of the LDL^T factor, with Ashcraft-Grimes-style
// relaxed amalgamation, built on top of ordering.hpp + elimination_tree.hpp.
//
// Pipeline (SymbolicFactor::analyze):
//   1. compute a fill-reducing permutation (ordering.hpp: AMD / METIS / Natural)
//   2. permute the pattern and build its elimination tree (elimination_tree.hpp)
//   3. relabel columns into elimination-tree *postorder*. This is a standard
//      step (any postorder of a given etree yields an equivalent elimination
//      order with identical fill, see Liu 1990) and is required here so that
//      a supernode's member columns are always a *contiguous* range of
//      indices -- without it, "parent[j] == j+1" (the fundamental supernode
//      test) would almost never fire, since AMD/METIS output order has no
//      reason to place a node immediately after its only child.
//   4. re-build the elimination tree + column counts on the postordered
//      pattern (elimination_tree.hpp again), this time retaining full
//      per-column row patterns Struct(L*j).
//   5. fundamental supernodes: greedily merge column j into the same
//      supernode as j+1 iff parent[j] == j+1 && colCount[j] == colCount[j+1]+1.
//      This is the standard cardinality test (Liu/Ng/Peyton 1993): under a
//      postordered etree, equal-cardinality-plus-parent-child implies the
//      row patterns actually match (Struct(L*j)\{j} == Struct(L*(j+1))), so
//      the block can be stored as one exact dense trapezoid with zero extra
//      fill.
//   6. relaxed amalgamation: walk the (index-contiguous, by construction)
//      list of fundamental supernodes and merge supernode k into an
//      in-progress group with supernode k+1 when they are connected by an
//      etree parent-child edge (tree.parent[last col of k] == firstCol of
//      k+1) *and* the merge satisfies both a size cap (`max_relax_size`,
//      default 64 columns) and a relative "extra fill" cap
//      (`max_relax_fill_fraction`, default 0.25 of the merged block's dense
//      storage). This is a simplified version of Ashcraft-Grimes: the full
//      scheme in the literature also merges parent/child supernodes that are
//      *not* index-adjacent (absorbing a "gap" of intervening unrelated
//      columns as extra logical fill in the stored block); we deliberately
//      only merge index-adjacent etree-connected pairs, which keeps the
//      dense-trapezoid storage model exact and the implementation simple,
//      and covers the common/most impactful case (the last-visited child of
//      a node in postorder is always index-adjacent to its parent). This is
//      a documented scope reduction for Phase 1 -- revisit if profiling
//      shows non-adjacent merges matter.

#include "symla/elimination_tree.hpp"
#include "symla/ordering.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

namespace symla {

namespace detail {

// Postorder of a forest given by parent[] (parent[j] == -1 for roots).
// Children of each node are visited in increasing index order, so the
// result is deterministic. Returns `postNodes` of size n where postNodes[k]
// is the node visited k-th (i.e. this *is* a valid permutation of 0..n-1).
inline std::vector<int> postorder(const Eigen::VectorXi& parent) {
  const int n = static_cast<int>(parent.size());
  std::vector<std::vector<int>> children(n);
  std::vector<int> roots;
  for (int j = 0; j < n; ++j) {
    if (parent[j] == -1) {
      roots.push_back(j);
    } else {
      children[parent[j]].push_back(j);
    }
  }

  std::vector<int> order;
  order.reserve(n);
  std::function<void(int)> visit = [&](int node) {
    for (int c : children[node]) visit(c);
    order.push_back(node);
  };
  for (int r : roots) visit(r);
  return order;
}

// Applies permutation `perm` (perm[k] = original index placed at position k)
// to build the full symmetric pattern (no diagonal-only assumption; the
// diagonal is included for genericity) of A in the new index space.
template <typename SparseMatrix>
Eigen::SparseMatrix<double, Eigen::ColMajor, int> permutePatternSymmetric(const SparseMatrix& A,
                                                                           const Eigen::VectorXi& perm) {
  const int n = static_cast<int>(A.rows());
  Eigen::VectorXi newIndexOf(n);
  for (int k = 0; k < n; ++k) newIndexOf[perm[k]] = k;

  std::vector<int> colPtr, rowIdx;
  buildSymmetricPatternNoDiag(A, colPtr, rowIdx);

  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(colPtr[n]) + static_cast<std::size_t>(n));
  for (int c = 0; c < n; ++c) {
    const int nc = newIndexOf[c];
    for (int idx = colPtr[c]; idx < colPtr[c + 1]; ++idx) {
      const int r = rowIdx[idx];
      trips.emplace_back(newIndexOf[r], nc, 1.0);
    }
  }
  for (int k = 0; k < n; ++k) trips.emplace_back(k, k, 1.0);

  Eigen::SparseMatrix<double, Eigen::ColMajor, int> P(n, n);
  P.setFromTriplets(trips.begin(), trips.end());
  return P;
}

}  // namespace detail

struct Supernode {
  int firstCol = 0;
  int ncols = 0;
  // Row pattern of the supernode's first (representative) column, sorted
  // ascending; by construction its first `ncols` entries are exactly
  // firstCol, firstCol+1, ..., firstCol+ncols-1 (the supernode's own
  // columns), followed by the rows shared by all columns in the supernode
  // (the "update" rows passed to the parent front in later phases).
  std::vector<int> rowPattern;

  // Dense trapezoidal storage count: a lower-triangular ncols x ncols block
  // on top of a (rowPattern.size()-ncols) x ncols rectangular block below.
  // For a *fundamental* supernode this equals exactly the sum of colCounts
  // of its member columns (no extra fill); relaxed-amalgamated supernodes
  // may have storageNnz() > sum of member fundamental colCounts (documented
  // "logical" zero fill traded for larger BLAS-3 blocks).
  std::int64_t storageNnz() const {
    const std::int64_t m = static_cast<std::int64_t>(rowPattern.size());
    const std::int64_t k = ncols;
    return k * m - k * (k - 1) / 2;
  }
};

struct SymbolicFactorOptions {
  int max_relax_size = 64;
  double max_relax_fill_fraction = 0.25;
};

struct SymbolicFactor {
  // Permutation actually used for elimination: perm[k] = original matrix
  // index of the k-th pivot (includes both the fill-reducing ordering step
  // and the subsequent etree-postorder relabeling).
  Eigen::VectorXi perm;
  EliminationTree etree;             // built on the final (postordered) permuted pattern
  std::vector<Supernode> supernodes;  // fundamental supernodes after relaxed amalgamation
  std::vector<Supernode> fundamentalSupernodes;  // kept for testing/inspection

  template <typename SparseMatrix>
  static SymbolicFactor analyze(const SparseMatrix& A, OrderingType ordering,
                                 const SymbolicFactorOptions& options = SymbolicFactorOptions()) {
    const int n = static_cast<int>(A.rows());
    SymbolicFactor sf;
    if (n == 0) {
      sf.perm.resize(0);
      sf.etree.n = 0;
      return sf;
    }

    // --- Step 1-2: fill-reducing order, then permute the pattern. ---
    Eigen::VectorXi perm0 = computeOrdering(A, ordering);
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Aperm0 = detail::permutePatternSymmetric(A, perm0);

    // --- Step 3: build etree on the fill-reduced order, take its
    // postorder, and fold that into the final permutation. ---
    EliminationTree tree0 = EliminationTree::build(Aperm0);
    std::vector<int> postNodes = detail::postorder(tree0.parent);

    Eigen::VectorXi finalPerm(n);
    for (int k = 0; k < n; ++k) finalPerm[k] = perm0[postNodes[k]];

    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Afinal = detail::permutePatternSymmetric(A, finalPerm);

    // --- Step 4: rebuild etree + column counts + full row patterns on the
    // final postordered pattern. ---
    std::vector<std::vector<int>> rowPatterns;
    EliminationTree tree = EliminationTree::build(Afinal, &rowPatterns);

    sf.perm = finalPerm;
    sf.etree = tree;

    // --- Step 5: fundamental supernodes. ---
    std::vector<Supernode> funda;
    int j = 0;
    while (j < n) {
      const int start = j;
      while (j + 1 < n && tree.parent[j] == j + 1 && tree.colCount[j] == tree.colCount[j + 1] + 1) {
        ++j;
      }
      Supernode sn;
      sn.firstCol = start;
      sn.ncols = j - start + 1;
      sn.rowPattern = rowPatterns[start];
      funda.push_back(std::move(sn));
      ++j;
    }
    sf.fundamentalSupernodes = funda;

    // --- Step 6: relaxed amalgamation over the (index-contiguous) list of
    // fundamental supernodes. ---
    std::vector<Supernode> result;
    Supernode cur = funda[0];
    for (std::size_t k = 1; k < funda.size(); ++k) {
      const Supernode& next = funda[k];
      const bool etreeAdjacent = (tree.parent[cur.firstCol + cur.ncols - 1] == next.firstCol);
      bool merged = false;
      if (etreeAdjacent) {
        const int mergedNcols = cur.ncols + next.ncols;
        if (mergedNcols <= options.max_relax_size) {
          std::vector<int> mergedPattern;
          mergedPattern.reserve(cur.rowPattern.size() + next.rowPattern.size());
          std::set_union(cur.rowPattern.begin(), cur.rowPattern.end(), next.rowPattern.begin(),
                          next.rowPattern.end(), std::back_inserter(mergedPattern));

          Supernode mergedSn;
          mergedSn.firstCol = cur.firstCol;
          mergedSn.ncols = mergedNcols;
          mergedSn.rowPattern = mergedPattern;

          const std::int64_t mergedStorage = mergedSn.storageNnz();
          const std::int64_t origStorage = cur.storageNnz() + next.storageNnz();
          const double extraFrac =
              static_cast<double>(mergedStorage - origStorage) / static_cast<double>(mergedStorage);
          if (extraFrac <= options.max_relax_fill_fraction) {
            cur = std::move(mergedSn);
            merged = true;
          }
        }
      }
      if (!merged) {
        result.push_back(std::move(cur));
        cur = next;
      }
    }
    result.push_back(std::move(cur));
    sf.supernodes = std::move(result);

    return sf;
  }
};

}  // namespace symla
