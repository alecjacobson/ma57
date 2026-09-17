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
//   6. relaxed amalgamation, general (non-adjacent) Ashcraft-Grimes form:
//      build the *supernode tree* over fundamental supernodes (supernode i's
//      parent is the fundamental supernode containing etree.parent[i's last
//      column]) and, processing fundamental supernodes in increasing index
//      order (a valid bottom-up/postorder traversal of that tree, since
//      etree.parent[j] > j always), try absorbing *every* child of the
//      current node into it -- not just an index-adjacent one -- subject to
//      a size cap (`max_relax_size`, default 64 columns) and a relative
//      "extra fill" cap (`max_relax_fill_fraction`, default 0.25 of the
//      merged block's dense storage), same criteria as before but evaluated
//      against the *general* supernode-tree parent/child relation via a
//      union-find over merge groups. This fixes a real defect in the
//      earlier index-adjacent-only scheme: under postorder, only the
//      *last-visited* child of a node ends up index-adjacent to it (postorder
//      places a node's last child immediately before the node itself); every
//      other child is separated from the parent by other subtrees'
//      columns. On branchy etrees (typical of 3D PDE stencils/meshes) this
//      meant almost no merging ever fired, since only one child per parent
//      was ever eligible. General (index-independent) merging fixes this,
//      but merged groups are no longer contiguous in the postorder used to
//      build the tree, so after merge decisions are made we perform a
//      *second* relabeling pass (mirroring step 3's postorder-for-
//      contiguity trick, applied here to the coarser *merge-group* forest
//      instead of the raw etree): each merge group is emitted as a
//      contiguous block of columns (its own columns last, preceded by its
//      absorbed children's columns in their own already-valid internal
//      order), and merge groups themselves are emitted in a postorder of
//      the merge-group forest, giving a new global permutation under which
//      every final supernode's columns are contiguous (preserving the
//      "first ncols entries of rowPattern are the supernode's own
//      contiguous columns" invariant multifrontal.hpp depends on). The
//      pattern/etree/column-counts are then rebuilt once more on this final
//      permutation (this is exact, not a heuristic: any postorder of a
//      fixed etree yields identical fill, Liu 1990, so the fill-fraction
//      decisions made against the pre-relabeling patterns are consistent
//      with what the rebuilt tree will show), and each final supernode's
//      stored row pattern is computed as the union of its member columns'
//      Struct(L*c) in the final index space (safe and general -- doesn't
//      rely on the nesting property fundamental supernodes get for free).

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
  // Kept at the Phase 1 default (0.25), *not* raised, despite evidence it
  // is conservative for some matrices -- see the long comment block above
  // for the measured tradeoff data. Measured on GHS_indef/bratu3d (a
  // genuinely 3D PDE problem, n=27792): raising this to 0.5 does improve
  // the average supernode size a lot (1.65 -> 25.7 cols, at a 1.32x vs
  // 1.05x storage-nnz cost) but was also observed to make `factorize()`
  // *slower* in wall-clock terms on this specific matrix (>700s vs 314.9s
  // at 0.25, both single-threaded) -- plausibly because the general
  // (non-nested-branch) merges introduce genuine "logical fill" zero
  // entries into a front's dense block, and on an already-hard indefinite
  // problem like bratu3d (which already has ~21% of columns needing
  // delayed pivoting even at 0.25) more of those structural zeros inside
  // bigger blocks may be triggering more delayed-pivot cascades that
  // inflate ancestor front sizes further up the tree, offsetting the
  // BLAS-3 blocking win. This was not fully root-caused (see the Phase
  // 1.1 report / Phase 6 handoff notes) so the default is left
  // unchanged; callers whose matrices are known to behave well under more
  // aggressive amalgamation (most non-pathological cases: stokes128,
  // aug3dcqp, helm2d03 all improve markedly here with no timing
  // regression observed) can raise `max_relax_fill_fraction` explicitly.
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
    auto computeFundamentalSupernodes = [](const EliminationTree& t,
                                            const std::vector<std::vector<int>>& patterns) {
      std::vector<Supernode> out;
      const int nn = t.n;
      int j = 0;
      while (j < nn) {
        const int start = j;
        while (j + 1 < nn && t.parent[j] == j + 1 && t.colCount[j] == t.colCount[j + 1] + 1) {
          ++j;
        }
        Supernode sn;
        sn.firstCol = start;
        sn.ncols = j - start + 1;
        sn.rowPattern = patterns[start];
        out.push_back(std::move(sn));
        ++j;
      }
      return out;
    };

    std::vector<Supernode> funda = computeFundamentalSupernodes(tree, rowPatterns);
    const int numFunda = static_cast<int>(funda.size());

    // --- Step 6a: build the supernode tree over fundamental supernodes. ---
    std::vector<int> colToFunda(n, -1);
    for (int i = 0; i < numFunda; ++i) {
      for (int c = funda[i].firstCol; c < funda[i].firstCol + funda[i].ncols; ++c) colToFunda[c] = i;
    }
    std::vector<int> parentFunda(numFunda, -1);
    std::vector<std::vector<int>> childrenFunda(numFunda);
    for (int i = 0; i < numFunda; ++i) {
      const int lastCol = funda[i].firstCol + funda[i].ncols - 1;
      const int p = tree.parent[lastCol];
      if (p != -1) {
        const int pf = colToFunda[p];
        parentFunda[i] = pf;
        childrenFunda[pf].push_back(i);
      }
    }

    // --- Step 6b: general relaxed amalgamation via union-find over the
    // supernode tree, processing fundamental supernodes bottom-up
    // (increasing index is a valid postorder of this tree: parentFunda[i]
    // is always > i, since it derives from tree.parent[lastCol] > lastCol
    // >= i). For each node, try absorbing *every* child (not just an
    // index-adjacent one) subject to the size/fill caps. ---
    std::vector<int> dsu(numFunda);
    for (int i = 0; i < numFunda; ++i) dsu[i] = i;
    auto find = [&](int x) {
      while (dsu[x] != x) {
        dsu[x] = dsu[dsu[x]];
        x = dsu[x];
      }
      return x;
    };

    struct Agg {
      int ncols = 0;
      std::vector<int> rowPattern;  // stage-1 (pre-relabel) index space; merge decisions only
      std::vector<int> segments;    // ordered list of fundamental-supernode indices, own last
    };
    std::vector<Agg> agg(numFunda);
    for (int i = 0; i < numFunda; ++i) {
      agg[i].ncols = funda[i].ncols;
      agg[i].rowPattern = funda[i].rowPattern;
      agg[i].segments = {i};
    }

    auto storageNnzOf = [](int ncols, std::size_t m) -> std::int64_t {
      const std::int64_t k = ncols;
      const std::int64_t mm = static_cast<std::int64_t>(m);
      return k * mm - k * (k - 1) / 2;
    };

    for (int i = 0; i < numFunda; ++i) {
      for (int c : childrenFunda[i]) {
        const int cr = find(c);
        if (cr == i) continue;
        const int mergedNcols = agg[i].ncols + agg[cr].ncols;
        if (mergedNcols > options.max_relax_size) continue;

        std::vector<int> mergedPattern;
        mergedPattern.reserve(agg[i].rowPattern.size() + agg[cr].rowPattern.size());
        std::set_union(agg[i].rowPattern.begin(), agg[i].rowPattern.end(), agg[cr].rowPattern.begin(),
                        agg[cr].rowPattern.end(), std::back_inserter(mergedPattern));

        const std::int64_t mergedStorage = storageNnzOf(mergedNcols, mergedPattern.size());
        const std::int64_t origStorage = storageNnzOf(agg[i].ncols, agg[i].rowPattern.size()) +
                                          storageNnzOf(agg[cr].ncols, agg[cr].rowPattern.size());
        const double extraFrac =
            static_cast<double>(mergedStorage - origStorage) / static_cast<double>(mergedStorage);
        if (extraFrac > options.max_relax_fill_fraction) continue;

        // Accept: absorb cr's group into i's group. cr's (already validly
        // ordered) segments go before i's own trailing "own" entry.
        dsu[cr] = i;
        agg[i].ncols = mergedNcols;
        agg[i].rowPattern = std::move(mergedPattern);
        agg[i].segments.insert(agg[i].segments.end() - 1, agg[cr].segments.begin(), agg[cr].segments.end());
      }
    }

    // --- Step 6c: final merge groups = union-find roots. Build the
    // (coarser) merge-group forest and take its postorder, so groups are
    // emitted with descendants before ancestors -- required for the
    // relabeling below to remain a valid elimination order. ---
    std::vector<int> finalRoots;
    for (int i = 0; i < numFunda; ++i) {
      if (find(i) == i) finalRoots.push_back(i);
    }
    const int numFinal = static_cast<int>(finalRoots.size());
    std::vector<int> compactOf(numFunda, -1);
    for (int k = 0; k < numFinal; ++k) compactOf[finalRoots[k]] = k;

    Eigen::VectorXi finalParentCompact(numFinal);
    for (int k = 0; k < numFinal; ++k) {
      const int r = finalRoots[k];
      const int pf = parentFunda[r];
      finalParentCompact[k] = (pf == -1) ? -1 : compactOf[find(pf)];
    }
    std::vector<int> groupPostorder = detail::postorder(finalParentCompact);

    // --- Step 6d: emit the new column relabeling + provisional supernode
    // boundaries (firstCol/ncols) in this group postorder. ---
    std::vector<Supernode> finalSupernodes;
    finalSupernodes.reserve(numFinal);
    std::vector<int> newOrder;  // stage-1 index space, size n
    newOrder.reserve(n);
    for (int gk : groupPostorder) {
      const int r = finalRoots[gk];
      Supernode sn;
      sn.firstCol = static_cast<int>(newOrder.size());
      sn.ncols = agg[r].ncols;
      for (int fi : agg[r].segments) {
        for (int c = funda[fi].firstCol; c < funda[fi].firstCol + funda[fi].ncols; ++c) newOrder.push_back(c);
      }
      finalSupernodes.push_back(std::move(sn));
    }

    // --- Step 6e: compose the relabeling into the final permutation and
    // rebuild the pattern/etree/column-counts on it (exact, not heuristic:
    // newOrder is a valid postorder of the same etree, so fill is
    // unchanged -- see the header comment). ---
    Eigen::VectorXi combinedPerm(n);
    for (int k = 0; k < n; ++k) combinedPerm[k] = finalPerm[newOrder[k]];

    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Afinal2 = detail::permutePatternSymmetric(A, combinedPerm);
    std::vector<std::vector<int>> rowPatterns2;
    EliminationTree tree2 = EliminationTree::build(Afinal2, &rowPatterns2);

    sf.perm = combinedPerm;
    sf.etree = tree2;
    sf.fundamentalSupernodes = computeFundamentalSupernodes(tree2, rowPatterns2);

    // --- Step 6f: fill in each final supernode's row pattern as the union
    // of its member columns' Struct(L*c) in the final index space. ---
    for (auto& sn : finalSupernodes) {
      std::vector<int> pattern;
      for (int c = sn.firstCol; c < sn.firstCol + sn.ncols; ++c) {
        std::vector<int> merged;
        merged.reserve(pattern.size() + rowPatterns2[c].size());
        std::set_union(pattern.begin(), pattern.end(), rowPatterns2[c].begin(), rowPatterns2[c].end(),
                        std::back_inserter(merged));
        pattern.swap(merged);
      }
      sn.rowPattern = std::move(pattern);
    }
    sf.supernodes = std::move(finalSupernodes);

    return sf;
  }
};

}  // namespace symla
