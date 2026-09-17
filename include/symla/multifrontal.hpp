#pragma once

// Phase 3: multifrontal numeric factorization driver.
//
// Given a Phase 1 `SymbolicFactor` (ordering + etree + supernode partition,
// all in "final order" -- i.e. index space after applying `SymbolicFactor::
// perm`, see symbolic.hpp) and the numeric values of A, this computes the
// LDL^T factorization by:
//
//   1. deriving the *supernode tree* (a coarsening of the elimination tree:
//      supernode S's parent is the supernode containing etree.parent[S's
//      last column]) from the already-computed etree + supernode partition,
//   2. visiting supernodes in increasing index order (which -- because
//      symbolic.hpp postorders columns by etree postorder and supernodes
//      are contiguous ranges of that postorder -- is already a valid
//      postorder of the supernode tree: parent index is always > every
//      descendant's index, so processing 0..N-1 in order visits children
//      strictly before their parent),
//   3. for each supernode, assembling a dense frontal matrix via
//      extend-add (original A entries for its own columns + each child's
//      generated element / update matrix, index-mapped via an O(1)-lookup
//      scratch array rather than a per-entry search),
//   4. factoring the front with Phase 2's `DenseLDLT::factor`, restricting
//      pivot selection to the front's *eligible* (fully-summed) columns --
//      the supernode's own columns plus any columns delayed into it by its
//      children -- via the `n_eligible` parameter added to `DenseLDLT::
//      factor` for this purpose (see dense_kernel.hpp),
//   5. forwarding anything left over (ancestor "not yet fully summed" rows,
//      plus any of this front's own eligible columns that still failed to
//      pivot -- newly delayed) to the parent supernode as *this* front's
//      generated element,
//   6. accumulating the successfully factored pivots' L/D data into a flat
//      `NumericFactor`, and the front's Schur complement into whatever
//      structure receives it next (parent, or -- if there is no parent --
//      permanently un-pivotable, i.e. the system is numerically singular).
//
// This header only performs single-threaded factorization (Phase 6 adds
// task-DAG parallelism over this same tree later); it does not implement
// solve (Phase 4).

#include "symla/dense_kernel.hpp"
#include "symla/solver.hpp"
#include "symla/symbolic.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <vector>

namespace symla {

// One frontal matrix's finalized numeric contribution, stored in a form
// close to what Phase 4's blocked forward/diag/back substitution wants:
// a single rectangular multiplier block (own pivots' unit-lower-triangular
// L plus the multipliers applied to the rows that were *not* finalized
// here) together with the small block-diagonal D for the pivots finalized
// at this front.
//
// Row/column identity is tracked via `rowIndices`, indices into the
// *final* (post `SymbolicFactor::perm`) pivot-order index space used
// throughout Phase 1 -- i.e. to recover the position in the *original*
// user matrix, look up `SymbolicFactor::perm[rowIndices[i]]`.
//
//   rowIndices[0 .. nPivots-1]      == finalized pivot columns, in local
//                                      factorization order (the physical
//                                      column order inside this front after
//                                      all Bunch-Kaufman swaps -- NOT
//                                      necessarily sorted, and NOT
//                                      necessarily equal to this
//                                      supernode's own [firstCol,
//                                      firstCol+ncols) range, since delayed
//                                      pivots forwarded up from children are
//                                      eligible here too and may finalize
//                                      before/after this supernode's
//                                      "native" columns).
//   rowIndices[nPivots .. m-1]      == rows forwarded to the parent as this
//                                      front's generated element (a mix of
//                                      genuine ancestor "not fully summed"
//                                      rows and any of this front's own
//                                      eligible columns that *still*
//                                      couldn't be pivoted -- i.e. newly
//                                      delayed columns).
//
// L is m x nPivots: L.topRows(nPivots) is unit-lower-triangular (diagonal
// implied 1, not stored) with the *same* "(k+1,k) slot inside a 2x2 pivot
// block holds D's d21, not an L multiplier" caveat as DenseLDLTResult (see
// dense_kernel.hpp); consult `pivotBlocks` to know which slots to skip.
// L.bottomRows(m - nPivots) is the rectangular multiplier block applied to
// the rows forwarded to the parent.
struct SupernodeFactor {
  std::vector<int> rowIndices;          // size m, final-order indices
  int nPivots = 0;                      // number of finalized pivots (== n_factored at this front)
  Eigen::MatrixXd L;                    // m x nPivots
  Eigen::MatrixXd D;                    // nPivots x nPivots, block-diagonal (see DenseLDLTResult::D_out)
  std::vector<PivotBlock> pivotBlocks;  // local (0..nPivots-1) pivot structure
};

// Complete numeric factorization, front-by-front, in processing
// (postorder-over-the-supernode-tree) order. This is mathematically
// equivalent to one big sparse LDL^T of P A P^T (P = permutation given by
// `perm`); the per-front layout exists so Phase 4 can do a blocked
// multifrontal-style forward/diag/back solve (process `fronts` in this same
// order for the forward solve, in reverse for the back solve) rather than a
// single flat triangular solve, but a flat solve built by concatenating all
// fronts' pivot columns in this same order would also be numerically valid
// (front order is a valid elimination order: every row forwarded to a
// parent has strictly larger final-order index than every pivot finalized
// at any of its descendants, by construction of the elimination tree).
struct NumericFactor {
  std::vector<SupernodeFactor> fronts;  // in processing order
  Eigen::VectorXi perm;                 // copy of SymbolicFactor::perm (final-order -> original index)
  Inertia inertia;
  bool singular = false;
  std::vector<int> singularCols;  // final-order indices that could never be
                                   // pivoted anywhere, including at the root
                                   // (genuine numerical rank deficiency)
  int n = 0;
};

namespace detail {

// Full symmetric permuted values, restricted to the lower triangle (row >=
// col) in the *final* index space, stored as a plain CSC-like structure
// (sorted rows within each column). Only the lower triangle of the
// *original* A is read (row_orig >= col_orig); if A stores both triangles
// with consistent (symmetric) values this is exactly equivalent to reading
// the whole matrix, if it stores only the lower triangle it is also
// correct, but a matrix that stores *only* its upper triangle would be
// silently read as all-zero -- documented as the same "lower triangle is
// authoritative" convention Eigen's own SimplicialLLT<..., Lower> uses.
template <typename Scalar, typename SparseMatrix>
struct PermutedLower {
  std::vector<std::vector<int>> rows;       // rows[c] sorted ascending, all >= c
  std::vector<std::vector<Scalar>> vals;    // vals[c][k] matches rows[c][k]

  static PermutedLower build(const SparseMatrix& A, const Eigen::VectorXi& perm) {
    const int n = static_cast<int>(A.rows());
    Eigen::VectorXi newIndexOf(n);
    for (int k = 0; k < n; ++k) newIndexOf[perm[k]] = k;

    PermutedLower out;
    out.rows.resize(n);
    out.vals.resize(n);
    for (int c = 0; c < A.outerSize(); ++c) {
      for (typename SparseMatrix::InnerIterator it(A, c); it; ++it) {
        const int r = static_cast<int>(it.row());
        const int cc = static_cast<int>(it.col());
        if (r < cc) continue;  // only the original matrix's lower triangle is authoritative
        const int nr = newIndexOf[r];
        const int nc = newIndexOf[cc];
        const int lo = std::min(nr, nc);
        const int hi = std::max(nr, nc);
        out.rows[lo].push_back(hi);
        out.vals[lo].push_back(static_cast<Scalar>(it.value()));
      }
    }
    for (int c = 0; c < n; ++c) {
      // sort by row for determinism (not required for correctness -- the
      // assembly step below is an O(1)-lookup scatter, not a merge).
      std::vector<int> order(out.rows[c].size());
      for (std::size_t k = 0; k < order.size(); ++k) order[k] = static_cast<int>(k);
      std::sort(order.begin(), order.end(), [&](int a, int b) { return out.rows[c][a] < out.rows[c][b]; });
      std::vector<int> newRows(order.size());
      std::vector<Scalar> newVals(order.size());
      for (std::size_t k = 0; k < order.size(); ++k) {
        newRows[k] = out.rows[c][order[k]];
        newVals[k] = out.vals[c][order[k]];
      }
      out.rows[c].swap(newRows);
      out.vals[c].swap(newVals);
    }
    return out;
  }
};

}  // namespace detail

template <typename Scalar>
class MultifrontalFactorizer {
 public:
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

  static NumericFactor factorize(const SymbolicFactor& sf, const SparseMatrix& A,
                                  const DenseLDLTOptions& options = {}) {
    const int n = sf.etree.n;
    NumericFactor nf;
    nf.perm = sf.perm;
    nf.n = n;
    if (n == 0) return nf;

    const auto& supernodes = sf.supernodes;
    const int numSN = static_cast<int>(supernodes.size());

    // --- Derive the supernode tree: parentSN[si] = supernode containing
    // etree.parent[last column of si], or -1 if si is a root. ---
    std::vector<int> colToSupernode(n, -1);
    for (int si = 0; si < numSN; ++si) {
      const auto& sn = supernodes[si];
      for (int c = sn.firstCol; c < sn.firstCol + sn.ncols; ++c) colToSupernode[c] = si;
    }
    std::vector<int> parentSN(numSN, -1);
    std::vector<std::vector<int>> childrenSN(numSN);
    for (int si = 0; si < numSN; ++si) {
      const auto& sn = supernodes[si];
      const int lastCol = sn.firstCol + sn.ncols - 1;
      const int p = sf.etree.parent[lastCol];
      if (p != -1) {
        const int psn = colToSupernode[p];
        parentSN[si] = psn;
        childrenSN[psn].push_back(si);
      }
    }

    // --- Numeric values of A, permuted into final order, lower triangle only. ---
    auto Aperm = detail::PermutedLower<Scalar, SparseMatrix>::build(A, sf.perm);

    // Generated element ("update matrix") produced by each supernode after
    // its own processing, consumed exactly once by its parent (freed
    // thereafter). Indexed by supernode index; empty/unused entries (roots
    // with nothing left over) simply never get read.
    struct GeneratedElement {
      std::vector<int> indices;  // final-order indices, size p
      MatrixX matrix;            // p x p, lower triangle valid (Schur complement)
    };
    std::vector<GeneratedElement> genElem(numSN);

    std::vector<int> globalToLocal(n, -1);  // scratch, reset after each front

    nf.fronts.reserve(numSN);

    for (int si = 0; si < numSN; ++si) {
      const auto& sn = supernodes[si];
      const int ncols = sn.ncols;

      // --- Step A: collect "extra" indices delayed in from children that
      // are not already part of this supernode's symbolic row pattern
      // (i.e. genuinely new eligible columns arriving at runtime). ---
      std::vector<int> extraFromChildren;
      for (int ci : childrenSN[si]) {
        const auto& ge = genElem[ci];
        for (int idx : ge.indices) {
          if (!std::binary_search(sn.rowPattern.begin(), sn.rowPattern.end(), idx)) {
            extraFromChildren.push_back(idx);
          }
        }
      }
      std::sort(extraFromChildren.begin(), extraFromChildren.end());
      extraFromChildren.erase(std::unique(extraFromChildren.begin(), extraFromChildren.end()),
                               extraFromChildren.end());
      const int nExtra = static_cast<int>(extraFromChildren.size());
      const int nEligible = ncols + nExtra;
      const int m = static_cast<int>(sn.rowPattern.size()) + nExtra;

      // --- Build the front's local index order: own columns, then
      // delayed-from-children extras (both eligible for pivoting here),
      // then the symbolic ancestor "update rows" (never eligible here). ---
      std::vector<int> frontIdx(m);
      for (int j = 0; j < ncols; ++j) frontIdx[j] = sn.firstCol + j;
      for (int j = 0; j < nExtra; ++j) frontIdx[ncols + j] = extraFromChildren[j];
      for (int j = ncols; j < static_cast<int>(sn.rowPattern.size()); ++j) {
        frontIdx[nEligible + (j - ncols)] = sn.rowPattern[j];
      }

      for (int t = 0; t < m; ++t) globalToLocal[frontIdx[t]] = t;

      MatrixX F = MatrixX::Zero(m, m);

      // --- Extend: A's own contribution to this supernode's own columns. ---
      for (int j = 0; j < ncols; ++j) {
        const int gj = sn.firstCol + j;
        const auto& rr = Aperm.rows[gj];
        const auto& vv = Aperm.vals[gj];
        for (std::size_t t = 0; t < rr.size(); ++t) {
          const int li = globalToLocal[rr[t]];
          F(li, j) += vv[t];
        }
      }

      // --- Add: each child's generated element, extend-add via the O(1)
      // scratch-array index map (never an O(m^2) search). ---
      for (int ci : childrenSN[si]) {
        auto& ge = genElem[ci];
        const int p = static_cast<int>(ge.indices.size());
        for (int a = 0; a < p; ++a) {
          const int la = globalToLocal[ge.indices[a]];
          for (int b = 0; b <= a; ++b) {
            const int lb = globalToLocal[ge.indices[b]];
            const Scalar v = ge.matrix(a, b);
            if (la >= lb)
              F(la, lb) += v;
            else
              F(lb, la) += v;
          }
        }
        ge.matrix.resize(0, 0);
        ge.indices.clear();
        ge.indices.shrink_to_fit();
      }

      for (int t = 0; t < m; ++t) globalToLocal[frontIdx[t]] = -1;  // reset scratch

      // --- Factor, restricted to the eligible (fully-summed) block. ---
      MatrixX D;
      DenseLDLTResult res = DenseLDLT<Scalar>::factor(F, D, options, nEligible);

      nf.inertia.n_pos += res.inertia.n_pos;
      nf.inertia.n_neg += res.inertia.n_neg;
      nf.inertia.n_zero += res.inertia.n_zero;

      const int nFactored = res.n_factored;

      // Global identity of physical front position i, post-factorization:
      // res.perm(i) is the *original* front-local index now sitting at
      // physical position i (identity for i >= nEligible, since swaps only
      // ever occur among eligible indices -- see dense_kernel.hpp).
      auto globalOf = [&](int i) { return frontIdx[res.perm(i)]; };

      SupernodeFactor sfac;
      sfac.nPivots = nFactored;
      sfac.rowIndices.resize(m);
      for (int i = 0; i < m; ++i) sfac.rowIndices[i] = globalOf(i);
      sfac.L = F.leftCols(nFactored);
      sfac.D = D.topLeftCorner(nFactored, nFactored);
      sfac.pivotBlocks = res.pivots;
      nf.fronts.push_back(std::move(sfac));

      // --- Forward the remainder (still-eligible-but-delayed columns +
      // ancestor rows) to the parent, or flag as singular if there is none. ---
      const int leftover = m - nFactored;
      if (leftover > 0) {
        const int psn = parentSN[si];
        if (psn == -1) {
          nf.singular = true;
          for (int i = nFactored; i < m; ++i) nf.singularCols.push_back(globalOf(i));
        } else {
          GeneratedElement ge;
          ge.indices.resize(leftover);
          for (int i = 0; i < leftover; ++i) ge.indices[i] = globalOf(nFactored + i);
          ge.matrix = F.block(nFactored, nFactored, leftover, leftover);
          genElem[si] = std::move(ge);
        }
      }
    }

    std::sort(nf.singularCols.begin(), nf.singularCols.end());
    return nf;
  }
};

}  // namespace symla
