#pragma once

// Phase 4: multifrontal triangular solve, operating directly on the
// per-front sparse structure produced by Phase 3 (`NumericFactor::fronts`)
// -- no dense n x n matrix is ever materialized, so this scales to large
// sparse problems (see multifrontal.hpp's docs for the exact meaning of
// `SupernodeFactor::{rowIndices,nPivots,L,D,pivotBlocks}`).
//
// Given P^T A P = L D L^T (P = permutation `NumericFactor::perm`, i.e.
// perm[t] = original row/col index of the t-th row/col in "final order"),
// solving A x = b is:
//
//   1. permute:      bp(t) = b(perm(t))                       for t in final order
//   2. forward solve L y = bp
//   3. diagonal solve D z = y   (1x1 or closed-form 2x2 per pivot block)
//   4. backward solve L^T w = z
//   5. un-permute:   x(perm(t)) = w(t)
//
// `NumericFactor::fronts` is stored in *processing* order: a valid
// elimination order where every front is processed strictly after all of
// its descendants (children before parents; see multifrontal.hpp). Two key
// facts make a tree-wise (not flattened) solve straightforward:
//
//   - A front's `nPivots` "own" rows are only ever written to (via extend-add
//     scatter) by its descendants, which -- by the above ordering guarantee
//     -- have all already been processed by the time this front is
//     processed. So a single left-to-right pass over `fronts` suffices for
//     the forward solve: gather the front's own-row partial sums (already
//     complete), do a small in-front triangular solve against the front's
//     own L block, then scatter-subtract the result's effect on the front's
//     "extra" rows (the rows forwarded to the parent as this front's
//     generated element) into the shared working RHS array -- mirroring
//     exactly the extend-add convention used during numeric factorization.
//   - Symmetrically, the backward solve is a single right-to-left
//     (reverse-processing-order) pass: a front's "extra" rows are only ever
//     read from an ancestor, which -- in reverse order -- has already been
//     processed (ancestors have strictly larger processing index than any
//     descendant, see multifrontal.hpp), so gather-subtract from those
//     already-finalized ancestor values, then do the small in-front
//     transposed triangular solve, and write the front's own rows into the
//     solution.
//
// All of the above operates on `nrhs` columns at once via Eigen block ops
// (no per-column loop), so multi-RHS solves get BLAS-3-ish throughput on
// each front's dense blocks.
//
// `SupernodeFactor::L`/`D` are stored as `double` regardless of the
// library's `Scalar` template parameter (see multifrontal.hpp -- numeric
// factorization already collapses to double storage internally), so this
// solve is likewise carried out in double precision internally regardless
// of `Scalar`, with the RHS/solution cast at the boundary.

#include "symla/dense_kernel.hpp"
#include "symla/multifrontal.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <vector>

namespace symla {

template <typename Scalar>
class MultifrontalSolver {
 public:
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXd = Eigen::MatrixXd;

  static DenseMatrix solve(const NumericFactor& nf, const DenseMatrix& B) {
    const int n = nf.n;
    if (static_cast<int>(B.rows()) != n) {
      throw std::invalid_argument("symla::MultifrontalSolver::solve: B.rows() must match the factorized matrix size");
    }
    const int nrhs = static_cast<int>(B.cols());
    if (n == 0) return DenseMatrix(0, nrhs);

    // Step 1: permute B into final order, in double precision.
    MatrixXd Y(n, nrhs);
    for (int t = 0; t < n; ++t) Y.row(t) = B.row(nf.perm[t]).template cast<double>();

    // Step 2: forward solve L y = bp, one pass over fronts in processing
    // (children-before-parents) order.
    for (const auto& f : nf.fronts) {
      const int m = static_cast<int>(f.rowIndices.size());
      const int k = f.nPivots;
      if (k == 0) continue;

      // Own-row RHS, already fully accumulated by earlier (descendant)
      // fronts' extend-add scatter below.
      MatrixXd yOwn(k, nrhs);
      for (int p = 0; p < k; ++p) yOwn.row(p) = Y.row(f.rowIndices[p]);

      MatrixXd Ltop = maskedTopBlock(f);
      Ltop.template triangularView<Eigen::UnitLower>().solveInPlace(yOwn);

      for (int p = 0; p < k; ++p) Y.row(f.rowIndices[p]) = yOwn.row(p);

      const int extra = m - k;
      if (extra > 0) {
        MatrixXd upd = f.L.bottomRows(extra) * yOwn;  // extra x nrhs
        for (int i = 0; i < extra; ++i) Y.row(f.rowIndices[k + i]) -= upd.row(i);
      }
    }

    // Step 3: block-diagonal solve D z = y, front-local pivot blocks (order
    // among fronts/blocks does not matter -- each pivot block is
    // independent).
    MatrixXd Z(n, nrhs);
    for (const auto& f : nf.fronts) {
      for (const auto& pb : f.pivotBlocks) {
        const int s = pb.start;
        if (pb.kind == PivotKind::OneByOne) {
          const int idx = f.rowIndices[s];
          Z.row(idx) = Y.row(idx) / f.D(s, s);
        } else {
          const int idx0 = f.rowIndices[s];
          const int idx1 = f.rowIndices[s + 1];
          const double d11 = f.D(s, s);
          const double d21 = f.D(s + 1, s);
          const double d22 = f.D(s + 1, s + 1);
          const double det = d11 * d22 - d21 * d21;
          const Eigen::RowVectorXd y0 = Y.row(idx0);
          const Eigen::RowVectorXd y1 = Y.row(idx1);
          Z.row(idx0) = (d22 * y0 - d21 * y1) / det;
          Z.row(idx1) = (-d21 * y0 + d11 * y1) / det;
        }
      }
    }

    // Step 4: backward solve L^T w = z, one pass over fronts in *reverse*
    // processing order.
    MatrixXd X(n, nrhs);
    for (auto it = nf.fronts.rbegin(); it != nf.fronts.rend(); ++it) {
      const auto& f = *it;
      const int m = static_cast<int>(f.rowIndices.size());
      const int k = f.nPivots;
      if (k == 0) continue;

      MatrixXd xOwn(k, nrhs);
      for (int p = 0; p < k; ++p) xOwn.row(p) = Z.row(f.rowIndices[p]);

      const int extra = m - k;
      if (extra > 0) {
        MatrixXd xExtra(extra, nrhs);
        for (int i = 0; i < extra; ++i) xExtra.row(i) = X.row(f.rowIndices[k + i]);
        xOwn.noalias() -= f.L.bottomRows(extra).transpose() * xExtra;
      }

      MatrixXd Ltop = maskedTopBlock(f);
      Ltop.template triangularView<Eigen::UnitLower>().transpose().solveInPlace(xOwn);

      for (int p = 0; p < k; ++p) X.row(f.rowIndices[p]) = xOwn.row(p);
    }

    // Step 5: un-permute back to original index space, cast to Scalar.
    DenseMatrix result(n, nrhs);
    for (int t = 0; t < n; ++t) result.row(nf.perm[t]) = X.row(t).template cast<Scalar>();
    return result;
  }

 private:
  // The front's own k x k leading block of L, with the strictly-lower "d21
  // slot" of every 2x2 pivot block zeroed out (that slot holds D's d21, not
  // an L multiplier -- see multifrontal.hpp/dense_kernel.hpp). The
  // remaining strictly-lower part is exactly the front-local (elimination-
  // order) unit-lower-triangular L multipliers; the stored diagonal is
  // whatever the dense kernel left behind (the pivot's D value or similar)
  // and is intentionally never read -- Eigen's UnitLower triangularView
  // always treats the diagonal as implicit 1, matching L's true structure.
  static MatrixXd maskedTopBlock(const SupernodeFactor& f) {
    const int k = f.nPivots;
    MatrixXd Ltop = f.L.topRows(k);
    for (const auto& pb : f.pivotBlocks) {
      if (pb.kind == PivotKind::TwoByTwo) Ltop(pb.start + 1, pb.start) = 0.0;
    }
    return Ltop;
  }
};

}  // namespace symla
