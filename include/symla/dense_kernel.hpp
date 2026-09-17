#pragma once

// Phase 2: dense frontal symmetric indefinite LDL^T kernel with
// Bunch-Kaufman-style threshold partial pivoting (1x1 and 2x2 pivot
// blocks), operating on a small-to-medium *dense* matrix representing a
// frontal matrix / supernode block in the eventual multifrontal method.
//
// This header is standalone dense linear algebra: no sparsity, no
// elimination tree, no assembly. Phase 3 will call DenseLDLT::factor() once
// per front.
//
// Algorithm: right-looking blocked (rank-1/rank-2 trailing update expressed
// via Eigen block operations, not naive triple loops) symmetric indefinite
// factorization, following the decision procedure of Bunch & Kaufman 1977
// ("Some stable methods for calculating inertia and solving symmetric
// linear systems", Math. Comp. 31:163-179) and the publicly documented
// algorithm behind LAPACK's dsytrf (see LAPACK user documentation; no
// LAPACK/HSL source was read to write this). LAPACK's dsytrf is used only
// as a black-box numerical oracle in the test suite, never as a reference
// implementation.
//
// P^T A P = L D L^T
//   - P: permutation (product of the symmetric pivot interchanges applied
//     during factorization), returned as `perm` such that
//     A_permuted(i,j) = A(perm(i), perm(j)).
//   - L: unit lower triangular, stored in the strictly-lower part of the
//     (in-place factored) input matrix A.
//   - D: block diagonal (1x1 and 2x2 blocks), returned separately in
//     `D_out` (a full n x n matrix but only the block-diagonal entries --
//     diagonal and the single off-diagonal per 2x2 block -- are populated;
//     everything else is left as zero).
//
// Degenerate / delayed pivots: if, at some column k, no numerically
// acceptable 1x1 or 2x2 pivot can be found in the trailing submatrix
// without violating the hard floor (all remaining candidate entries are
// (numerically) zero, or the best 2x2 candidate is singular to within
// `zero_tolerance` relative to the matrix norm), factorization of that
// leading block stops there: `n_factored < n`, and the remaining local
// column indices are reported in `delayed_cols`, front-local (post
// already-applied permutation) indices. This maps onto MA57's delayed
// pivot concept -- Phase 3's multifrontal driver is expected to pass these
// delayed rows/columns up to the parent front rather than treat this as an
// error.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "symla/inertia.hpp"  // symla::Inertia

namespace symla {

struct DenseLDLTOptions {
  // Bunch-Kaufman threshold parameter `u`. Must lie in (0, 0.5]. Smaller
  // values favor the diagonal (fewer swaps, less fill in a sparse driver)
  // at some cost to stability; u = alpha = (1+sqrt(17))/8 is the classical
  // "optimal worst case growth" choice used internally as the decision
  // threshold `alpha` regardless of this option -- `pivot_threshold` here
  // plays the same role as MA57's/LAPACK's `u` in some presentations, but
  // to keep this implementation simple and textbook-faithful we use the
  // fixed alpha = (1+sqrt(17))/8 for the Bunch-Kaufman test itself and
  // reserve `pivot_threshold` as an additional configurable floor a future
  // phase (KKT/static pivoting) may use. Kept in the options struct now so
  // the API doesn't need to change later.
  double pivot_threshold = 0.01;

  // Magnitudes at/below this are treated as structurally/numerically zero
  // when searching for a pivot (used both for the "lambda == 0" fast path
  // and for judging 2x2 pivot-block singularity / triggering delayed
  // pivots). Deliberately tiny (near DBL_MIN) so it only catches literal
  // (or bit-pattern-near) zeros, not "small relative to this matrix"
  // values -- see `relative_pivot_floor` below for that.
  double zero_tolerance = 1e-300;

  // Phase 5 fix (found via real-world SuiteSparse Collection matrices,
  // e.g. GHS_indef/sit100 -- see test/correctness/real_matrix_test.cpp):
  // a column that has *no* off-diagonal support at all within the current
  // eligible block (lambda <= zero_tolerance, the "isolated diagonal"
  // case) used to be force-accepted as a 1x1 pivot as long as |A(k,k)|
  // exceeded the absurdly small `zero_tolerance` (1e-300) -- i.e.
  // essentially always, even when A(k,k) was itself down at floating-point
  // noise level (e.g. ~1e-40) relative to the rest of the matrix. Ordinary
  // Bunch-Kaufman relative comparisons are intentionally scale-invariant
  // (that's what gives the bounded-growth-factor guarantee) and are left
  // untouched by this option, but the "isolated, no other candidate"
  // fallback has no relative comparison to anchor it at all, so a
  // catastrophically tiny isolated pivot would get divided into directly,
  // blowing up L's multipliers (and hence the whole solve) by many orders
  // of magnitude. MA57 itself avoids this by delaying such columns up the
  // tree (where extend-add assembly with ancestor rows may give them real
  // off-diagonal support) rather than pivoting on numerical noise. This
  // option is that floor: an isolated column's |A(k,k)| must exceed
  // `relative_pivot_floor * (max|entry| of the original front)` to be
  // accepted directly; otherwise it is delayed (same code path as a
  // genuinely-zero isolated diagonal). Expressed relative to the front's
  // own scale (not an absolute constant) so it behaves consistently
  // whether the matrix's natural magnitudes are ~1e-6 or ~1e6.
  double relative_pivot_floor = 1e-12;
};

enum class PivotKind { OneByOne, TwoByTwo };

struct PivotBlock {
  int start;       // front-local column index (post already-applied
                    // permutation) where this pivot block begins
  PivotKind kind;
};

struct DenseLDLTResult {
  int n_factored = 0;                 // number of leading columns successfully factored
  std::vector<int> delayed_cols;      // local column indices (in the *permuted* frame) delayed to the parent
  std::vector<PivotBlock> pivots;     // pivot structure of the factored leading part
  Eigen::VectorXi perm;               // permutation applied (local to this front), size n;
                                       // A_permuted(i,j) == A_original(perm(i), perm(j))
  Inertia inertia;                    // sign counts among the n_factored factored pivots
  double max_growth = 1.0;            // max|entry| encountered / max|entry| of original A (growth factor)
};

template <typename Scalar>
class DenseLDLT {
 public:
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // Factors `A` (n x n, symmetric; only the lower triangle is read) in
  // place. On return:
  //   - The strictly-lower part of A (rows/cols < n_factored) holds the
  //     multipliers of L (unit diagonal implied, not stored).
  //   - D_out holds the block-diagonal D (n x n, mostly zero; see above).
  //   - Entries of A/D_out at or beyond n_factored (i.e. touching any
  //     delayed column) are left in an unspecified but harmless state; the
  //     caller (Phase 3) is expected to re-assemble delayed columns into
  //     the parent front rather than trust this kernel's output for them.
  // `n_eligible` (Phase 3 extension): restricts which leading columns are
  // allowed to be *chosen* as a pivot (either directly, at column k, or as
  // the partner row/column r of a 1x1-swap or 2x2 pivot). Columns/rows at
  // index >= n_eligible (e.g. a multifrontal front's not-yet-fully-summed
  // ancestor rows) still fully participate in the trailing rank-1/rank-2
  // symmetric updates once a pivot elsewhere is applied, but are never
  // themselves pivoted -- exactly the "fully summed vs. not fully summed"
  // distinction in the multifrontal method (Duff & Reid 1983 / Liu 1990).
  // A sentinel of -1 (the default) means "all n columns are eligible",
  // reproducing the original (Phase 2) behavior exactly.
  //
  // Implementation note: since n_eligible only restricts the *search* range
  // for lambda/sigma/r (never the trailing-update extent, which always
  // spans the full remaining n-k rows/cols), and swaps only ever occur
  // between indices < n_eligible, physical positions >= n_eligible are
  // never permuted (result.perm stays the identity there) and, if the loop
  // stops early (k < n_eligible) because no acceptable pivot exists within
  // the eligible block, A's trailing block from k onward already holds
  // exactly the Schur complement with respect to the k successfully
  // factored pivots -- i.e. it is directly usable as a multifrontal
  // "generated element" / update matrix, not just "harmless but
  // unspecified" as in the pure Phase 2 (n_eligible == n) case.
  static DenseLDLTResult factor(Eigen::Ref<MatrixX> A, MatrixX& D_out,
                                 const DenseLDLTOptions& options = {}, int n_eligible = -1) {
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n) throw std::invalid_argument("DenseLDLT::factor: A must be square");
    const int effEnd = (n_eligible < 0) ? n : n_eligible;
    if (effEnd < 0 || effEnd > n) throw std::invalid_argument("DenseLDLT::factor: n_eligible out of range");

    const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;

    DenseLDLTResult result;
    result.perm = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
    D_out = MatrixX::Zero(n, n);

    // Mirror lower -> upper so we can freely read either triangle without
    // having to remember which is authoritative at each step (cheap; only
    // needed once up front, and re-mirrored locally after each swap).
    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < n; ++i) A(j, i) = A(i, j);

    double orig_max = A.cwiseAbs().maxCoeff();
    if (orig_max < options.zero_tolerance) orig_max = 1.0;  // avoid div-by-zero on the all-zero matrix
    double running_max = orig_max;

    auto swap_rowcol = [&](int i, int j) {
      if (i == j) return;
      A.row(i).swap(A.row(j));
      A.col(i).swap(A.col(j));
      std::swap(result.perm(i), result.perm(j));
    };

    int k = 0;
    while (k < effEnd) {
      const int m = n - k;  // trailing submatrix size (full, incl. ineligible rows)

      // Step 1: lambda = max_{k<i<effEnd} |A(i,k)|, at row r. Restricted to
      // the eligible (fully-summed) block: an entry below the diagonal that
      // lives in a not-fully-summed row can never become a pivot partner.
      double lambda = 0.0;
      int r = -1;
      for (int i = k + 1; i < effEnd; ++i) {
        const double v = std::abs(A(i, k));
        if (v > lambda) {
          lambda = v;
          r = i;
        }
      }

      bool accept_1x1_at_k = false;
      bool accept_1x1_swap_kr = false;

      if (lambda <= options.zero_tolerance) {
        // No off-diagonal mass below the diagonal (or last column): the
        // only candidate is A(k,k) itself -- but accept it only if it's
        // not down at noise level relative to this front's own scale (see
        // `relative_pivot_floor`'s doc comment above). Either an
        // exactly/near-zero isolated diagonal (structurally singular here)
        // or a numerically negligible one are both handled the same way:
        // delay column k (and everything after) to the parent, where
        // extend-add assembly with ancestor rows may give it real
        // off-diagonal support.
        const double isolatedFloor = std::max(options.zero_tolerance, options.relative_pivot_floor * orig_max);
        if (std::abs(A(k, k)) <= isolatedFloor) {
          break;  // delay column k (and everything after) to the parent
        }
        accept_1x1_at_k = true;
      } else if (std::abs(A(k, k)) >= alpha * lambda) {
        accept_1x1_at_k = true;
      } else {
        // sigma = max_{i != r, k<=i<effEnd} |A(i,r)| (restricted to the
        // eligible block, same rationale as lambda above).
        double sigma = 0.0;
        for (int i = k; i < effEnd; ++i) {
          if (i == r) continue;
          const double v = std::abs(A(i, r));
          if (v > sigma) sigma = v;
        }

        // Note: if sigma <= zero_tolerance (column r has no other
        // off-diagonal mass in the trailing block), the first test below
        // reduces to "A(k,k) still not good enough" (since it already
        // failed the alpha*lambda test above and sigma~0 makes the LHS
        // ~0), so it correctly falls through toward the 2x2 case unless
        // r's own diagonal saves it.
        if (std::abs(A(k, k)) * sigma >= alpha * lambda * lambda) {
          accept_1x1_at_k = true;
        } else if (std::abs(A(r, r)) >= alpha * sigma) {
          accept_1x1_swap_kr = true;
        }
        // else: accept a 2x2 pivot at (k, r) -- handled by the fallthrough
        // block below (neither accept_1x1_at_k nor accept_1x1_swap_kr set).
      }

      // Note: we deliberately do *not* apply an absolute floor to the
      // ordinary (non-isolated) Bunch-Kaufman relative comparisons above --
      // an experiment doing so (rejecting an about-to-be-accepted 1x1
      // pivot whenever it fell below `relative_pivot_floor * orig_max`,
      // falling through to try a 2x2 at (k, r) instead) was tried during
      // Phase 5 and made real-matrix residuals *worse*, not better,
      // presumably by forcing 2x2 pivots in cases the classical relative
      // test had good (bounded-growth-factor) reasons to avoid. Left as a
      // real, open item for Phase 6/7: see the Phase 5 report's notes on
      // GHS_indef/sit100 and friends still showing elevated (~1e-2)
      // residuals -- pure textbook Bunch-Kaufman is scale-invariant by
      // design, and fixing this properly likely needs MA57/PARDISO-style
      // static pivoting + regularization (already planned for Phase 7),
      // not another ad hoc floor here.

      if (accept_1x1_at_k) {
        running_max = std::max(running_max, std::abs(A(k, k)));
        const double d = A(k, k);
        D_out(k, k) = d;
        result.pivots.push_back({k, PivotKind::OneByOne});
        if (d > 0)
          ++result.inertia.n_pos;
        else if (d < 0)
          ++result.inertia.n_neg;
        else
          ++result.inertia.n_zero;

        if (m > 1) {
          VectorX l = A.col(k).segment(k + 1, m - 1) / d;
          running_max = std::max(running_max, A.block(k + 1, k, m - 1, 1).cwiseAbs().maxCoeff());
          // Rank-1 symmetric trailing update:
          // A(i,j) -= l(i) * d * l(j)  for i,j in (k, n)
          auto trailing = A.block(k + 1, k + 1, m - 1, m - 1);
          trailing.noalias() -= (d * l) * l.transpose();
          // keep exact symmetry (avoid drift from floating point order of ops)
          trailing = ((trailing + trailing.transpose()) * 0.5).eval();
          A.col(k).segment(k + 1, m - 1) = l;
          A.row(k).segment(k + 1, m - 1) = l.transpose();
          running_max = std::max(running_max, trailing.cwiseAbs().maxCoeff());
        }
        ++k;
        continue;
      }

      if (accept_1x1_swap_kr) {
        swap_rowcol(k, r);
        running_max = std::max(running_max, std::abs(A(k, k)));
        const double d = A(k, k);
        D_out(k, k) = d;
        result.pivots.push_back({k, PivotKind::OneByOne});
        if (d > 0)
          ++result.inertia.n_pos;
        else if (d < 0)
          ++result.inertia.n_neg;
        else
          ++result.inertia.n_zero;

        if (m > 1) {
          VectorX l = A.col(k).segment(k + 1, m - 1) / d;
          running_max = std::max(running_max, A.block(k + 1, k, m - 1, 1).cwiseAbs().maxCoeff());
          auto trailing = A.block(k + 1, k + 1, m - 1, m - 1);
          trailing.noalias() -= (d * l) * l.transpose();
          trailing = ((trailing + trailing.transpose()) * 0.5).eval();
          A.col(k).segment(k + 1, m - 1) = l;
          A.row(k).segment(k + 1, m - 1) = l.transpose();
          running_max = std::max(running_max, trailing.cwiseAbs().maxCoeff());
        }
        ++k;
        continue;
      }

      // accept_2x2: bring r into position k+1 (k is already fine as-is).
      {
        if (r != k + 1) swap_rowcol(k + 1, r);
        const double d11 = A(k, k);
        const double d21 = A(k + 1, k);
        const double d22 = A(k + 1, k + 1);
        const double det = d11 * d22 - d21 * d21;

        if (std::abs(det) <= options.zero_tolerance * std::max(1.0, running_max) * std::max(1.0, running_max)) {
          // Degenerate 2x2 block: delay column k (and k+1 onward).
          // Undo is unnecessary since we simply stop factoring here; the
          // caller treats everything from k onward as delayed. (The r<->k+1
          // swap already applied is harmless: it's still a valid symmetric
          // permutation of the still-unfactored trailing block, and the
          // delayed columns carry `perm` with them.)
          break;
        }

        running_max = std::max(running_max, std::max({std::abs(d11), std::abs(d21), std::abs(d22)}));

        D_out(k, k) = d11;
        D_out(k + 1, k) = d21;
        D_out(k, k + 1) = d21;
        D_out(k + 1, k + 1) = d22;
        result.pivots.push_back({k, PivotKind::TwoByTwo});

        // Inertia of a 2x2 symmetric indefinite-by-construction block:
        // det < 0 always here (Bunch-Kaufman guarantees the 2x2 case is
        // indefinite), so one +, one -.
        if (det < 0) {
          ++result.inertia.n_pos;
          ++result.inertia.n_neg;
        } else {
          // Shouldn't happen by construction, but handle gracefully via
          // trace sign if it ever does (numerical edge case).
          const double tr = d11 + d22;
          if (tr > 0) {
            result.inertia.n_pos += 2;
          } else if (tr < 0) {
            result.inertia.n_neg += 2;
          } else {
            ++result.inertia.n_pos;
            ++result.inertia.n_neg;
          }
        }

        const int mm = n - (k + 2);  // rows/cols strictly below the 2x2 block
        if (mm > 0) {
          // inv(D2) = 1/det * [[d22, -d21], [-d21, d11]]
          const double inv11 = d22 / det;
          const double inv21 = -d21 / det;
          const double inv22 = d11 / det;

          MatrixX rhs = A.block(k + 2, k, mm, 2);  // [A(:,k) A(:,k+1)]
          running_max = std::max(running_max, rhs.cwiseAbs().maxCoeff());
          MatrixX L2(mm, 2);
          L2.col(0) = rhs.col(0) * inv11 + rhs.col(1) * inv21;
          L2.col(1) = rhs.col(0) * inv21 + rhs.col(1) * inv22;

          // Rank-2 symmetric trailing update:
          // A -= L2 * D2 * L2^T, D2 = [[d11,d21],[d21,d22]]
          MatrixX D2(2, 2);
          D2 << d11, d21, d21, d22;
          auto trailing = A.block(k + 2, k + 2, mm, mm);
          trailing.noalias() -= L2 * D2 * L2.transpose();
          trailing = ((trailing + trailing.transpose()) * 0.5).eval();

          A.block(k + 2, k, mm, 2) = L2;
          A.block(k, k + 2, 2, mm) = L2.transpose();
          running_max = std::max(running_max, trailing.cwiseAbs().maxCoeff());
        }
        k += 2;
        continue;
      }
    }

    result.n_factored = k;
    // Only columns within the eligible block that failed to pivot are
    // "delayed" in the MA57 sense; ineligible rows/cols (>= effEnd, e.g. a
    // multifrontal front's ancestor rows) were never candidates to begin
    // with and must not be reported as delayed.
    for (int j = k; j < effEnd; ++j) result.delayed_cols.push_back(j);
    result.max_growth = running_max / orig_max;
    return result;
  }
};

}  // namespace symla
