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

#include "symla/solver.hpp"  // symla::Inertia

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
  // pivots).
  double zero_tolerance = 1e-300;
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
  static DenseLDLTResult factor(Eigen::Ref<MatrixX> A, MatrixX& D_out,
                                 const DenseLDLTOptions& options = {}) {
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n) throw std::invalid_argument("DenseLDLT::factor: A must be square");

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
    while (k < n) {
      const int m = n - k;  // trailing submatrix size

      // Step 1: lambda = max_{i>k} |A(i,k)|, at row r.
      double lambda = 0.0;
      int r = -1;
      for (int i = k + 1; i < n; ++i) {
        const double v = std::abs(A(i, k));
        if (v > lambda) {
          lambda = v;
          r = i;
        }
      }

      bool accept_1x1_at_k = false;
      bool accept_1x1_swap_kr = false;

      if (m == 1 || lambda <= options.zero_tolerance) {
        // No off-diagonal mass below the diagonal (or last column): accept
        // a 1x1 pivot at k directly. If A(k,k) itself is (numerically)
        // zero too, this is a structurally singular pivot -- signal a
        // delayed pivot rather than dividing by ~0.
        if (std::abs(A(k, k)) <= options.zero_tolerance) {
          break;  // delay column k (and everything after) to the parent
        }
        accept_1x1_at_k = true;
      } else if (std::abs(A(k, k)) >= alpha * lambda) {
        accept_1x1_at_k = true;
      } else {
        // sigma = max_{i != r, k<=i<n} |A(i,r)|
        double sigma = 0.0;
        for (int i = k; i < n; ++i) {
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
    for (int j = k; j < n; ++j) result.delayed_cols.push_back(j);
    result.max_growth = running_max / orig_max;
    return result;
  }
};

}  // namespace symla
