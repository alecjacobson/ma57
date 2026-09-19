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
//     `D_out` (a `BlockDiagonalD`, O(n) storage: a diagonal vector plus a
//     parallel vector holding the single off-diagonal per 2x2 block --
//     see `BlockDiagonalD`'s doc comment above for the compact layout and
//     why a dense n x n matrix is unnecessary).
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
#include <limits>
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

  // Phase 8 (panel-blocked dense kernel): number of columns processed per
  // "panel" before the accumulated rank-p update from that panel's pivots
  // is applied to the remaining trailing submatrix as a small number of
  // large BLAS-3-style Eigen GEMM calls, instead of one rank-1/rank-2 call
  // per pivot as the original (Phase 2) unblocked algorithm did -- see
  // factor()'s implementation comment for the full design and rationale.
  // Must be >= 1 (values <= 0 fall back to the default). A value >=
  // n_eligible degenerates to exactly one panel covering the whole
  // eligible range, which the test suite uses to cross-check the blocked
  // and (mathematically equivalent) fully-unblocked-style code paths
  // against each other.
  int panel_size = 64;
};

// D is block-diagonal by construction (only 1x1 and 2x2 blocks along the
// diagonal -- see PivotBlock/PivotKind below), so representing it as a dense
// n x n matrix (as an earlier version of this code did) wastes O(n^2)
// storage/zeroing time for what only ever needs O(n): the diagonal, plus (at
// most) one off-diagonal entry per 2x2 pivot block. This type stores exactly
// that -- a diagonal vector plus a parallel "d21" vector, indexed by the
// *first* (lower) column of a would-be 2x2 block; entries at any index that
// is not the start of an actual 2x2 pivot block stay structurally zero,
// exactly matching the corresponding entry of a real dense block-diagonal D.
//
// `operator()(i, j)` mimics a dense matrix accessor restricted to exactly
// the entries a block-diagonal D can ever have: the diagonal (i == j) and
// the single permitted off-diagonal of a 2x2 block (|i - j| == 1). This
// keeps every existing call site that reads/writes `D(k, k)`, `D(k+1, k)`,
// `D(k, k+1)`, `D(k+1, k+1)` unchanged in spirit -- only the storage
// underneath is compact. The mutable overload throws on any other (i, j)
// (a caller asking for such an entry is a bug -- a real block-diagonal D
// simply cannot have one); the const overload returns 0 there, matching
// what a dense D would actually contain.
template <typename Scalar>
struct BlockDiagonalD {
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  VectorX diag;     // size n: D(k, k)
  VectorX offdiag;  // size n: offdiag(k) == D(k+1, k) for a 2x2 block starting at column k, else 0

  void resize(int n) {
    diag = VectorX::Zero(n);
    offdiag = VectorX::Zero(n);
  }
  int size() const { return static_cast<int>(diag.size()); }
  int rows() const { return size(); }
  int cols() const { return size(); }

  Scalar operator()(int i, int j) const {
    if (i == j) return diag(i);
    if (i == j + 1) return offdiag(j);
    if (j == i + 1) return offdiag(i);
    return Scalar(0);
  }
  Scalar& operator()(int i, int j) {
    if (i == j) return diag(i);
    if (i == j + 1) return offdiag(j);
    if (j == i + 1) return offdiag(i);
    throw std::invalid_argument(
        "BlockDiagonalD::operator(): only diagonal / adjacent 2x2-block entries are addressable");
  }

  // Leading `n`-sized sub-block (block-diagonal structure means this is just
  // `head(n)` of both vectors -- there is no real "corner" to speak of).
  BlockDiagonalD head(int n) const {
    BlockDiagonalD out;
    out.diag = diag.head(n);
    out.offdiag = offdiag.head(n);
    return out;
  }

  template <typename Target>
  BlockDiagonalD<Target> cast() const {
    BlockDiagonalD<Target> out;
    out.diag = diag.template cast<Target>();
    out.offdiag = offdiag.template cast<Target>();
    return out;
  }

  // Materializes the full dense n x n block-diagonal matrix. Test/debug
  // convenience only -- production code should never need this; avoiding
  // exactly this materialization is the point of this type.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> toDense() const {
    const int n = size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
    for (int k = 0; k < n; ++k) out(k, k) = diag(k);
    for (int k = 0; k + 1 < n; ++k) {
      if (offdiag(k) != Scalar(0)) {
        out(k + 1, k) = offdiag(k);
        out(k, k + 1) = offdiag(k);
      }
    }
    return out;
  }
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

  // Phase 7 (static pivoting only; always 0/empty for the ordinary
  // Bunch-Kaufman `factor()` path above): diagonal perturbation applied
  // while accepting pivots, see `DenseLDLT::factorStatic` below.
  double total_perturbation = 0.0;    // sum of |delta| applied across all perturbed pivots
  int num_perturbed = 0;              // count of pivots that needed perturbation
};

// Phase 7: options controlling `DenseLDLT::factorStatic` (KKT-aware static
// pivoting), see the function's doc comment.
struct StaticPivotOptions {
  // Diagonal perturbation magnitude to add (sign-matched to the expected/
  // observed pivot sign) when a pivot is judged unacceptable. If <= 0, a
  // default of `sqrt(machine epsilon) * front_scale` is used (front_scale ==
  // this call's `orig_max`, i.e. the max|entry| of the front before
  // factoring). The inertia-controlled retry loop in solver.hpp escalates
  // this geometrically across factorize() attempts.
  double delta = 0.0;

  // A pivot is judged "too small to use directly" (and thus perturbed) when
  // |A(k,k)| < max(absolute_floor, relative_pivot_floor * front_scale).
  // Mirrors DenseLDLTOptions::relative_pivot_floor's semantics/rationale.
  double relative_pivot_floor = 1e-8;
  double absolute_floor = 1e-300;

  // Phase 9: mirrors `DenseLDLTOptions::panel_size` -- process eligible
  // columns in panels of up to this many columns, deferring the O(m^2)
  // trailing-submatrix update to one blocked GEMM per panel instead of
  // applying every single column's rank-1 update immediately to the entire
  // remaining trailing block. See `DenseLDLT::factorStatic`'s doc comment
  // for why this needs none of `factor()`'s swap-driven raw/touched
  // bookkeeping: with no swaps and no 2x2 pivots, every column's up-to-date
  // value is always a simple, exact function of a per-panel raw snapshot
  // and the panel's pivots-so-far, computed fresh (never incrementally)
  // exactly once, when that column itself becomes the pivot.
  int panel_size = 64;
};

template <typename Scalar>
class DenseLDLT {
 public:
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // Shared by `factor()` and `factorStatic()`: re-averages both triangles of
  // a (potentially large) trailing block, `A(i,j) = A(j,i) = 0.5*(A(i,j) +
  // A(j,i))`, via an explicit cache-blocked (tile-at-a-time) loop rather
  // than a single whole-block Eigen expression -- see `factor()`'s header
  // comment above its original inline copy of this lambda for the profiling
  // story (a naive `((A + A.transpose()) * 0.5).eval()` on a 2745x2745
  // front cost ~40 of ~45 total seconds; this tiled version is ~5-6x
  // faster, bit-identical in the values it produces).
  template <typename Block>
  static void resymmetrizeTrailingTiled(Block trailingBlock) {
    const int extent = static_cast<int>(trailingBlock.rows());
    if (extent == 0) return;
    constexpr int kTile = 64;
    for (int jb = 0; jb < extent; jb += kTile) {
      const int jn = std::min(kTile, extent - jb);
      for (int ib = jb; ib < extent; ib += kTile) {
        const int in = std::min(kTile, extent - ib);
        if (ib == jb) {
          for (int j = jb; j < jb + jn; ++j) {
            for (int i = j + 1; i < jb + jn; ++i) {
              const double avg = 0.5 * (trailingBlock(i, j) + trailingBlock(j, i));
              trailingBlock(i, j) = avg;
              trailingBlock(j, i) = avg;
            }
          }
        } else {
          for (int j = jb; j < jb + jn; ++j) {
            for (int i = ib; i < ib + in; ++i) {
              const double avg = 0.5 * (trailingBlock(i, j) + trailingBlock(j, i));
              trailingBlock(i, j) = avg;
              trailingBlock(j, i) = avg;
            }
          }
        }
      }
    }
  }

  // Factors `A` (n x n, symmetric; only the lower triangle is read) in
  // place. On return:
  //   - The strictly-lower part of A (rows/cols < n_factored) holds the
  //     multipliers of L (unit diagonal implied, not stored).
  //   - D_out holds the block-diagonal D (compact `BlockDiagonalD`; see above).
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
  // Phase 8: panel-blocked, delayed-trailing-update right-looking
  // factorization (LAPACK dsytrf/dlasyf-style; see the header comment's
  // "no LAPACK/HSL source was read" note -- this follows the publicly
  // documented design of a blocked-panel-with-delayed-trailing-update
  // scheme, not any proprietary implementation).
  //
  // The 1x1/2x2 Bunch-Kaufman decision logic below is *exactly* the
  // decision logic of the original (Phase 2) unblocked algorithm --
  // unchanged thresholds, unchanged lambda/sigma/r computation, unchanged
  // delayed-pivot conditions. The only thing this phase changes is *when*
  // the rank-1/rank-2 trailing update from an accepted pivot is applied:
  //
  //   - Columns are processed in "panels" of up to `options.panel_size`
  //     eligible columns at a time (`panelStart` .. `panelEnd`).
  //   - Within a panel, each pivot's rank-1/rank-2 update is applied
  //     *lazily and only to the specific column(s) the pivot-search logic
  //     is about to read* (the current column `k`, and -- only if the
  //     lambda/sigma test needs it -- the candidate partner column `r`,
  //     which may lie anywhere in [k, effEnd), not necessarily inside the
  //     panel's own column range). This is done via `ensureColumnCurrent`,
  //     which brings a column's live rows [k, n) up to date with however
  //     many of this panel's pivots-so-far haven't yet been applied to it
  //     (tracked per-column via `appliedCount`), using a small GEMV over
  //     just the panel-local L/D buffers (`Lbuf`/`Dbuf`) -- O(m * p) where
  //     p <= panel_size, instead of the O(m) rank-1 update the unblocked
  //     algorithm applied to the *entire* trailing matrix after literally
  //     every pivot.
  //   - A column examined as a candidate `r` during the search is not
  //     always the column that ends up chosen (e.g. the sigma test can
  //     still land on accept_1x1_at_k). Such a column keeps whatever
  //     partial correction `ensureColumnCurrent` already applied to it;
  //     `appliedCount` tracks exactly how much, so any later touch (as a
  //     future `k` or a future `r`) only applies the remaining delta --
  //     never re-derives or double-applies anything.
  //   - Once the panel finishes (either by reaching `panel_size` columns,
  //     or by stopping early because a pivot was delayed/degenerate --
  //     exactly the original algorithm's delayed-pivot `break`), any
  //     column in the live trailing region that still carries a *partial*
  //     panel-local correction (i.e. was examined as an `r` but never
  //     finalized as a pivot) is first reverted back to its pre-panel
  //     ("raw") state, so that a single blocked rank-p update --
  //     `trailingBeyondPanel -= Lpanel * Dpanel * Lpanel^T`, expressed as
  //     genuine Eigen block GEMM calls -- can then be applied uniformly to
  //     the *entire* remaining trailing extent at once. This is the
  //     BLAS-3-throughput step the whole scheme exists to enable.
  //
  // Row/column swaps (`panelSwap`, a thin wrapper around the original
  // `swap_rowcol` that additionally keeps `Lbuf` rows and `appliedCount`
  // in sync with whatever physical row/column they belong to) can move a
  // column from outside the panel's own range into pivot position (e.g. a
  // 2x2 pivot's partner `r` found beyond `panelEnd`, within the eligible
  // range) -- swaps always happen *before* any panel-local correction is
  // applied for that pivot, exactly mirroring the unblocked code's
  // ordering, so a swapped-in column's already-applied partial correction
  // (if any) simply moves with it.
  //
  // A `panel_size` >= `n_eligible` collapses this to exactly one panel
  // spanning the whole eligible range, which is mathematically equivalent
  // to the original fully unblocked algorithm (one deferred update instead
  // of many, but covering the same total rank -- see the correctness test
  // suite's explicit head-to-head sweep across panel sizes, including huge
  // ones, cross-checked against small panel sizes that stress panel-
  // boundary logic hard even on small matrices).
  static DenseLDLTResult factor(Eigen::Ref<MatrixX> A, BlockDiagonalD<Scalar>& D_out,
                                 const DenseLDLTOptions& options = {}, int n_eligible = -1) {
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n) throw std::invalid_argument("DenseLDLT::factor: A must be square");
    const int effEnd = (n_eligible < 0) ? n : n_eligible;
    if (effEnd < 0 || effEnd > n) throw std::invalid_argument("DenseLDLT::factor: n_eligible out of range");

    const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
    const int panelSizeOpt = options.panel_size > 0 ? options.panel_size : 64;

    DenseLDLTResult result;
    result.perm = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
    D_out.resize(n);

    // Mirror lower -> upper so we can freely read either triangle without
    // having to remember which is authoritative at each step (cheap; only
    // needed once up front, and re-mirrored locally after each swap).
    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < n; ++i) A(j, i) = A(i, j);

    double orig_max = A.cwiseAbs().maxCoeff();
    if (orig_max < options.zero_tolerance) orig_max = 1.0;  // avoid div-by-zero on the all-zero matrix
    double running_max = orig_max;

    // After each rank-1/rank-2 trailing update below, both triangles of the
    // (potentially huge) trailing block are re-averaged for exact symmetric
    // consistency: `A(i,j) = A(j,i) = 0.5*(A(i,j)+A(j,i))`. A previous
    // version of this code did this via a single Eigen expression,
    // `A = ((A + A.transpose()) * 0.5).eval()`, which forces Eigen to
    // evaluate a transposed read of a large non-contiguous `Block` of a
    // column-major matrix -- extremely cache-hostile (each "row" of the
    // transpose read strides across the whole matrix's leading dimension)
    // -- profiling on a 2745x2745 front found that single operation
    // responsible for ~40 of ~45 total seconds of factor() time.
    //
    // Two cheaper alternatives were tried and rejected before this one:
    //   - Deleting the re-symmetrization entirely (relying on the rank-1/
    //     rank-2 update already being "close enough" to symmetric):
    //     measurably worsened real-matrix residuals (e.g. GHS_indef/sit100
    //     went from ~0.031 to ~0.35 against a 0.05 test bound) -- on
    //     borderline ill-conditioned inputs, which of two mathematically
    //     equal but last-bit-different roundings ends up stored evidently
    //     does affect downstream Bunch-Kaufman tie-breaking enough to
    //     matter, and simply averaging the two, as the original code did,
    //     measurably helps.
    //   - Only re-symmetrizing the small "eligible" (not-yet-pivoted,
    //     fully-summed) leading corner rather than the whole trailing
    //     block, on the reasoning that pivot search never reads outside
    //     that corner: also measurably worsened residuals, because the
    //     *averaged* value (not just a consistently-mirrored one) in the
    //     larger "ancestor" region is exactly what gets forwarded to the
    //     parent front as this front's generated element / Schur
    //     complement, so skipping the averaging there changes the numeric
    //     values a later front's pivot search *does* see, even though this
    //     front's own search never touches that region.
    // What actually works: the exact same "average both triangles" formula
    // as the original, just computed via an explicit cache-blocked
    // (tile-at-a-time) loop instead of Eigen's whole-block transpose
    // expression, so both the read and write working sets stay
    // cache-resident. This is bit-identical to the original (same formula,
    // same operands, order-independent per entry) and, on the same
    // 2745x2745 front, roughly 5-6x faster than the original single-shot
    // expression.
    auto resymmetrizeTrailing = [&](auto trailingBlock) { resymmetrizeTrailingTiled(trailingBlock); };

    auto swap_rowcol = [&](int i, int j) {
      if (i == j) return;
      A.row(i).swap(A.row(j));
      A.col(i).swap(A.col(j));
      std::swap(result.perm(i), result.perm(j));
    };

    int k = 0;
    while (k < effEnd) {
      // ==================== Begin one panel ====================
      const int panelStart = k;
      const int panelWidthTarget = std::min(panelSizeOpt, effEnd - panelStart);
      const int panelEnd = panelStart + panelWidthTarget;
      // +1: room for a 2x2 pivot whose first column lands exactly at
      // panelEnd - 1 (its partner column is always < effEnd, so this is
      // always sufficient -- see the class-level comment above).
      const int panelCapacity = panelWidthTarget + 1;
      const int m0 = n - panelStart;  // rows spanned by this panel's L buffer (down to n, incl. ineligible rows)

      MatrixX Lbuf = MatrixX::Zero(m0, panelCapacity);
      BlockDiagonalD<Scalar> Dbuf;
      Dbuf.resize(panelCapacity);

      // Snapshot of this panel's starting state (i.e. reflecting every
      // *prior* panel's effect, but none of this panel's own pivots yet),
      // kept in lockstep with every row/column swap performed while this
      // panel is live (see `panelSwap` below). This is the anchor every
      // column-current computation below recomputes from -- see
      // `setColumnRaw`'s comment for why a fresh, from-scratch recompute
      // (rather than incremental delta application) is required for
      // correctness once swaps are involved.
      MatrixX Araw = A.block(panelStart, panelStart, m0, m0);

      // Per physical column (offset from panelStart): how many of this
      // panel's pivots-so-far this column's live rows currently reflect
      // (0 == still raw / matches Araw exactly).
      std::vector<int> appliedCount(m0, 0);
      bool panelStoppedEarly = false;

      // Sets column `col`'s live rows [k, n) (and mirrors the matching
      // live entries of row `col`) to *exactly* "raw minus the full
      // correction from this panel's pivot columns [0, p)", computed fresh
      // from `Araw` every time -- never incrementally from whatever `A`
      // currently holds. This is essential once swaps are involved: a
      // swap can relocate a row/column whose *own* prior correction (from
      // when it was some other physical position, e.g. column k's own
      // corrected row was mirrored across every live column at the time)
      // into a position that a *different*, not-yet-touched column later
      // reads -- if that column then applied only an incremental delta
      // (assuming its target was still raw), the portion already baked in
      // via the earlier column's mirror would be double-subtracted. Always
      // recomputing "raw (from the swap-tracked snapshot) minus the full
      // rank-p correction" is idempotent: two different columns computing
      // a shared entry (e.g. column i's row at column j, and column j's
      // row at column i) both derive it from the same Araw/Lbuf/Dbuf state
      // via the same symmetric formula, so whichever computes it first or
      // last, the result is identical (up to ordinary floating-point
      // reassociation) -- no double counting is possible.
      auto setColumnRaw = [&](int col, int p) {
        const int idx = col - panelStart;
        const int rowsFrom = k - panelStart;
        const int rowsCount = m0 - rowsFrom;  // == n - k
        if (rowsCount <= 0) {
          appliedCount[idx] = p;
          return;
        }
        if (p == 0) {
          A.col(col).segment(k, rowsCount) = Araw.col(idx).segment(rowsFrom, rowsCount);
        } else {
          // Associate as (L * D) * l_col rather than L * (D * l_col): this
          // matches the left-to-right evaluation Eigen already performs for
          // the deferred blocked update below (`Lpanel * Dpanel *
          // Lpanel.transpose()`), so a column's JIT-recomputed value and the
          // eventual bulk update agree bit-for-bit in their rank-p term's
          // internal grouping, minimizing (not eliminating -- some
          // reassociation relative to the fully unblocked algorithm's
          // rank-1-at-a-time updates is unavoidable once more than one
          // pivot is grouped into a single correction) floating-point
          // reassociation drift on ill-conditioned/near-singular fronts.
          MatrixX Dp = Dbuf.head(p).toDense();
          MatrixX LD = Lbuf.block(rowsFrom, 0, rowsCount, p) * Dp;
          VectorX Lrow = Lbuf.row(idx).segment(0, p).transpose();
          VectorX corr = LD * Lrow;
          A.col(col).segment(k, rowsCount) = Araw.col(idx).segment(rowsFrom, rowsCount) - corr;
        }
        A.row(col).segment(k, rowsCount) = A.col(col).segment(k, rowsCount).transpose();
        appliedCount[idx] = p;
      };

      // Brings column `col`'s live rows up to date with all `p` pivots
      // accumulated so far in this panel -- a no-op if it already is
      // (tracked by `appliedCount`). This is what lets the pivot-search
      // decisions below see exactly the numbers the fully unblocked
      // algorithm would compute, without paying for a whole-trailing-
      // matrix update on every single pivot: only the O(panel_size)
      // columns actually touched during this panel's search pay any cost
      // at all, and each such touch costs O(m * p) rather than O(m^2).
      auto ensureColumnCurrent = [&](int col, int p) {
        const int idx = col - panelStart;
        if (appliedCount[idx] == p) return;
        setColumnRaw(col, p);
      };

      // Full row/column swap (same physical effect as the original
      // `swap_rowcol`) that additionally keeps this panel's L buffer,
      // raw-snapshot, and applied-correction bookkeeping attached to
      // whichever physical row/column they belong to.
      auto panelSwap = [&](int i, int j) {
        if (i == j) return;
        swap_rowcol(i, j);
        const int ii = i - panelStart, jj = j - panelStart;
        Lbuf.row(ii).swap(Lbuf.row(jj));
        Araw.row(ii).swap(Araw.row(jj));
        Araw.col(ii).swap(Araw.col(jj));
        std::swap(appliedCount[ii], appliedCount[jj]);
      };

      int p = 0;  // pivot *columns* accumulated in this panel so far (always == k - panelStart)
      while (k < panelEnd) {
        const int m = n - k;  // trailing submatrix size (full, incl. ineligible rows)

        // Column k must reflect this panel's pivots-so-far before its
        // entries are searched/read below (mirrors what the unblocked
        // algorithm guarantees by having already updated the *entire*
        // trailing matrix after every prior pivot).
        ensureColumnCurrent(k, p);

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
            panelStoppedEarly = true;  // delay column k (and everything after) to the parent
            break;
          }
          accept_1x1_at_k = true;
        } else if (std::abs(A(k, k)) >= alpha * lambda) {
          accept_1x1_at_k = true;
        } else {
          // Column r must be brought current before it's read (sigma test,
          // A(r,r), or -- if chosen below -- as an actual pivot column):
          // it may lie anywhere in [k, effEnd), not necessarily inside this
          // panel's own column range.
          ensureColumnCurrent(r, p);

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
          Dbuf.diag(p) = d;
          result.pivots.push_back({k, PivotKind::OneByOne});
          if (d > 0)
            ++result.inertia.n_pos;
          else if (d < 0)
            ++result.inertia.n_neg;
          else
            ++result.inertia.n_zero;

          if (m > 1) {
            running_max = std::max(running_max, A.block(k + 1, k, m - 1, 1).cwiseAbs().maxCoeff());
            VectorX l = A.col(k).segment(k + 1, m - 1) / d;
            Lbuf.col(p).segment(k + 1 - panelStart, m - 1) = l;
            A.col(k).segment(k + 1, m - 1) = l;
            A.row(k).segment(k + 1, m - 1) = l.transpose();
          }
          ++k;
          ++p;
          continue;
        }

        if (accept_1x1_swap_kr) {
          panelSwap(k, r);
          running_max = std::max(running_max, std::abs(A(k, k)));
          const double d = A(k, k);
          D_out(k, k) = d;
          Dbuf.diag(p) = d;
          result.pivots.push_back({k, PivotKind::OneByOne});
          if (d > 0)
            ++result.inertia.n_pos;
          else if (d < 0)
            ++result.inertia.n_neg;
          else
            ++result.inertia.n_zero;

          if (m > 1) {
            running_max = std::max(running_max, A.block(k + 1, k, m - 1, 1).cwiseAbs().maxCoeff());
            VectorX l = A.col(k).segment(k + 1, m - 1) / d;
            Lbuf.col(p).segment(k + 1 - panelStart, m - 1) = l;
            A.col(k).segment(k + 1, m - 1) = l;
            A.row(k).segment(k + 1, m - 1) = l.transpose();
          }
          ++k;
          ++p;
          continue;
        }

        // accept_2x2: bring r into position k+1 (k is already fine as-is).
        // The swap happens before any panel-local correction is applied for
        // this pivot, exactly mirroring the unblocked code's ordering --
        // `r`'s already-applied partial correction (from ensureColumnCurrent
        // above) simply moves with it via panelSwap.
        {
          if (r != k + 1) panelSwap(k + 1, r);
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
            panelStoppedEarly = true;
            break;
          }

          running_max = std::max(running_max, std::max({std::abs(d11), std::abs(d21), std::abs(d22)}));

          D_out(k, k) = d11;
          D_out(k + 1, k) = d21;
          D_out(k, k + 1) = d21;
          D_out(k + 1, k + 1) = d22;
          Dbuf.diag(p) = d11;
          Dbuf.offdiag(p) = d21;
          Dbuf.diag(p + 1) = d22;
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

            Lbuf.col(p).segment(k + 2 - panelStart, mm) = L2.col(0);
            Lbuf.col(p + 1).segment(k + 2 - panelStart, mm) = L2.col(1);

            A.block(k + 2, k, mm, 2) = L2;
            A.block(k, k + 2, 2, mm) = L2.transpose();
          }
          k += 2;
          p += 2;
          continue;
        }
      }

      // ==================== End of panel ====================
      // Revert any column in the live trailing region [k, n) that still
      // carries a *partial* panel-local correction (examined as an `r`
      // candidate during pivot search but never finalized as a pivot) back
      // to its pre-panel ("raw", i.e. matching `Araw`) state, then apply
      // exactly one blocked rank-p update to the entire remaining trailing
      // extent at once -- this is the BLAS-3-throughput GEMM step the whole
      // scheme exists to enable. See the class-level comment above
      // `factor()` for why this is mathematically equivalent to what the
      // unblocked algorithm would have produced (up to legitimate
      // floating-point reassociation).
      if (p > 0) {
        for (int col = k; col < n; ++col) {
          const int idx = col - panelStart;
          if (appliedCount[idx] != 0) setColumnRaw(col, 0);
        }

        const int rowsCount = n - k;
        if (rowsCount > 0) {
          auto Lpanel = Lbuf.block(k - panelStart, 0, rowsCount, p);
          MatrixX Dpanel = Dbuf.head(p).toDense();
          auto trailingBeyond = A.block(k, k, rowsCount, rowsCount);
          trailingBeyond.noalias() -= Lpanel * Dpanel * Lpanel.transpose();
          resymmetrizeTrailing(trailingBeyond);
          running_max = std::max(running_max, trailingBeyond.cwiseAbs().maxCoeff());
        }
      }

      if (panelStoppedEarly) break;
      // else: continue the outer loop -- k has already advanced to panelEnd
      // (or panelEnd + 1, in the rare case where a 2x2 pivot straddled the
      // panel boundary).
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

  // Phase 7: KKT-aware *static* pivoting (PARDISO-SBK-style / Vanderbei
  // SQD-theory-style). Unlike `factor()` above, this performs **no pivot
  // search and no row/column swaps at all**: columns 0..n_eligible-1 are
  // taken strictly in the given order, always as 1x1 pivots.
  //
  // Rationale (Vanderbei 1995, "Symmetric Quasidefinite Matrices"): a
  // symmetric quasidefinite matrix K = [[-E, A^T], [A, F]] with E, F
  // symmetric positive definite admits an LDL^T factorization (all 1x1
  // pivots, no zero/near-zero pivots) for *any* symmetric permutation of
  // its rows/columns, not just a specially chosen one -- so once the
  // caller has arranged (via `expectedSign`) which pivots should come out
  // negative (the -E block) vs. positive (the F block), pure diagonal
  // pivoting in a fixed, sparsity-driven order (e.g. AMD/METIS, applied
  // without regard to numerical stability) is provably safe for an exactly
  // quasidefinite input. Real KKT systems from interior-point methods are
  // usually only *near*-quasidefinite (e.g. a barrier term driving one
  // diagonal block toward singular, or an exactly-zero block in an
  // equality-constrained QP), so this function also supports diagonal
  // regularization (Gill/Saunders/Shinnerl 1996; Wachter & Biegler 2006
  // IPOPT inertia control; PARDISO static-pivoting + diagonal perturbation,
  // Schenk & Gaertner): whenever a pivot's magnitude is below a floor, or
  // its sign disagrees with the caller-supplied expectation, this adds
  // `delta * expected_sign` to the diagonal before using it -- i.e. it
  // factors A + Delta for some (tracked) diagonal perturbation Delta, not
  // exactly A. The caller (solver.hpp's inertia-controlled retry loop) is
  // responsible for checking the resulting inertia and escalating `delta`
  // if it doesn't match expectations, and `refine.hpp`'s iterative
  // refinement is responsible for recovering the accuracy lost to Delta
  // against the true (unperturbed) system.
  //
  // `expectedSign` is indexed exactly like the front-local physical
  // position (no permutation ever happens in this function, so "front-local
  // physical position" and "front-local column index" are the same thing
  // throughout): expectedSign[k] in {-1, 0, +1}; 0 means "no expectation for
  // this column" (perturb only if the raw magnitude is below the floor,
  // pushing towards whatever sign the unperturbed pivot already had, or +1
  // for an exact zero). A short/empty vector is treated as all-zero
  // (unconstrained) for the remaining columns.
  //
  // Because every eligible column is always finalized here (perturbation
  // guarantees an acceptable pivot always exists), `n_factored` is always
  // exactly `n_eligible` and `delayed_cols`/`perm` are always empty/identity
  // -- static pivoting never delays a column to the parent front, by
  // construction (see multifrontal.hpp's Phase 7 notes on why this
  // simplifies the driver's bookkeeping).
  //
  // Phase 9: panel-blocked, delayed-trailing-update, exactly like `factor()`
  // (see that function's header comment for the general BLAS-3-throughput
  // motivation) but *substantially* simpler, because static pivoting has
  // neither row/column swaps nor 2x2 pivots: every column is examined
  // exactly once, in fixed physical order, and finalized unconditionally the
  // moment it's examined (never held as a tentative "candidate partner" that
  // might later be discarded, unlike `factor()`'s `r`). Consequently there
  // is no need for `factor()`'s `appliedCount`/"touched but not finalized,
  // revert to raw" bookkeeping at all: within a panel spanning
  // `[panelStart, panelEnd)`, column `k`'s up-to-date diagonal and
  // below-diagonal entries are always exactly
  //   raw(k) - Lbuf.block(k-panelStart, 0, ..., p) * Dbuf.head(p) * Lbuf.row(k-panelStart, 0, p)^T
  // where `p = k - panelStart` is the number of this panel's pivots already
  // finalized -- a single fresh computation against a per-panel raw
  // snapshot (`Araw`, only `m0 x panelWidth`, far smaller than `factor()`'s
  // `m0 x m0` snapshot since there is nothing to swap into it), performed
  // exactly once per column, immediately before that column is used as the
  // pivot. Once a panel's columns are all finalized, one blocked GEMM
  // (`Lpanel * Dpanel * Lpanel^T`, `Dpanel` a pure diagonal since every
  // static pivot is 1x1) applies the panel's combined correction to the
  // entire trailing extent beyond the panel, followed by one resymmetrize --
  // mirroring `factor()`'s reasoning for why resymmetrization timing (once
  // per panel, not once per column) matters numerically, verified for this
  // function too via a standalone head-to-head sweep across panel_size in
  // {1,2,4,...,huge} on random SPD/quasidefinite/KKT-shaped dense matrices
  // (inertia, perturbation counts, and reconstruction residual all agree
  // closely across panel sizes), plus this project's existing `ctest` suite
  // (`kkt_static_pivoting_test.cpp`, `dense_kernel_test.cpp`, and the
  // multifrontal correctness/real-matrix tests, which exercise this
  // function at its new default `panel_size = 64` on fronts both above and
  // below that width).
  // `options.panel_size >= n_eligible` collapses this to exactly one panel
  // spanning the whole eligible range -- mathematically equivalent to the
  // original fully unblocked algorithm.
  static DenseLDLTResult factorStatic(Eigen::Ref<MatrixX> A, BlockDiagonalD<Scalar>& D_out,
                                       const StaticPivotOptions& options, const std::vector<int>& expectedSign,
                                       int n_eligible = -1) {
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n) throw std::invalid_argument("DenseLDLT::factorStatic: A must be square");
    const int effEnd = (n_eligible < 0) ? n : n_eligible;
    if (effEnd < 0 || effEnd > n) throw std::invalid_argument("DenseLDLT::factorStatic: n_eligible out of range");

    DenseLDLTResult result;
    result.perm = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
    D_out.resize(n);

    for (int j = 0; j < n; ++j)
      for (int i = j + 1; i < n; ++i) A(j, i) = A(i, j);

    double orig_max = A.cwiseAbs().maxCoeff();
    if (orig_max < options.absolute_floor) orig_max = 1.0;
    double running_max = orig_max;

    const double delta = options.delta > 0.0 ? options.delta : std::sqrt(std::numeric_limits<double>::epsilon()) * orig_max;
    const double floor = std::max(options.absolute_floor, options.relative_pivot_floor * orig_max);

    const int panelSizeOpt = options.panel_size > 0 ? options.panel_size : 64;

    int k = 0;
    while (k < effEnd) {
      const int panelStart = k;
      const int panelWidth = std::min(panelSizeOpt, effEnd - panelStart);
      const int panelEnd = panelStart + panelWidth;
      const int m0 = n - panelStart;  // rows spanned by this panel's L buffer (down to n)

      // Raw snapshot of just this panel's own columns (only m0 x panelWidth,
      // not m0 x m0 as `factor()` needs -- there is nothing to swap into
      // this snapshot, so it never needs to cover columns outside the
      // panel). Row offset r (0-based from panelStart) of column-offset c
      // holds A(panelStart + r, panelStart + c) as of the start of this
      // panel (i.e. reflecting every prior panel's effect, none of this
      // one's).
      MatrixX Araw = A.block(panelStart, panelStart, m0, panelWidth);

      MatrixX Lbuf = MatrixX::Zero(m0, panelWidth);
      VectorX Dbuf(panelWidth);

      for (; k < panelEnd; ++k) {
        const int idx = k - panelStart;         // this column's offset within the panel
        const int rowsCount = m0 - idx;          // == n - k

        // Column k's up-to-date (diagonal-and-below) entries: raw minus the
        // correction from this panel's `idx` pivots finalized so far,
        // computed fresh from `Araw`/`Lbuf`/`Dbuf` every time -- see the
        // class-level comment above this function for why no incremental
        // update-and-revert bookkeeping is needed here (unlike `factor()`):
        // static pivoting never swaps, and a column is only ever touched
        // once, exactly when it becomes pivot k.
        VectorX col;
        if (idx == 0) {
          col = Araw.col(0).segment(0, rowsCount);
        } else {
          VectorX Lrow = Lbuf.row(idx).segment(0, idx).transpose();
          VectorX Dl = Lrow.cwiseProduct(Dbuf.segment(0, idx));
          col = Araw.col(idx).segment(idx, rowsCount) - Lbuf.block(idx, 0, rowsCount, idx) * Dl;
        }

        const double a_kk = col(0);
        const int sign = (k < static_cast<int>(expectedSign.size())) ? expectedSign[k] : 0;

        double d = a_kk;
        double applied = 0.0;
        bool needPerturb = false;
        if (sign > 0) {
          needPerturb = a_kk < floor;
        } else if (sign < 0) {
          needPerturb = a_kk > -floor;
        } else {
          needPerturb = std::abs(a_kk) < floor;
        }

        if (needPerturb) {
          // IPOPT/PARDISO-SBK-style: add a single sign-matched perturbation
          // of magnitude `delta` and use whatever pivot that produces --
          // deliberately NOT forced/looped to guarantee an acceptable
          // magnitude or sign within this single pass. If `delta` isn't large
          // enough (e.g. a_kk had the wrong sign by more than `delta`), the
          // resulting pivot may still be small or wrong-signed; that is
          // exactly the case the inertia-controlled retry loop in
          // solver.hpp's `factorizeStaticRegularized()` is meant to detect
          // (via a mismatched `inertia()` afterwards) and correct by
          // re-factoring from scratch with `delta` escalated, not something
          // this single-pass function silently papers over.
          const double s = (sign != 0) ? static_cast<double>(sign) : (a_kk >= 0.0 ? 1.0 : -1.0);
          d = a_kk + s * delta;
          applied = d - a_kk;
        }

        D_out(k, k) = d;
        Dbuf(idx) = d;
        result.pivots.push_back({k, PivotKind::OneByOne});
        if (applied != 0.0) {
          result.total_perturbation += std::abs(applied);
          ++result.num_perturbed;
        }
        if (d > 0)
          ++result.inertia.n_pos;
        else if (d < 0)
          ++result.inertia.n_neg;
        else
          ++result.inertia.n_zero;

        running_max = std::max(running_max, std::abs(d));
        if (rowsCount > 1) {
          running_max = std::max(running_max, col.segment(1, rowsCount - 1).cwiseAbs().maxCoeff());
          VectorX l = col.segment(1, rowsCount - 1) / d;
          Lbuf.col(idx).segment(idx + 1, rowsCount - 1) = l;
          A.col(k).segment(k + 1, rowsCount - 1) = l;
          A.row(k).segment(k + 1, rowsCount - 1) = l.transpose();
        }
      }

      // ==================== End of panel ====================
      // One blocked rank-p update (Lpanel * Dpanel * Lpanel^T, Dpanel purely
      // diagonal since every static pivot is 1x1) applied to the entire
      // trailing extent beyond the panel, via genuine Eigen block GEMM --
      // this is the BLAS-3-throughput step the whole scheme exists to
      // enable -- followed by one resymmetrize (mirrors `factor()`'s timing:
      // once per panel, not once per column; see this function's
      // class-level comment above).
      const int rowsBeyond = n - panelEnd;
      if (rowsBeyond > 0) {
        auto Lpanel = Lbuf.block(panelEnd - panelStart, 0, rowsBeyond, panelWidth);
        MatrixX LD = Lpanel * Dbuf.asDiagonal();
        auto trailingBeyond = A.block(panelEnd, panelEnd, rowsBeyond, rowsBeyond);
        trailingBeyond.noalias() -= LD * Lpanel.transpose();
        resymmetrizeTrailingTiled(trailingBeyond);
        running_max = std::max(running_max, trailingBeyond.cwiseAbs().maxCoeff());
      }
    }

    result.n_factored = effEnd;
    result.max_growth = running_max / orig_max;
    return result;
  }
};

}  // namespace symla
