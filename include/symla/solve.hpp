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
//
// --- Task-DAG parallel solve (mirrors multifrontal.hpp's Phase 6 approach) ---
//
// The diagonal solve (step 3) is already embarrassingly parallel (every
// pivot block is independent) but cheap (O(n)) and left serial here -- not
// worth the scheduling complexity.
//
// The backward solve (step 4) turns out to already be race-free under a
// top-down (parent-before-children) fan-out with no restructuring: a
// front's "own" rows (`rowIndices[0..nPivots)`) are written by exactly one
// front each (disjoint across the whole tree, see NumericFactor::fronts),
// and its "extra"/ancestor rows are only ever *read*, and only after the
// owning ancestor front has already been processed -- which a top-down
// traversal (`parallel::runTaskDagTopDown`) guarantees by construction (a
// node's `process` call happens strictly before any of its descendants').
// So the exact same per-front computation as the serial loop below is
// reused unchanged, just driven by a different traversal.
//
// The forward solve (step 2) is the one that needs real restructuring: the
// serial version's `Y.row(extra) -= upd.row(i)` is a scatter-subtract into
// a *shared* array, mutated by every front that has that row as an
// "extra"/ancestor row -- an actual data race under concurrent execution
// with no synchronization. The parallel version below instead has each
// front compute a *private* contribution over its own "extra" rows
// (`RhsElement`, directly analogous to multifrontal.hpp's per-node
// `GeneratedElement` for the K-matrix extend-add), consumed by its
// immediate parent via the same O(1)-lookup index-mapping merge
// multifrontal.hpp uses for the K-matrix. Note this genuinely does need to
// be a *sum*, not a plain overwrite: unlike the backward solve's disjoint
// "own rows", a single shared ancestor row can receive independent
// contributions from *multiple sibling subtrees* -- the classic
// multifrontal "fan-in" (e.g. two different children can each have one of
// a common ancestor's native columns as a genuine row-pattern entry; the
// exact same reason the K-matrix's own Schur-complement assembly needs
// extend-*add* rather than extend-*assign*). `RhsElement` therefore carries
// a pure *cumulative subtraction delta* (never a "current value"),
// initialized to zero and only ever combined with the immutable `Bp` once,
// at the point a row is finally gathered as some front's *own* row --
// exactly mirroring the serial algorithm's single shared `Y` array (which
// starts as `Bp` and only ever receives `-=` updates from every front that
// has a given row as one of its "extra" rows, regardless of tree
// distance). Concretely, per front:
//
//   1. Gather this front's own `m` rows' pending cumulative delta by
//      summing (`+=`) every child's forwarded delta for the rows they
//      share with this front (an O(1)-lookup index-mapping merge, same
//      mechanism as multifrontal.hpp's extend-add).
//   2. For the own-pivot subset, combine with the immutable `Bp` (`Bp -
//      delta`) to get the actual gathered RHS, do the small in-front
//      triangular solve, and write the *final* y values directly into a
//      shared `Y` array by global row index -- safe as a direct
//      (unsynchronized) write, since -- symmetric to the backward solve's
//      argument -- every front's own rows are disjoint across the whole
//      tree.
//   3. For the "extra" rows, add this front's own newly-finalized pivots'
//      local elimination effect (`L.bottomRows(extra) * yOwn`, same
//      quantity the serial version subtracts) on top of the cumulative
//      delta from step 1, and forward the result as this front's own
//      `RhsElement` to its parent.
//
// Root fronts' "extra" rows (should be empty for a non-singular
// factorization, see the SupernodeFactor::rowIndices docs) are simply never
// read by anyone -- there is no parent to consume them.

#include "symla/dense_kernel.hpp"
#include "symla/multifrontal.hpp"
#include "symla/parallel/task_graph.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifdef SYMLA_HAVE_OPENMP
#include <omp.h>
#endif

namespace symla {

// Options controlling task-DAG parallel execution of `MultifrontalSolver::
// solve()` over the same supernode tree `factorize()` used (see
// NumericFactor::parentSN/childrenSN/rootsSN). Mirrors
// `MultifrontalOptions` (multifrontal.hpp) in spirit, but solve()'s
// per-front cost is much smaller than factorize()'s (a triangular solve
// plus a couple of O(m) merges on an `m x nrhs` block, vs. factorize()'s
// `O(ncols * m^2)` dense kernel), so it gets its own, separately-calibrated
// cost model and cutoffs rather than reusing MultifrontalOptions' numeric
// constants (see the calibration comment in `solve()` below).
struct SolveOptions {
  bool parallel = false;
  int num_threads = 0;  // 0 => omp_get_max_threads()
  // Absolute task-spawn cutoff override, in this file's own cost units (see
  // below); negative (the default) means "derive automatically from the
  // tree's total estimated cost and thread count", same policy as
  // MultifrontalOptions::task_cutoff.
  long long task_cutoff = -1;
};

template <typename Scalar>
class MultifrontalSolver {
 public:
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixXd = Eigen::MatrixXd;

  static DenseMatrix solve(const NumericFactor& nf, const DenseMatrix& B, const SolveOptions& solveOptions = {}) {
    const int n = nf.n;
    if (static_cast<int>(B.rows()) != n) {
      throw std::invalid_argument("symla::MultifrontalSolver::solve: B.rows() must match the factorized matrix size");
    }
    const int nrhs = static_cast<int>(B.cols());
    if (n == 0) return DenseMatrix(0, nrhs);

    // Step 1: permute B into final order, in double precision. Read-only
    // from here on (`Bp`), safe to share across threads.
    MatrixXd Bp(n, nrhs);
    for (int t = 0; t < n; ++t) Bp.row(t) = B.row(nf.perm[t]).template cast<double>();

    const int numSN = static_cast<int>(nf.fronts.size());

#ifdef SYMLA_HAVE_OPENMP
    const int maxThreads = solveOptions.parallel
                                ? std::max(1, solveOptions.num_threads > 0 ? solveOptions.num_threads
                                                                           : omp_get_max_threads())
                                : 1;
#else
    const int maxThreads = 1;
#endif

    // Subtree-cost estimate for the task-spawn-vs-inline cutoff, in units
    // scaled for *this* solve's nrhs (unlike MultifrontalOptions'
    // factorize()-time cutoff, this is necessarily computed fresh per
    // solve() call, since it depends on nrhs). Per-front cost is
    // `m * max(1,k) * nrhs`-ish (dominant term of the triangular solve plus
    // the extend-add merge/update on an `m x nrhs` block) -- much smaller
    // than factorize()'s `ncols * m^2` per front, so the constants below
    // are calibrated independently (see the empirical notes at each
    // constant), not copied from multifrontal.hpp's factorize() cutoff.
    std::vector<long long> subtreeWeight(numSN, 0);
    for (int si = 0; si < numSN; ++si) {
      const auto& f = nf.fronts[si];
      const long long m = static_cast<long long>(f.rowIndices.size());
      const long long k = std::max<long long>(1, f.nPivots);
      const long long ownCost = std::max<long long>(1, m * k) * std::max(1, nrhs);
      subtreeWeight[si] += ownCost;
      const int p = nf.parentSN[si];
      if (p != -1) subtreeWeight[p] += subtreeWeight[si];
    }
    long long totalCost = 0;
    for (int r : nf.rootsSN) totalCost += subtreeWeight[r];

    bool runParallel = false;
    long long cutoff = 0;
    int requestedThreads = 1;
#ifdef SYMLA_HAVE_OPENMP
    if (solveOptions.parallel && numSN > 0) {
      // Same overall policy as factorize()'s auto-cutoff derivation
      // (multifrontal.hpp), but with a separately-calibrated absolute
      // floor: solve()'s per-front work is orders of magnitude cheaper than
      // factorize()'s dense kernel, so the same node clears "worth a task"
      // at a different cost-unit value. Calibrated empirically on this
      // machine (see bench/solve_parallel_diag_main.cpp) against: (a) real
      // GHS_indef/qpband (~1442 mostly-tiny supernodes, the exact matrix
      // that caused the original factorize() Task 13 scheduling-
      // granularity regression) at nrhs in {1, 10} across thread counts
      // 1-16 -- totalCost there is ~460K (nrhs=1) to ~800K (nrhs=10) units,
      // and must stay *below* the bypass floor or parallel solve()
      // regresses by ~150x (all task-spawn overhead, ~1.2ms of genuine
      // work spread wrongly thin); (b) a synthetic block-arrow matrix (16
      // independent ~220x220 dense blocks feeding a small shared coupling
      // front -- deliberately shaped for genuine sibling-subtree
      // parallelism) at nrhs in {1, 10, 50} -- each sibling subtree is
      // ~2.4M units at nrhs=50 and must clear the *per-node* cutoff or no
      // task ever gets spawned and the parallel path is pure overhead. The
      // gap between qpband's ~460K-800K (must stay serial) and
      // block-arrow's ~2.4M-per-sibling (must parallelize) sets the floor.
      constexpr long long kCostPerThread = 2'000'000;
      int effectiveThreads = maxThreads;
      if (solveOptions.num_threads <= 0) {
        const long long threadsFromCost = totalCost / kCostPerThread;
        effectiveThreads = static_cast<int>(std::min<long long>(maxThreads, std::max<long long>(1, threadsFromCost)));
      }

      cutoff = solveOptions.task_cutoff;
      if (cutoff < 0) {
        constexpr long long kTargetTasksPerThread = 4;
        const long long targetTasks = std::max<long long>(1, kTargetTasksPerThread * effectiveThreads);
        cutoff = totalCost > 0 ? std::max<long long>(1, totalCost / targetTasks) : 1;
        // Absolute floor: below this a front's real solve-step cost is
        // microseconds, too small for any thread count to amortize
        // OpenMP's per-task overhead against.
        constexpr long long kMinTaskCost = 1'000'000;
        cutoff = std::max(cutoff, kMinTaskCost);
      }

      // Global bypass: skip opening a parallel region entirely unless doing
      // so can plausibly spawn at least one substantial *concurrent* task.
      // A large *aggregate* totalCost is not by itself sufficient evidence
      // of that -- e.g. a tree with many small-to-medium fronts can have a
      // large totalCost while no single subtree individually clears
      // `cutoff`, in which case every child is executed inline by the one
      // thread that opened the region and the parallel path is pure
      // overhead (empirically ~5-10x slower than serial in this situation,
      // observed while calibrating against a synthetic block-arrow matrix
      // at nrhs=10, see bench/solve_parallel_diag_main.cpp). So the real
      // bypass criterion is: does *some* subtree, considered individually,
      // clear `cutoff`? `runTaskDag`/`runTaskDagTopDown` only ever spawn a
      // task for (a) a node with a parent whose own `subtreeWeight` clears
      // `cutoff`, or (b) unconditionally for every root when there is more
      // than one root (multiple roots run concurrently regardless of
      // `cutoff`) -- `maxSpawnable` below is the largest weight among
      // exactly those candidates.
      long long maxSpawnable = 0;
      for (int si = 0; si < numSN; ++si) {
        if (nf.parentSN[si] != -1) maxSpawnable = std::max(maxSpawnable, subtreeWeight[si]);
      }
      if (nf.rootsSN.size() >= 2) {
        for (int r : nf.rootsSN) maxSpawnable = std::max(maxSpawnable, subtreeWeight[r]);
      }
      constexpr long long kMinTotalCostForParallel = 1'000'000 * 4;
      if ((totalCost >= kMinTotalCostForParallel && maxSpawnable >= cutoff) || solveOptions.task_cutoff >= 0) {
        runParallel = true;
        requestedThreads = solveOptions.num_threads > 0 ? solveOptions.num_threads : effectiveThreads;
      }
      if (std::getenv("SYMLA_DEBUG_SCHED")) {
        std::fprintf(stderr,
                      "SYMLA_DEBUG_SCHED[solve] numSN=%d nrhs=%d maxThreads=%d effectiveThreads=%d totalCost=%lld "
                      "cutoff=%lld runParallel=%d requestedThreads=%d\n",
                      numSN, nrhs, maxThreads, effectiveThreads, totalCost, cutoff, runParallel, requestedThreads);
      }
    }
#endif

    MatrixXd Y(n, nrhs);
    MatrixXd Z(n, nrhs);
    MatrixXd X(n, nrhs);

    // One private "generated element"-style RHS contribution per front,
    // consumed exactly once by its parent (see the header comment above).
    struct RhsElement {
      std::vector<int> indices;
      MatrixXd matrix;
    };
    std::vector<RhsElement> rhsElem(numSN);

    if (runParallel) {
#ifdef SYMLA_HAVE_OPENMP
      std::vector<std::vector<int>> globalToLocalTls(std::max(1, requestedThreads), std::vector<int>(n, -1));

      auto forwardProcess = [&](int si) {
        const int tid = omp_get_thread_num();
        std::vector<int>& globalToLocal = globalToLocalTls[tid];
        const auto& f = nf.fronts[si];
        const int m = static_cast<int>(f.rowIndices.size());
        const int k = f.nPivots;
        if (m == 0) return;

        // `G` accumulates the *cumulative subtraction delta* owed against
        // this front's own `m` rows -- NOT a "current value" -- because a
        // shared ancestor row can receive independent contributions from
        // *multiple sibling subtrees* (the classic multifrontal "fan-in":
        // e.g. two different children can each have one of this front's
        // own native columns as a genuine row-pattern entry, exactly the
        // same reason the K-matrix Schur-complement assembly needs
        // extend-*add* rather than extend-*assign*). So every child's
        // forwarded delta for a shared row must be *summed* (`+=`), not
        // overwritten, and initialized to zero (not to `Bp`) -- the raw RHS
        // value is only ever combined in once, at the point a row is
        // finally gathered as some front's *own* row below, exactly
        // mirroring the serial algorithm's single shared `Y` array (which
        // starts as `Bp` and only ever receives `-=` updates from every
        // front that has a given row as one of its "extra" rows,
        // regardless of tree distance).
        MatrixXd G = MatrixXd::Zero(m, nrhs);
        for (int t = 0; t < m; ++t) globalToLocal[f.rowIndices[t]] = t;
        for (int ci : nf.childrenSN[si]) {
          RhsElement& ce = rhsElem[ci];
          const int p = static_cast<int>(ce.indices.size());
          for (int a = 0; a < p; ++a) {
            const int li = globalToLocal[ce.indices[a]];
            G.row(li) += ce.matrix.row(a);
          }
          ce.matrix.resize(0, 0);
          ce.indices.clear();
          ce.indices.shrink_to_fit();
        }
        for (int t = 0; t < m; ++t) globalToLocal[f.rowIndices[t]] = -1;

        MatrixXd yOwn(k, nrhs);
        if (k > 0) {
          for (int p = 0; p < k; ++p) yOwn.row(p) = Bp.row(f.rowIndices[p]) - G.row(p);
          MatrixXd Ltop = maskedTopBlock(f);
          Ltop.template triangularView<Eigen::UnitLower>().solveInPlace(yOwn);
          for (int p = 0; p < k; ++p) Y.row(f.rowIndices[p]) = yOwn.row(p);
        }

        const int extra = m - k;
        if (extra > 0) {
          // Forward the cumulative delta owed against these rows so far
          // (from everything below, already summed above) *plus* this
          // front's own newly-finalized pivots' contribution -- the parent
          // (or further ancestor, transitively) will keep accumulating on
          // top of this via the same `+=` merge, exactly matching the
          // serial algorithm's running `-=` into a single shared array.
          MatrixXd newExtra = G.bottomRows(extra);
          if (k > 0) newExtra.noalias() += f.L.bottomRows(extra) * yOwn;
          rhsElem[si].indices.assign(f.rowIndices.begin() + k, f.rowIndices.end());
          rhsElem[si].matrix = std::move(newExtra);
        }
      };

      auto backwardProcess = [&](int si) {
        const auto& f = nf.fronts[si];
        const int m = static_cast<int>(f.rowIndices.size());
        const int k = f.nPivots;
        if (k == 0) return;

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
      };

      // Same oversubscription-avoidance / nested-parallelism guarding as
      // multifrontal.hpp's factorize() (see its comments for the full
      // rationale): pin Eigen to single-threaded GEMM and cap OpenMP
      // nesting to 1 active level for the duration of this call.
      const int savedEigenThreads = Eigen::nbThreads();
      Eigen::setNbThreads(1);
      const int savedMaxActiveLevels = omp_get_max_active_levels();
      omp_set_max_active_levels(1);

#pragma omp parallel num_threads(requestedThreads)
      {
#pragma omp single
        {
          parallel::runTaskDag(nf.rootsSN, nf.childrenSN, subtreeWeight, cutoff, forwardProcess);
        }
      }

      // Step 3: block-diagonal solve (cheap, O(n); left serial, see the
      // header comment above).
      diagonalSolve(nf, Y, Z);

#pragma omp parallel num_threads(requestedThreads)
      {
#pragma omp single
        {
          parallel::runTaskDagTopDown(nf.rootsSN, nf.childrenSN, subtreeWeight, cutoff, backwardProcess);
        }
      }

      omp_set_max_active_levels(savedMaxActiveLevels);
      Eigen::setNbThreads(savedEigenThreads);
#endif
    } else {
      // Serial fallback (parallel disabled, no OpenMP, or the tree's
      // estimated total cost is too small to bother): the exact original
      // Phase 4 algorithm, one left-to-right / right-to-left pass over
      // `fronts` with a single shared mutable working array.
      forwardSolveSerial(nf, Bp, Y);
      diagonalSolve(nf, Y, Z);
      backwardSolveSerial(nf, Z, X);
    }

    // Step 5: un-permute back to original index space, cast to Scalar.
    DenseMatrix result(n, nrhs);
    for (int t = 0; t < n; ++t) result.row(nf.perm[t]) = X.row(t).template cast<Scalar>();
    return result;
  }

 private:
  static void diagonalSolve(const NumericFactor& nf, const MatrixXd& Y, MatrixXd& Z) {
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
  }

  static void forwardSolveSerial(const NumericFactor& nf, const MatrixXd& Bp, MatrixXd& Y) {
    const int n = nf.n;
    const int nrhs = static_cast<int>(Bp.cols());
    MatrixXd Ywork = Bp;  // shared, mutable working copy, exactly like the original algorithm
    for (const auto& f : nf.fronts) {
      const int m = static_cast<int>(f.rowIndices.size());
      const int k = f.nPivots;
      if (k == 0) continue;

      MatrixXd yOwn(k, nrhs);
      for (int p = 0; p < k; ++p) yOwn.row(p) = Ywork.row(f.rowIndices[p]);

      MatrixXd Ltop = maskedTopBlock(f);
      Ltop.template triangularView<Eigen::UnitLower>().solveInPlace(yOwn);

      for (int p = 0; p < k; ++p) Ywork.row(f.rowIndices[p]) = yOwn.row(p);

      const int extra = m - k;
      if (extra > 0) {
        MatrixXd upd = f.L.bottomRows(extra) * yOwn;  // extra x nrhs
        for (int i = 0; i < extra; ++i) Ywork.row(f.rowIndices[k + i]) -= upd.row(i);
      }
    }
    (void)n;
    Y = std::move(Ywork);
  }

  static void backwardSolveSerial(const NumericFactor& nf, const MatrixXd& Z, MatrixXd& X) {
    const int n = nf.n;
    const int nrhs = static_cast<int>(Z.cols());
    MatrixXd Xwork(n, nrhs);
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
        for (int i = 0; i < extra; ++i) xExtra.row(i) = Xwork.row(f.rowIndices[k + i]);
        xOwn.noalias() -= f.L.bottomRows(extra).transpose() * xExtra;
      }

      MatrixXd Ltop = maskedTopBlock(f);
      Ltop.template triangularView<Eigen::UnitLower>().transpose().solveInPlace(xOwn);

      for (int p = 0; p < k; ++p) Xwork.row(f.rowIndices[p]) = xOwn.row(p);
    }
    X = std::move(Xwork);
  }

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
