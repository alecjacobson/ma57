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
#include "symla/parallel/task_graph.hpp"
#include "symla/solver.hpp"
#include "symla/symbolic.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <mutex>
#include <vector>

#ifdef SYMLA_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef SYMLA_PROFILE
#include <chrono>
#include <cstdio>
#endif

namespace symla {

#ifdef SYMLA_PROFILE
struct MultifrontalProfile {
  double extraFromChildrenSec = 0;
  double extendSec = 0;
  double addSec = 0;
  double factorSec = 0;
  long long numFronts = 0;
  long long sumFrontSize = 0;
  long long sumFrontSizeSq = 0;
  double sumEligTimesMsq = 0;
  long long maxFrontSize = 0;
  int maxFrontNEligible = 0;
};
inline MultifrontalProfile g_mfProfile;
#endif

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
  std::vector<SupernodeFactor> fronts;  // fronts[si] == the front for supernode si; since
                                         // symbolic.hpp's postorder-by-construction numbering
                                         // already makes supernode-index order a valid
                                         // processing order (see header comment above),
                                         // "indexed by supernode index" and "in processing
                                         // order" are the same statement, both under the
                                         // Phase 3 serial driver and the Phase 6 parallel one
                                         // below (each front is written to its own
                                         // predetermined slot `fronts[si]`, never appended, so
                                         // out-of-order parallel completion does not disturb
                                         // this ordering).
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

// Phase 6: options controlling task-DAG parallel execution of `factorize()`
// over the supernode tree. See parallel/task_graph.hpp for the scheduling
// primitive this drives.
struct MultifrontalOptions {
  // If false (or if SYMLA_HAVE_OPENMP is not defined), factorize() runs the
  // exact same single-threaded postorder loop Phase 3 always has.
  bool parallel = false;

  // 0 means "use whatever `omp_get_max_threads()` / OMP_NUM_THREADS
  // currently reports"; a positive value pins the parallel region to that
  // many threads for this call via `omp_set_num_threads()`.
  int num_threads = 0;

  // Subtree-size (in supernodes) below which a child subtree is executed
  // inline by the same thread rather than spawned as a separate OpenMP
  // task -- avoids task-creation overhead dominating for the very common
  // case of many tiny leaf supernodes. See parallel::runTaskDag.
  long long task_cutoff = 8;
};

template <typename Scalar>
class MultifrontalFactorizer {
 public:
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

  static NumericFactor factorize(const SymbolicFactor& sf, const SparseMatrix& A,
                                  const DenseLDLTOptions& options = {}, const MultifrontalOptions& mfOptions = {}) {
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
    std::vector<int> rootsSN;
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
    for (int si = 0; si < numSN; ++si) {
      if (parentSN[si] == -1) rootsSN.push_back(si);
    }

    // --- Numeric values of A, permuted into final order, lower triangle only. ---
    auto Aperm = detail::PermutedLower<Scalar, SparseMatrix>::build(A, sf.perm);

    // Generated element ("update matrix") produced by each supernode after
    // its own processing, consumed exactly once by its parent (freed
    // thereafter). Indexed by supernode index; empty/unused entries (roots
    // with nothing left over) simply never get read. Under the Phase 6
    // parallel driver this is still race-free without any locking: slot
    // `si` is written exactly once, by the single task that processes
    // supernode `si`, and is only ever read by supernode `si`'s parent --
    // which the task-DAG scheduler (parallel/task_graph.hpp) guarantees
    // cannot start until an `#pragma omp taskwait` has joined every child
    // task, establishing the happens-before edge this relies on.
    struct GeneratedElement {
      std::vector<int> indices;  // final-order indices, size p
      MatrixX matrix;            // p x p, lower triangle valid (Schur complement)
    };
    std::vector<GeneratedElement> genElem(numSN);

    // Phase 6: `fronts` is pre-sized and each supernode writes only to its
    // own slot `fronts[si]` -- safe for concurrent out-of-order completion
    // without locking (see the NumericFactor::fronts comment above; this is
    // exactly the "pre-sized vector, indexed writes" pattern the Phase 6
    // plan calls for, replacing the old push_back-in-postorder-only
    // approach, which relied on the loop's strict sequential order and
    // would race under concurrent completion).
    nf.fronts.resize(numSN);

    // Per-front inertia contributions and a `singular` critical section:
    // avoids a shared `nf.inertia +=` (a real data race under parallel
    // execution) by having each task write only its own slot, then summing
    // serially once all fronts are done; `nf.singular`/`nf.singularCols`
    // are written extremely rarely (only for a genuinely rank-deficient
    // root) so a plain mutex is sufficient and never contended in the
    // common case.
    std::vector<Inertia> perFrontInertia(numSN);
    std::mutex singularMutex;

    // Phase 6: `globalToLocal` used to be a single array shared across the
    // whole (strictly sequential) loop; under task parallelism, concurrent
    // fronts must not stomp on each other's scratch space, so this becomes
    // one buffer per *thread* (not per front -- reused across all fronts a
    // given thread processes, exactly like the old single shared buffer
    // was reused across all fronts in the serial driver, just now one per
    // worker instead of one globally). Every use below touches only the
    // entries it just set (and resets exactly those before returning), so
    // reuse across fronts on the same thread is safe.
#ifdef SYMLA_HAVE_OPENMP
    const int maxThreads = mfOptions.parallel ? std::max(1, mfOptions.num_threads > 0 ? mfOptions.num_threads
                                                                                       : omp_get_max_threads())
                                               : 1;
#else
    const int maxThreads = 1;
#endif
    std::vector<std::vector<int>> globalToLocalTls(maxThreads, std::vector<int>(n, -1));

    // Subtree weight (in supernodes) for the task-spawn-vs-inline cutoff
    // heuristic; increasing-si order is already a valid postorder (see the
    // header comment), so a single forward pass suffices.
    std::vector<long long> subtreeWeight(numSN, 1);
    for (int si = 0; si < numSN; ++si) {
      const int p = parentSN[si];
      if (p != -1) subtreeWeight[p] += subtreeWeight[si];
    }

    // --- The actual per-supernode work: extend-add assembly from children's
    // generated elements + A's own contribution, dense LDL^T factorization
    // restricted to the eligible columns, and forwarding the remainder to
    // the parent (or flagging singularity at a root). Identical math to the
    // Phase 3 driver; only the surrounding scheduling changed. ---
    auto processSupernode = [&](int si) {
#ifdef SYMLA_HAVE_OPENMP
      const int tid = mfOptions.parallel ? omp_get_thread_num() : 0;
#else
      const int tid = 0;
#endif
      std::vector<int>& globalToLocal = globalToLocalTls[tid];

      const auto& sn = supernodes[si];
      const int ncols = sn.ncols;

      // --- Step A: collect "extra" indices delayed in from children that
      // are not already part of this supernode's symbolic row pattern
      // (i.e. genuinely new eligible columns arriving at runtime). ---
#ifdef SYMLA_PROFILE
      auto __tA0 = std::chrono::steady_clock::now();
#endif
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

#ifdef SYMLA_PROFILE
#pragma omp critical(symla_mf_profile)
      {
        g_mfProfile.extraFromChildrenSec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - __tA0).count();
        g_mfProfile.numFronts++;
        g_mfProfile.sumFrontSize += m;
        g_mfProfile.sumFrontSizeSq += (long long)m * m;
        g_mfProfile.sumEligTimesMsq += (double)nEligible * (double)m * (double)m;
        if (m > g_mfProfile.maxFrontSize) {
          g_mfProfile.maxFrontSize = m;
          g_mfProfile.maxFrontNEligible = nEligible;
        }
      }
      auto __tB0 = std::chrono::steady_clock::now();
#endif

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

#ifdef SYMLA_PROFILE
#pragma omp critical(symla_mf_profile)
      { g_mfProfile.extendSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - __tB0).count(); }
      auto __tC0 = std::chrono::steady_clock::now();
#endif
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

#ifdef SYMLA_PROFILE
#pragma omp critical(symla_mf_profile)
      { g_mfProfile.addSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - __tC0).count(); }
      auto __tD0 = std::chrono::steady_clock::now();
#endif
      // --- Factor, restricted to the eligible (fully-summed) block. ---
      MatrixX D;
      DenseLDLTResult res = DenseLDLT<Scalar>::factor(F, D, options, nEligible);
#ifdef SYMLA_PROFILE
#pragma omp critical(symla_mf_profile)
      { g_mfProfile.factorSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - __tD0).count(); }
#endif

      perFrontInertia[si] = res.inertia;

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
      nf.fronts[si] = std::move(sfac);

      // --- Forward the remainder (still-eligible-but-delayed columns +
      // ancestor rows) to the parent, or flag as singular if there is none. ---
      const int leftover = m - nFactored;
      if (leftover > 0) {
        const int psn = parentSN[si];
        if (psn == -1) {
          std::lock_guard<std::mutex> lock(singularMutex);
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
    };

#ifdef SYMLA_HAVE_OPENMP
    if (mfOptions.parallel) {
      // Avoid oversubscription: Eigen's own internal multi-threaded GEMM
      // would otherwise compete with the outer OpenMP task parallelism
      // across many concurrent small-to-medium fronts (each front's
      // `DenseLDLT::factor`/extend-add uses Eigen ops internally). Pinned
      // for the duration of this call; restored afterward. Left as a
      // simple global pin rather than adaptively re-enabling Eigen
      // threading for the handful of very large fronts real matrices like
      // bratu3d produce (a nested-parallelism tuning left for later, see
      // the Phase 6 handoff notes) -- this is the "keep it simple" default
      // the Phase 6 plan calls for.
      const int savedEigenThreads = Eigen::nbThreads();
      Eigen::setNbThreads(1);

      if (mfOptions.num_threads > 0) {
        omp_set_num_threads(mfOptions.num_threads);
      }

#pragma omp parallel
      {
#pragma omp single
        { parallel::runTaskDag(rootsSN, childrenSN, subtreeWeight, mfOptions.task_cutoff, processSupernode); }
      }

      Eigen::setNbThreads(savedEigenThreads);
    } else {
      for (int si = 0; si < numSN; ++si) processSupernode(si);
    }
#else
    for (int si = 0; si < numSN; ++si) processSupernode(si);
#endif

    for (int si = 0; si < numSN; ++si) {
      nf.inertia.n_pos += perFrontInertia[si].n_pos;
      nf.inertia.n_neg += perFrontInertia[si].n_neg;
      nf.inertia.n_zero += perFrontInertia[si].n_zero;
    }

    std::sort(nf.singularCols.begin(), nf.singularCols.end());
    return nf;
  }
};

}  // namespace symla
