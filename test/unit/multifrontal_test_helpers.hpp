#pragma once

// Test-only helpers for Phase 3 (multifrontal.hpp / solver.hpp) correctness
// checks: reconstructs one flat, monolithic (dense) L/D/permutation from a
// symla::NumericFactor's per-supernode fronts, in the *actual numeric
// elimination order* (which -- because of delayed pivots -- can differ from
// SymbolicFactor::perm's purely-symbolic final order: a column delayed past
// its own supernode is only finalized once it reaches some ancestor front,
// i.e. strictly later than the symbolic order would suggest). This lets
// tests verify P A P^T == L D L^T and do a plain dense forward/diag/back
// solve, independent of Phase 4's (not-yet-implemented) tree-wise solve.
//
// This is test-only scaffolding (not part of the library's public API):
// production code (Phase 4) is expected to do a genuinely multifrontal
// (tree-wise, blocked) solve directly over NumericFactor::fronts rather
// than flattening to a dense n x n L, which would defeat the point of
// sparsity for large problems; flattening here is fine because unit/
// correctness test sizes are small-to-medium.

#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <Eigen/Dense>

#include <vector>

namespace symla_test {

struct FlattenedFactor {
  Eigen::VectorXi origIndex;  // size n; origIndex[t] = original A index of the t-th eliminated pivot
  Eigen::MatrixXd L;          // n x n, unit lower triangular in elimination order t
  Eigen::MatrixXd D;          // n x n, block-diagonal in elimination order t
};

// Flattens every finalized pivot across all fronts (in `nf.fronts` order,
// which is a valid elimination order -- see multifrontal.hpp's doc comment
// on NumericFactor) into one dense L/D pair. Requires the factorization to
// be non-singular (asserts via SYMLA_CHECK otherwise -- callers testing the
// singular path should not call this).
inline FlattenedFactor flatten(const symla::SymbolicFactor& sf, const symla::NumericFactor& nf) {
  const int n = nf.n;
  SYMLA_CHECK(!nf.singular);

  std::vector<int> order;
  order.reserve(n);
  for (const auto& f : nf.fronts) {
    for (int p = 0; p < f.nPivots; ++p) order.push_back(f.rowIndices[p]);
  }
  SYMLA_CHECK(static_cast<int>(order.size()) == n);

  std::vector<int> posOfFinal(n, -1);
  for (int t = 0; t < n; ++t) posOfFinal[order[t]] = t;

  FlattenedFactor out;
  out.origIndex.resize(n);
  for (int t = 0; t < n; ++t) out.origIndex[t] = sf.perm[order[t]];
  out.L = Eigen::MatrixXd::Identity(n, n);
  out.D = Eigen::MatrixXd::Zero(n, n);

  for (const auto& f : nf.fronts) {
    const int m = static_cast<int>(f.rowIndices.size());
    const int k = f.nPivots;
    std::vector<bool> isD21(k, false);
    for (const auto& pb : f.pivotBlocks) {
      if (pb.kind == symla::PivotKind::TwoByTwo) isD21[pb.start + 1] = true;
    }
    for (int p = 0; p < k; ++p) {
      const int tp = posOfFinal[f.rowIndices[p]];
      // Only i > p: DenseLDLT mirrors the strictly-lower L multipliers into
      // the strictly-upper triangle too (needed internally for its
      // symmetric trailing-update formula), so f.L(i, p) for i < p is just
      // a duplicate of f.L(p, i), not independent data -- reading it here
      // would corrupt the (unit lower triangular) global L into a
      // symmetric matrix.
      for (int i = p + 1; i < m; ++i) {
        if (i < k && isD21[i] && i == p + 1) continue;  // d21 slot, not an L multiplier
        const int ti = posOfFinal[f.rowIndices[i]];
        out.L(ti, tp) = f.L(i, p);
      }
    }
    for (int p = 0; p < k; ++p) {
      for (int q = 0; q < k; ++q) {
        out.D(posOfFinal[f.rowIndices[p]], posOfFinal[f.rowIndices[q]]) = f.D(p, q);
      }
    }
  }
  return out;
}

// Dense P A P^T reconstruction (using FlattenedFactor::origIndex as P) for
// comparison against L D L^T.
template <typename SparseMatrix>
inline Eigen::MatrixXd permuteDense(const SparseMatrix& A, const Eigen::VectorXi& origIndex) {
  const int n = static_cast<int>(origIndex.size());
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < A.outerSize(); ++k) {
    for (typename SparseMatrix::InnerIterator it(A, k); it; ++it) {
      Ad(static_cast<int>(it.row()), static_cast<int>(it.col())) = it.value();
    }
  }
  Eigen::VectorXi newIndexOf(n);
  for (int t = 0; t < n; ++t) newIndexOf[origIndex[t]] = t;
  Eigen::MatrixXd out(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) out(i, j) = Ad(origIndex[i], origIndex[j]);
  return out;
}

// Solves A x = b using the flattened factorization (forward L, block-
// diagonal D, back L^T, un-permute), for testing purposes.
inline Eigen::VectorXd solveFlattened(const symla::NumericFactor& nf, const FlattenedFactor& fac,
                                       const Eigen::VectorXd& b) {
  const int n = fac.origIndex.size();
  // Permute b into elimination order: bp(t) = b(origIndex(t)).
  Eigen::VectorXd bp(n);
  for (int t = 0; t < n; ++t) bp(t) = b(fac.origIndex(t));

  // Collect global pivot-block structure (start position, kind) in
  // elimination-order coordinates.
  std::vector<std::pair<int, symla::PivotKind>> blocks;
  {
    std::vector<int> order;
    for (const auto& f : nf.fronts)
      for (int p = 0; p < f.nPivots; ++p) order.push_back(f.rowIndices[p]);
    std::vector<int> posOfFinal(n, -1);
    for (int t = 0; t < n; ++t) posOfFinal[order[t]] = t;
    for (const auto& f : nf.fronts) {
      for (const auto& pb : f.pivotBlocks) {
        const int globalStart = posOfFinal[f.rowIndices[pb.start]];
        blocks.emplace_back(globalStart, pb.kind);
      }
    }
  }
  std::vector<bool> isD21(n, false);
  for (auto& blk : blocks)
    if (blk.second == symla::PivotKind::TwoByTwo) isD21[blk.first + 1] = true;

  // Forward solve L y = bp.
  Eigen::VectorXd y = bp;
  for (int j = 0; j < n; ++j)
    for (int i = j + 1; i < n; ++i) {
      if (isD21[i] && i == j + 1) continue;
      y(i) -= fac.L(i, j) * y(j);
    }

  // Block-diagonal solve D z = y.
  Eigen::VectorXd z(n);
  for (const auto& blk : blocks) {
    const int k = blk.first;
    if (blk.second == symla::PivotKind::OneByOne) {
      z(k) = y(k) / fac.D(k, k);
    } else {
      const double d11 = fac.D(k, k), d21 = fac.D(k + 1, k), d22 = fac.D(k + 1, k + 1);
      const double det = d11 * d22 - d21 * d21;
      const double y0 = y(k), y1 = y(k + 1);
      z(k) = (d22 * y0 - d21 * y1) / det;
      z(k + 1) = (-d21 * y0 + d11 * y1) / det;
    }
  }

  // Back solve L^T w = z.
  Eigen::VectorXd w = z;
  for (int j = n - 1; j >= 0; --j)
    for (int i = j + 1; i < n; ++i) {
      if (isD21[i] && i == j + 1) continue;
      w(j) -= fac.L(i, j) * w(i);
    }

  // Un-permute.
  Eigen::VectorXd x(n);
  for (int t = 0; t < n; ++t) x(fac.origIndex(t)) = w(t);
  return x;
}

}  // namespace symla_test
