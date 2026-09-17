#pragma once

// Fill-reducing ordering wrappers (AMD / METIS nested dissection / natural).
//
// Input assumption: the caller passes an Eigen::SparseMatrix<Scalar, ColMajor,
// int> that stores the pattern of a *symmetric* matrix. We do not require a
// particular triangle to be stored: we build the full symmetric pattern of
// A + A^T (with the diagonal removed) ourselves before calling into AMD or
// METIS, since both expect the compressed pattern of a symmetric graph
// without self-loops. This means it is safe to pass a matrix that stores
// only the lower triangle, only the upper triangle, or the full symmetric
// pattern -- the result is the same either way.

#include <Eigen/Sparse>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

extern "C" {
#include <amd.h>
}

#ifdef SYMLA_HAVE_METIS
#include <metis.h>
#endif

namespace symla {

enum class OrderingType {
  AMD,
  Metis,
  Natural,
};

namespace detail {

// Build the compressed-column pattern of (A + A^T) with the diagonal removed,
// from a SparseMatrix<Scalar, ColMajor, int> that may store only one
// triangle (or the full pattern -- duplicates are removed either way).
// Returns CSC arrays (colPtr of size n+1, rowIdx of size nnz) suitable for
// AMD / METIS.
template <typename SparseMatrix>
void buildSymmetricPatternNoDiag(const SparseMatrix& A, std::vector<int>& colPtr,
                                  std::vector<int>& rowIdx) {
  const int n = static_cast<int>(A.rows());
  if (A.cols() != A.rows()) {
    throw std::invalid_argument("symla::ordering: matrix must be square");
  }

  // Collect (row, col) and (col, row) for every stored off-diagonal entry,
  // then dedupe per column.
  std::vector<std::vector<int>> perCol(n);
  for (int k = 0; k < A.outerSize(); ++k) {
    for (typename SparseMatrix::InnerIterator it(A, k); it; ++it) {
      const int r = static_cast<int>(it.row());
      const int c = static_cast<int>(it.col());
      if (r == c) continue;
      perCol[c].push_back(r);
      perCol[r].push_back(c);
    }
  }

  colPtr.assign(n + 1, 0);
  for (int c = 0; c < n; ++c) {
    std::sort(perCol[c].begin(), perCol[c].end());
    perCol[c].erase(std::unique(perCol[c].begin(), perCol[c].end()), perCol[c].end());
    colPtr[c + 1] = colPtr[c] + static_cast<int>(perCol[c].size());
  }
  rowIdx.resize(colPtr[n]);
  for (int c = 0; c < n; ++c) {
    std::copy(perCol[c].begin(), perCol[c].end(), rowIdx.begin() + colPtr[c]);
  }
}

}  // namespace detail

// Computes a fill-reducing permutation `perm` such that permuted column j
// corresponds to original column perm[j] (i.e. perm is the same convention
// as AMD's P: P[k] = i means row/col i is the k-th row/col in the permuted
// (pivot) order). Returns an Eigen::VectorXi of size n.
template <typename SparseMatrix>
Eigen::VectorXi computeOrdering(const SparseMatrix& A, OrderingType type) {
  const int n = static_cast<int>(A.rows());
  if (A.cols() != A.rows()) {
    throw std::invalid_argument("symla::computeOrdering: matrix must be square");
  }

  if (type == OrderingType::Natural) {
    Eigen::VectorXi perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    return perm;
  }

  std::vector<int> colPtr, rowIdx;
  detail::buildSymmetricPatternNoDiag(A, colPtr, rowIdx);

  if (type == OrderingType::AMD) {
    Eigen::VectorXi perm(n);
    double control[AMD_CONTROL];
    double info[AMD_INFO];
    amd_defaults(control);
    // amd_order treats a null Ai as invalid even when nz == 0 (e.g. an
    // n==1 matrix with no off-diagonal entries); std::vector<int>::data()
    // is permitted to return nullptr for an empty vector, so guard against
    // that by always keeping at least one element allocated.
    if (rowIdx.empty()) rowIdx.push_back(0);
    const int status = amd_order(n, colPtr.data(), rowIdx.data(), perm.data(), control, info);
    if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
      throw std::runtime_error("symla::computeOrdering: amd_order failed");
    }
    return perm;
  }

  if (type == OrderingType::Metis) {
#ifdef SYMLA_HAVE_METIS
    if (n == 0) return Eigen::VectorXi(0);
    std::vector<idx_t> xadj(colPtr.begin(), colPtr.end());
    std::vector<idx_t> adjncy(rowIdx.begin(), rowIdx.end());
    if (adjncy.empty()) adjncy.push_back(0);  // avoid a null data() pointer, see AMD note above
    std::vector<idx_t> metisPerm(n), metisIperm(n);
    idx_t nvtxs = n;
    const int status = METIS_NodeND(&nvtxs, xadj.data(), adjncy.data(), /*vwgt=*/nullptr,
                                     /*options=*/nullptr, metisPerm.data(), metisIperm.data());
    if (status != METIS_OK) {
      throw std::runtime_error("symla::computeOrdering: METIS_NodeND failed");
    }
    // METIS_NodeND's `perm` output satisfies: perm[i] = position of original
    // node i in the new (permuted) ordering, i.e. it is the *inverse*
    // permutation relative to our convention (perm[k] = original index of
    // the k-th pivot, matching AMD's P). Convert accordingly: our perm is
    // METIS's `iperm`.
    Eigen::VectorXi perm(n);
    for (int i = 0; i < n; ++i) {
      const idx_t v = metisIperm[i];
      if (v < 0 || v >= static_cast<idx_t>(n)) {
        throw std::runtime_error("symla::computeOrdering: METIS iperm out of range (idx_t width mismatch?)");
      }
      perm[i] = static_cast<int>(v);
    }
    return perm;
#else
    throw std::runtime_error("symla::computeOrdering: METIS support not compiled in (SYMLA_HAVE_METIS undefined)");
#endif
  }

  throw std::invalid_argument("symla::computeOrdering: unknown OrderingType");
}

// Convenience overload returning an Eigen::PermutationMatrix.
template <typename SparseMatrix>
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> computeOrderingPermutation(
    const SparseMatrix& A, OrderingType type) {
  Eigen::VectorXi perm = computeOrdering(A, type);
  return Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>(perm);
}

}  // namespace symla
