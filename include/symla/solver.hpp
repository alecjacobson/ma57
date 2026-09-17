#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdexcept>

#include "symla/inertia.hpp"
#include "symla/symbolic.hpp"
#include "symla/multifrontal.hpp"
#include "symla/solve.hpp"

namespace symla {

enum class Mode {
  ThresholdPivot,
  StaticRegularized,
};

// Public solver API. Phase 0-2 established the skeleton and the dense
// frontal kernel; Phase 3 wires `analyzePattern` to the Phase 1 symbolic
// analysis and `factorize` to the Phase 3 multifrontal numeric driver.
// `solve()` remains unimplemented until Phase 4.
template <typename Scalar_>
class SymLDLT {
 public:
  using Scalar = Scalar_;
  using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  void setMode(Mode mode) { mode_ = mode; }
  void setPivotThreshold(double u) { pivot_threshold_ = u; }
  void setOrdering(OrderingType ordering) { ordering_ = ordering; }

  // Analyze sparsity pattern only (ordering + symbolic factorization).
  void analyzePattern(const SparseMatrix& A) {
    symbolic_ = SymbolicFactor::analyze(A, ordering_);
    pattern_analyzed_ = true;
    factorized_ = false;
  }

  // Numeric factorization against the previously analyzed pattern.
  void factorize(const SparseMatrix& A) {
    if (!pattern_analyzed_) {
      throw std::logic_error("symla::SymLDLT::factorize: analyzePattern must be called first");
    }
    DenseLDLTOptions opts;
    opts.pivot_threshold = pivot_threshold_;
    numeric_ = MultifrontalFactorizer<Scalar>::factorize(symbolic_, A, opts);
    inertia_ = numeric_.inertia;
    factorized_ = true;
  }

  // Convenience: analyzePattern + factorize.
  void compute(const SparseMatrix& A) {
    analyzePattern(A);
    factorize(A);
  }

  // Re-factorize with new values on the same pattern (no re-analysis).
  void refactorize(const SparseMatrix& A) { factorize(A); }

  // Multifrontal triangular solve (Phase 4), operating directly on the
  // per-front sparse structure (see solve.hpp) -- never materializes a
  // dense n x n matrix. Throws if `factorize()` has not been called yet.
  //
  // Design choice: if the factorization is numerically singular
  // (`isSingular()`), this throws rather than attempting a best-effort
  // solve. A singular factorization means one or more columns could never
  // be pivoted anywhere in the tree, including at a root front, so the
  // corresponding rows of `L`/`D` for those columns were never finalized
  // (dense_kernel.hpp documents their state as "unspecified but harmless"
  // for factorization purposes only) -- there is no well-defined
  // `SupernodeFactor::pivotBlocks` entry to diagonal-solve against for
  // them, so a "best effort" solve would either need ad hoc handling (e.g.
  // treating them as a pseudo-inverse/least-squares step) or silently
  // produce entries that are not just imprecise but structurally
  // meaningless. Throwing keeps `solve()`'s contract simple and matches
  // this project's existing convention of throwing on
  // precondition/consistency violations (e.g. calling `factorize()` before
  // `analyzePattern()`). Callers who legitimately need a solve against a
  // rank-deficient system (e.g. KKT systems with detected zero pivots) are
  // expected to use `isSingular()`/`singularColumns()` first and handle
  // rank deficiency explicitly (Phase 7's KKT/regularization mode is the
  // intended long-term answer for that use case, not a silent best-effort
  // path here).
  DenseMatrix solve(const DenseMatrix& B) const {
    if (!factorized_) {
      throw std::logic_error("symla::SymLDLT::solve: factorize() must be called before solve()");
    }
    if (numeric_.singular) {
      throw std::logic_error(
          "symla::SymLDLT::solve: factorization is numerically singular (see isSingular()/"
          "singularColumns()); refusing to solve");
    }
    if (B.rows() != numeric_.n) {
      throw std::invalid_argument("symla::SymLDLT::solve: B.rows() must match the factorized matrix size");
    }
    return MultifrontalSolver<Scalar>::solve(numeric_, B);
  }

  const Inertia& inertia() const { return inertia_; }

  bool patternAnalyzed() const { return pattern_analyzed_; }
  bool factorized() const { return factorized_; }

  // Numerical-singularity report: true if, after factorize(), some
  // column(s) could never be pivoted anywhere in the tree (including at a
  // root front) -- a genuine rank deficiency, not a bug. `singularColumns`
  // are given in *original* matrix index space.
  bool isSingular() const { return numeric_.singular; }
  std::vector<int> singularColumns() const {
    std::vector<int> out;
    out.reserve(numeric_.singularCols.size());
    for (int finalIdx : numeric_.singularCols) out.push_back(symbolic_.perm[finalIdx]);
    return out;
  }

  // Direct access to the Phase 1 symbolic analysis and Phase 3 numeric
  // factorization, for tests and for Phase 4's solve() implementation.
  const SymbolicFactor& symbolicFactor() const { return symbolic_; }
  const NumericFactor& numericFactor() const { return numeric_; }

 private:
  Mode mode_ = Mode::ThresholdPivot;
  double pivot_threshold_ = 0.01;
  OrderingType ordering_ = OrderingType::AMD;
  bool pattern_analyzed_ = false;
  bool factorized_ = false;
  Inertia inertia_;
  SymbolicFactor symbolic_;
  NumericFactor numeric_;
};

}  // namespace symla
