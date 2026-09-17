#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdexcept>

#include "symla/inertia.hpp"
#include "symla/symbolic.hpp"
#include "symla/multifrontal.hpp"

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

  DenseMatrix solve(const DenseMatrix& B) const {
    (void)B;
    throw std::logic_error("symla::SymLDLT::solve: not yet implemented (Phase 4)");
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
