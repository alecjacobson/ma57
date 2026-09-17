#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdexcept>

namespace symla {

enum class Mode {
  ThresholdPivot,
  StaticRegularized,
};

struct Inertia {
  int n_pos = 0;
  int n_neg = 0;
  int n_zero = 0;
};

// Public solver skeleton. Analyze/factorize/solve staging is implemented
// incrementally across later phases (symbolic analysis, dense frontal
// kernel, multifrontal driver, solve phase). This header currently only
// establishes the API surface and compiles against Eigen's sparse types.
template <typename Scalar_>
class SymLDLT {
 public:
  using Scalar = Scalar_;
  using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  void setMode(Mode mode) { mode_ = mode; }
  void setPivotThreshold(double u) { pivot_threshold_ = u; }

  // Analyze sparsity pattern only (ordering + symbolic factorization).
  void analyzePattern(const SparseMatrix& A) {
    (void)A;
    throw std::logic_error("symla::SymLDLT::analyzePattern: not yet implemented (Phase 1)");
  }

  // Numeric factorization against the previously analyzed pattern.
  void factorize(const SparseMatrix& A) {
    (void)A;
    throw std::logic_error("symla::SymLDLT::factorize: not yet implemented (Phase 2/3)");
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

 private:
  Mode mode_ = Mode::ThresholdPivot;
  double pivot_threshold_ = 0.01;
  bool pattern_analyzed_ = false;
  bool factorized_ = false;
  Inertia inertia_;
};

}  // namespace symla
