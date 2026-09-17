#pragma once

// Phase 7: classical iterative refinement.
//
// A static-pivoted + regularized factorization (see dense_kernel.hpp's
// `DenseLDLT::factorStatic` and solver.hpp's `Mode::StaticRegularized`)
// factors A + Delta, not A itself, for some (small, tracked) diagonal
// perturbation Delta. `SymLDLT::solve()` against that factorization is
// therefore only an *approximate* solve of the true system A x = b -- how
// approximate depends on how much regularization was needed. Classical
// iterative refinement (a standard textbook technique; see e.g. Golub & Van
// Loan, "Matrix Computations", or Higham, "Accuracy and Stability of
// Numerical Algorithms", ch. 12) recovers accuracy cheaply by reusing the
// existing factorization as a fixed-point iteration preconditioner:
//
//   r = b - A x        (computed against the TRUE, unperturbed A)
//   dx = solve(r)       (cheap: triangular solves against the existing
//                        factorization, no re-factorization)
//   x <- x + dx
//   repeat until the relative residual stops improving, hits a target
//   tolerance, or a max iteration count is reached.
//
// This is also useful (independent of static pivoting) as a general
// accuracy booster for the ordinary `Mode::ThresholdPivot` path on
// ill-conditioned systems, so it is written as a free function templated on
// the solver type rather than folded into a single mode -- `solver.hpp`
// wires it up as `SymLDLT::solveWithRefinement()` for both modes.
//
// This header intentionally does NOT include solver.hpp (avoiding an
// include cycle -- solver.hpp includes this header to implement
// `solveWithRefinement`); it only requires `SolverT::solve(const
// DenseMatrix&) const` to exist, checked at the call site via templates.

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <vector>

namespace symla {

struct RefinementOptions {
  // Hard cap on the number of refinement iterations (each iteration is one
  // extra sparse matvec against the true A plus one triangular solve
  // against the existing factorization -- cheap relative to a
  // re-factorization).
  int max_iterations = 10;

  // Target relative residual ||r|| / (||A|| * ||x|| + ||b||) (infinity-norm
  // based); refinement stops early once this is reached. The default is
  // near machine epsilon for double, i.e. "refine until it stops helping or
  // we hit roughly the best a fixed-precision residual computation can
  // resolve" -- true full machine-precision recovery would need the
  // residual computed in extended precision (a documented Phase 8 stretch
  // item; classical *fixed*-precision iterative refinement as implemented
  // here is limited by the precision of the residual computation itself).
  double tolerance = 1e-14;
};

struct RefinementResult {
  int iterations = 0;                        // refinement iterations actually taken (0 == unrefined solve was kept)
  double initial_relative_residual = 0.0;     // residual of the input (unrefined) solve
  double final_relative_residual = 0.0;       // residual after refinement
};

namespace detail {

template <typename Scalar, typename SparseMatrix>
inline double sparseMaxAbsRowSum(const SparseMatrix& A) {
  const int n = static_cast<int>(A.rows());
  std::vector<double> rowAbsSum(n, 0.0);
  for (int c = 0; c < A.outerSize(); ++c) {
    for (typename SparseMatrix::InnerIterator it(A, c); it; ++it) {
      rowAbsSum[static_cast<int>(it.row())] += std::abs(static_cast<double>(it.value()));
    }
  }
  double m = 0.0;
  for (double v : rowAbsSum) m = std::max(m, v);
  return m;
}

}  // namespace detail

// Refines `X` (already an approximate solution of `A * X = B`, e.g. from
// `solver.solve(B)`) in place. `solver.solve(R)` is used to apply the
// existing (fixed) factorization to each residual `R`; no re-factorization
// ever occurs here. `A` is read via `selfadjointView<Lower>()` (matching
// this project's "lower triangle is authoritative" convention used
// throughout, see multifrontal.hpp's PermutedLower), so it is correctly
// treated as symmetric regardless of whether only the lower triangle or
// both triangles are physically stored.
//
// Multi-RHS: all columns of B/X are refined together, sharing a single
// iteration count (the residual norm used for the stopping test is the max
// over columns), matching the plan's "multi-RHS should work" requirement.
template <typename SolverT, typename Scalar>
inline RefinementResult iterativeRefine(const SolverT& solver,
                                         const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>& A,
                                         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
                                         const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
                                         const RefinementOptions& options = RefinementOptions()) {
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  RefinementResult result;
  const double normA = detail::sparseMaxAbsRowSum<Scalar>(A);
  const double normB = B.template lpNorm<Eigen::Infinity>();

  auto relResidual = [&](const DenseMatrix& R, const DenseMatrix& Xcur) {
    const double normR = R.template lpNorm<Eigen::Infinity>();
    const double normX = Xcur.template lpNorm<Eigen::Infinity>();
    const double denom = normA * normX + normB;
    return denom > 0.0 ? normR / denom : normR;
  };

  DenseMatrix R = B - A.template selfadjointView<Eigen::Lower>() * X;
  double res = relResidual(R, X);
  result.initial_relative_residual = res;
  result.final_relative_residual = res;

  for (int it = 0; it < options.max_iterations; ++it) {
    if (res <= options.tolerance) break;

    DenseMatrix dx = solver.solve(R);
    DenseMatrix Xnew = X + dx;
    DenseMatrix Rnew = B - A.template selfadjointView<Eigen::Lower>() * Xnew;
    const double resNew = relResidual(Rnew, Xnew);

    if (!(resNew < res)) {
      // Stopped improving (or got worse, e.g. hit the floor set by the
      // precision of the residual computation itself) -- keep the better of
      // the two candidate iterates and stop.
      break;
    }

    X = std::move(Xnew);
    R = std::move(Rnew);
    res = resNew;
    result.iterations = it + 1;
    result.final_relative_residual = res;
  }

  return result;
}

}  // namespace symla
