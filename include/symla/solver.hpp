#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "symla/inertia.hpp"
#include "symla/symbolic.hpp"
#include "symla/multifrontal.hpp"
#include "symla/refine.hpp"
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

  // --- Phase 7: KKT-aware Mode::StaticRegularized configuration ---
  //
  // Describes the expected block-sign structure of a KKT/SQD system
  //     K = [[-E, A^T], [A, F]],  E, F symmetric PSD, n1 + n2 == n
  // in *original* matrix index space (i.e. as the caller assembled K, the
  // first n1 rows/columns are the -E block, expected to pivot negative; the
  // remaining n2 are the F block, expected to pivot positive) -- NOT in
  // post-ordering "final" index space, since callers generally do not know
  // (and should not need to know) what AMD/METIS did to their matrix.
  // `factorize()` maps this through the analyzed ordering (`symbolic_.
  // perm`) internally before handing it to the multifrontal driver. Only
  // meaningful under `Mode::StaticRegularized`; ignored otherwise. Setting
  // this also clears any previously-set explicit sign pattern (the two are
  // mutually exclusive).
  void setKKTBlockSizes(int n1, int n2) {
    if (n1 < 0 || n2 < 0) throw std::invalid_argument("symla::SymLDLT::setKKTBlockSizes: sizes must be nonnegative");
    kkt_n1_ = n1;
    kkt_n2_ = n2;
    use_kkt_blocks_ = true;
    use_explicit_signs_ = false;
  }

  // General form of the above: one expected sign per column, in *original*
  // index space, values in {-1, 0, +1} (0 == "no expectation for this
  // column" -- static pivoting still perturbs it if its raw magnitude is
  // below the floor, just without a preferred direction; see
  // dense_kernel.hpp's `DenseLDLT::factorStatic`). Used both to choose the
  // perturbation direction and (via `factorize()`'s retry loop) to detect a
  // wrong-inertia result. `signs.size()` must equal the matrix size once
  // `analyzePattern()`/`factorize()` is called (checked there, not here, so
  // this can be called before `analyzePattern()`).
  void setExpectedSignPattern(std::vector<int> signs) {
    expected_signs_orig_ = std::move(signs);
    use_explicit_signs_ = true;
    use_kkt_blocks_ = false;
  }

  // Inertia-controlled regularization retry loop (Wachter & Biegler
  // 2006-style), used only under `Mode::StaticRegularized` and only when an
  // expected sign pattern is known (via `setKKTBlockSizes`/
  // `setExpectedSignPattern`): if the resulting inertia doesn't match the
  // expectation, `factorize()` re-factors from scratch with `delta`
  // escalated by `regularization_growth_` (default 10x), up to
  // `max_regularization_retries_` (default 10) attempts, throwing
  // `std::runtime_error` if the budget is exhausted without a matching
  // inertia (never silently returns a wrong-inertia factorization as
  // "succeeded").
  void setInitialRegularizationDelta(double delta) { initial_regularization_delta_ = delta; }
  void setMaxRegularizationRetries(int retries) { max_regularization_retries_ = retries; }
  void setRegularizationGrowth(double growth) { regularization_growth_ = growth; }

  // Threshold below which a static-pivoted pivot is judged "too small to
  // use directly" and perturbed (see dense_kernel.hpp's `StaticPivotOptions`
  // for the exact semantics); exposed here mainly so tests/callers can force
  // a deliberately conservative (large) floor -- e.g. to exercise the
  // regularization + iterative-refinement path on an input that is already
  // genuinely well-conditioned/quasidefinite (so the resulting perturbed
  // factorization stays an accurate preconditioner, useful for isolating
  // "does refinement recover the accuracy regularization costs" from "is
  // regularization needed at all"). Defaults match StaticPivotOptions'
  // defaults.
  void setStaticPivotFloor(double relativeFloor, double absoluteFloor = 1e-300) {
    static_pivot_relative_floor_ = relativeFloor;
    static_pivot_absolute_floor_ = absoluteFloor;
  }

  // Diagnostics from the most recent `factorize()` call under
  // `Mode::StaticRegularized` (0/unset otherwise).
  double lastRegularizationDelta() const { return last_regularization_delta_; }
  int regularizationRetriesUsed() const { return regularization_retries_used_; }
  double totalPerturbation() const { return numeric_.totalPerturbation; }
  int numPerturbedPivots() const { return numeric_.numPerturbed; }

  // Phase 6: task-DAG parallel factorize() over the supernode tree (see
  // multifrontal.hpp / parallel/task_graph.hpp). Default is parallel with
  // however many threads OpenMP reports available
  // (`omp_get_max_threads()`/`OMP_NUM_THREADS`) whenever the library was
  // built with OpenMP support (`SYMLA_HAVE_OPENMP`); with no OpenMP support
  // compiled in, this setting is a no-op and factorize() is always the
  // single-threaded Phase 3 driver. `setParallel(false)` forces the exact
  // same single-threaded postorder loop regardless of OpenMP availability
  // (useful for deterministic debugging/reference runs, and for the
  // single- vs multi-thread equivalence tests in test/concurrency).
  void setParallel(bool parallel) { parallel_ = parallel; }
  // 0 (default) means "use whatever omp_get_max_threads() currently
  // reports"; a positive value pins factorize() to that many threads.
  void setNumThreads(int numThreads) { num_threads_ = numThreads; }

  // Analyze sparsity pattern only (ordering + symbolic factorization).
  // `options` controls the relaxed-amalgamation caps (symbolic.hpp); the
  // default reproduces the library's standard behavior, an explicit value
  // is mainly useful for tests that want to pin down a specific supernode
  // partition (e.g. disabling amalgamation with max_relax_size = 1).
  void analyzePattern(const SparseMatrix& A, const SymbolicFactorOptions& options = SymbolicFactorOptions()) {
    symbolic_ = SymbolicFactor::analyze(A, ordering_, options);
    pattern_analyzed_ = true;
    factorized_ = false;
  }

  // Numeric factorization against the previously analyzed pattern.
  void factorize(const SparseMatrix& A) {
    if (!pattern_analyzed_) {
      throw std::logic_error("symla::SymLDLT::factorize: analyzePattern must be called first");
    }
    if (mode_ == Mode::StaticRegularized) {
      factorizeStaticRegularized(A);
      return;
    }

    DenseLDLTOptions opts;
    opts.pivot_threshold = pivot_threshold_;
    MultifrontalOptions mfOpts;
#ifdef SYMLA_HAVE_OPENMP
    mfOpts.parallel = parallel_;
#else
    mfOpts.parallel = false;
#endif
    mfOpts.num_threads = num_threads_;
    numeric_ = MultifrontalFactorizer<Scalar>::factorize(symbolic_, A, opts, mfOpts);
    inertia_ = numeric_.inertia;
    factorized_ = true;
    last_regularization_delta_ = 0.0;
    regularization_retries_used_ = 0;
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
    SolveOptions solveOpts;
#ifdef SYMLA_HAVE_OPENMP
    solveOpts.parallel = parallel_;
#else
    solveOpts.parallel = false;
#endif
    solveOpts.num_threads = num_threads_;
    return MultifrontalSolver<Scalar>::solve(numeric_, B, solveOpts);
  }

  // Phase 7: `solve()` followed by classical iterative refinement against
  // the *original, unperturbed* `A` (see refine.hpp) -- the intended way to
  // recover full accuracy from a `Mode::StaticRegularized` factorization
  // (which factors A + Delta, not A exactly), but also usable as a general
  // accuracy booster under `Mode::ThresholdPivot` on ill-conditioned
  // systems. `A` must be supplied because the solver does not retain the
  // matrix it was factorized against (only its `analyzePattern`/
  // `factorize` calls take it as a parameter); it must match the matrix
  // most recently passed to `factorize()` in both pattern and values (this
  // is the caller's responsibility to ensure, same as `refactorize()`).
  struct RefinedSolve {
    DenseMatrix x;
    RefinementResult diagnostics;
  };
  RefinedSolve solveWithRefinement(const SparseMatrix& A, const DenseMatrix& B,
                                    const RefinementOptions& options = RefinementOptions()) const {
    RefinedSolve out;
    out.x = solve(B);
    out.diagnostics = iterativeRefine<SymLDLT<Scalar>, Scalar>(*this, A, out.x, B, options);
    return out;
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
#ifdef SYMLA_HAVE_OPENMP
  bool parallel_ = true;
#else
  bool parallel_ = false;
#endif
  int num_threads_ = 0;
  bool pattern_analyzed_ = false;
  bool factorized_ = false;
  Inertia inertia_;
  SymbolicFactor symbolic_;
  NumericFactor numeric_;

  // Phase 7: KKT-aware Mode::StaticRegularized configuration/diagnostics.
  bool use_kkt_blocks_ = false;
  int kkt_n1_ = 0;
  int kkt_n2_ = 0;
  bool use_explicit_signs_ = false;
  std::vector<int> expected_signs_orig_;

  double initial_regularization_delta_ = 0.0;  // 0 => let factorStatic pick its own default
  int max_regularization_retries_ = 10;
  double regularization_growth_ = 10.0;
  double static_pivot_relative_floor_ = 1e-8;
  double static_pivot_absolute_floor_ = 1e-300;

  double last_regularization_delta_ = 0.0;
  int regularization_retries_used_ = 0;

  // Builds the expected-sign vector in *original* matrix index space, size
  // n, from whichever of `setKKTBlockSizes`/`setExpectedSignPattern` was
  // last called (or all-zero if neither was). Throws on inconsistent sizes.
  std::vector<int> buildExpectedSignOriginal(int n) const {
    if (use_explicit_signs_) {
      if (static_cast<int>(expected_signs_orig_.size()) != n) {
        throw std::invalid_argument(
            "symla::SymLDLT::factorize: setExpectedSignPattern's vector size does not match the matrix size");
      }
      return expected_signs_orig_;
    }
    std::vector<int> signs(n, 0);
    if (use_kkt_blocks_) {
      if (kkt_n1_ + kkt_n2_ != n) {
        throw std::invalid_argument(
            "symla::SymLDLT::factorize: setKKTBlockSizes(n1, n2) sizes do not sum to the matrix size");
      }
      for (int i = 0; i < kkt_n1_; ++i) signs[i] = -1;
      for (int i = kkt_n1_; i < n; ++i) signs[i] = +1;
    }
    return signs;
  }

  // Phase 7: Mode::StaticRegularized factorize() path -- static pivoting
  // (dense_kernel.hpp's DenseLDLT::factorStatic) plus an inertia-controlled
  // regularization retry loop (IPOPT/Wachter & Biegler 2006-style): after
  // each attempt, if an expected inertia is known (from setKKTBlockSizes/
  // setExpectedSignPattern) and doesn't match, escalate `delta`
  // geometrically and re-factor from scratch, up to
  // `max_regularization_retries_` attempts. If no expected sign pattern was
  // ever set, this degrades to a single static-pivoting pass with no retry
  // (there is nothing to check the inertia against).
  void factorizeStaticRegularized(const SparseMatrix& A) {
    const int n = symbolic_.etree.n;
    const bool haveExpectation = use_kkt_blocks_ || use_explicit_signs_;

    const std::vector<int> signsOrig = buildExpectedSignOriginal(n);
    // Map original index space -> final (post-ordering) index space:
    // signsFinal[t] corresponds to symbolic_.perm[t] == original index.
    std::vector<int> signsFinal(n, 0);
    for (int t = 0; t < n; ++t) signsFinal[t] = signsOrig[symbolic_.perm[t]];

    int expectedPos = 0, expectedNeg = 0;
    for (int s : signsOrig) {
      if (s > 0)
        ++expectedPos;
      else if (s < 0)
        ++expectedNeg;
    }

    double delta = initial_regularization_delta_;
    if (delta <= 0.0) {
      // Match factorStatic's own default scale choice (sqrt(eps) * A's own
      // max magnitude) so the first attempt is a sensible "try with a tiny
      // regularization" pass, not an arbitrary constant.
      double maxAbs = 0.0;
      for (int c = 0; c < A.outerSize(); ++c) {
        for (typename SparseMatrix::InnerIterator it(A, c); it; ++it) {
          maxAbs = std::max(maxAbs, std::abs(static_cast<double>(it.value())));
        }
      }
      if (maxAbs <= 0.0) maxAbs = 1.0;
      delta = std::sqrt(std::numeric_limits<double>::epsilon()) * maxAbs;
    }

    const int maxAttempts = std::max(1, max_regularization_retries_ + 1);
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
      DenseLDLTOptions dummyOpts;  // unused by the static path but required by the driver's signature
      dummyOpts.pivot_threshold = pivot_threshold_;

      MultifrontalOptions mfOpts;
#ifdef SYMLA_HAVE_OPENMP
      mfOpts.parallel = parallel_;
#else
      mfOpts.parallel = false;
#endif
      mfOpts.num_threads = num_threads_;
      mfOpts.static_pivoting = true;
      mfOpts.staticOptions.delta = delta;
      mfOpts.staticOptions.relative_pivot_floor = static_pivot_relative_floor_;
      mfOpts.staticOptions.absolute_floor = static_pivot_absolute_floor_;
      mfOpts.expectedSign = signsFinal;

      NumericFactor attemptResult = MultifrontalFactorizer<Scalar>::factorize(symbolic_, A, dummyOpts, mfOpts);

      last_regularization_delta_ = delta;
      regularization_retries_used_ = attempt;

      // Full coverage (every column has a declared expected sign, the
      // normal KKT/SQD case) lets us demand the *exact* expected pos/neg
      // split; partial coverage (some columns' expectation left at 0, i.e.
      // "unconstrained") only lets us demand no zero pivots and no
      // singularity -- there's no expected count to compare the
      // unconstrained columns' contribution against.
      const bool fullCoverage = (expectedPos + expectedNeg == n);
      const bool inertiaOk =
          !haveExpectation || (attemptResult.inertia.n_zero == 0 && !attemptResult.singular &&
                                (!fullCoverage ||
                                 (attemptResult.inertia.n_pos == expectedPos && attemptResult.inertia.n_neg == expectedNeg)));

      if (inertiaOk) {
        numeric_ = std::move(attemptResult);
        inertia_ = numeric_.inertia;
        factorized_ = true;
        return;
      }

      if (attempt + 1 >= maxAttempts) {
        std::ostringstream oss;
        oss << "symla::SymLDLT::factorize (Mode::StaticRegularized): inertia-controlled regularization "
               "retry budget ("
            << max_regularization_retries_ << " retries) exhausted without matching the expected inertia. "
            << "expected (n_pos=" << expectedPos << ", n_neg=" << expectedNeg << ", n_zero=0), got (n_pos="
            << attemptResult.inertia.n_pos << ", n_neg=" << attemptResult.inertia.n_neg
            << ", n_zero=" << attemptResult.inertia.n_zero << ", singular=" << (attemptResult.singular ? "true" : "false")
            << ") at delta=" << delta << ". Refusing to report a wrong-inertia factorization as successful.";
        throw std::runtime_error(oss.str());
      }

      delta *= regularization_growth_;
    }
  }
};

}  // namespace symla
