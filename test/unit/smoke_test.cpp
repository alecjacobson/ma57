// Phase 0 smoke test: confirms the project builds, links against Eigen +
// SuiteSparse AMD, and the SymLDLT API skeleton compiles and instantiates.
#include "symla/solver.hpp"

#include <cstdlib>
#include <iostream>

// NDEBUG (set by the project's default Release build) compiles out bare
// assert(), which would make this test pass vacuously under `ctest`. Use an
// always-active check instead (mirrors SYMLA_CHECK in test_helpers.hpp,
// duplicated here since this is Phase 0's only test and predates that helper).
#define SYMLA_SMOKE_CHECK(cond)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "check failed: " #cond " at " __FILE__ ":" << __LINE__      \
                << "\n";                                                       \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

int main() {
  symla::SymLDLT<double> solver;
  solver.setPivotThreshold(0.01);
  solver.setMode(symla::Mode::ThresholdPivot);
  SYMLA_SMOKE_CHECK(!solver.patternAnalyzed());
  SYMLA_SMOKE_CHECK(!solver.factorized());

  Eigen::SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 2.0;
  A.insert(1, 1) = 3.0;

  // As of Phase 3, analyzePattern/factorize genuinely run (Phase 1 symbolic
  // analysis + Phase 3 multifrontal numeric factorization) rather than
  // throwing "not yet implemented".
  solver.analyzePattern(A);
  SYMLA_SMOKE_CHECK(solver.patternAnalyzed());
  SYMLA_SMOKE_CHECK(!solver.factorized());

  solver.factorize(A);
  SYMLA_SMOKE_CHECK(solver.factorized());
  SYMLA_SMOKE_CHECK(!solver.isSingular());
  SYMLA_SMOKE_CHECK(solver.inertia().n_pos == 2);
  SYMLA_SMOKE_CHECK(solver.inertia().n_neg == 0);
  SYMLA_SMOKE_CHECK(solver.inertia().n_zero == 0);

  // As of Phase 4, solve() genuinely solves (diag(2,3) x = b).
  Eigen::MatrixXd b(2, 1);
  b << 4.0, 9.0;
  Eigen::MatrixXd x = solver.solve(b);
  SYMLA_SMOKE_CHECK(std::abs(x(0, 0) - 2.0) < 1e-12);
  SYMLA_SMOKE_CHECK(std::abs(x(1, 0) - 3.0) < 1e-12);

  // solve() still throws if called before factorize().
  symla::SymLDLT<double> unfactored;
  bool threw = false;
  try {
    unfactored.solve(Eigen::MatrixXd::Zero(2, 1));
  } catch (const std::logic_error&) {
    threw = true;
  }
  SYMLA_SMOKE_CHECK(threw);

  std::cout << "symla Phase 0 smoke test OK\n";
  return 0;
}
