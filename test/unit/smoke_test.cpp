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
  A.insert(0, 0) = 1.0;
  A.insert(1, 1) = 1.0;

  bool threw = false;
  try {
    solver.analyzePattern(A);
  } catch (const std::logic_error&) {
    threw = true;
  }
  SYMLA_SMOKE_CHECK(threw);

  std::cout << "symla Phase 0 smoke test OK\n";
  return 0;
}
