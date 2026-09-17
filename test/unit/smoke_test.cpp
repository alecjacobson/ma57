// Phase 0 smoke test: confirms the project builds, links against Eigen +
// SuiteSparse AMD, and the SymLDLT API skeleton compiles and instantiates.
#include "symla/solver.hpp"

#include <cassert>
#include <iostream>

int main() {
  symla::SymLDLT<double> solver;
  solver.setPivotThreshold(0.01);
  solver.setMode(symla::Mode::ThresholdPivot);
  assert(!solver.patternAnalyzed());
  assert(!solver.factorized());

  Eigen::SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 1.0;
  A.insert(1, 1) = 1.0;

  bool threw = false;
  try {
    solver.analyzePattern(A);
  } catch (const std::logic_error&) {
    threw = true;
  }
  assert(threw);

  std::cout << "symla Phase 0 smoke test OK\n";
  return 0;
}
