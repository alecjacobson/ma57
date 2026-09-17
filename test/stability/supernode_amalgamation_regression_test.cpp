// Phase 1.1 regression guard: relaxed amalgamation must actually amalgamate
// on real, branchy 3D/2D-PDE-like matrices, not just on chains.
//
// Context: symbolic.hpp's original relaxed-amalgamation implementation only
// merged supernodes that were *index-adjacent* in the postordered column
// numbering. Under a postordered elimination tree, only the last-visited
// child of a node ends up index-adjacent to it (postorder places a node's
// last child immediately before the node itself) -- every other child is
// separated from its parent by other subtrees' columns. On branchy etrees
// (typical of PDE stencils/meshes, where several elements/nodes merge into a
// coarser separator) this meant almost no merging ever fired: measured on
// GHS_indef/stokes128 (n=49666, a Stokes-flow saddle-point matrix), the old
// scheme produced 16618 supernodes averaging just ~3.0 columns each; on
// GHS_indef/bratu3d (n=27792, a 3D PDE problem) it produced 17282
// supernodes averaging ~1.6 columns -- essentially defeating the point of
// the multifrontal method (no meaningful BLAS-3 block sizes). The fix
// generalizes amalgamation to merge any parent/child pair in the *supernode
// tree* (not just index-adjacent ones), with a renumbering pass afterward to
// restore column contiguity within each merged supernode.
//
// stokes128 is used here (rather than bratu3d) because it shows the clearer
// signal: at this library's current (conservative) default relaxation
// settings, general merging lifts its average from ~3.0 to ~10.8
// columns/supernode -- bratu3d's sibling branches happen to have more
// heterogeneous row patterns, so it only reaches ~1.65 at the same default
// settings (see symbolic.hpp's SymbolicFactorOptions comment for the full
// tradeoff data and why the default fill-fraction cap is kept conservative).
// Skips cleanly if bench/matrices/ hasn't been populated (same convention as
// test/correctness/real_matrix_test.cpp).
#include "symla/solver.hpp"
#include "test_helpers.hpp"

#include <unsupported/Eigen/SparseExtra>

#include <filesystem>
#include <iostream>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using symla::OrderingType;
using symla::SymLDLT;

namespace fs = std::filesystem;

int main() {
  const fs::path matricesRoot =
      fs::path(__FILE__).parent_path().parent_path().parent_path() / "bench" / "matrices";
  const fs::path mtxPath = matricesRoot / "GHS_indef" / "stokes128" / "stokes128.mtx";

  if (!fs::exists(mtxPath)) {
    std::cout << "symla supernode_amalgamation_regression_test: SKIPPED -- " << mtxPath
              << " not found. Run `python3 bench/fetch_matrices.py` to populate it.\n";
    return 0;
  }

  Sparse A;
  if (!Eigen::loadMarket(A, mtxPath.string()) || A.rows() == 0) {
    std::cerr << "symla supernode_amalgamation_regression_test: FAILED to load " << mtxPath << "\n";
    return 1;
  }
  const int n = static_cast<int>(A.rows());

  SymLDLT<double> solver;
  solver.setOrdering(OrderingType::AMD);
  solver.analyzePattern(A);  // default SymbolicFactorOptions

  const auto& sf = solver.symbolicFactor();
  const auto& sns = sf.supernodes;
  SYMLA_CHECK(!sns.empty());

  const double avgNcols = static_cast<double>(n) / static_cast<double>(sns.size());
  std::cout << "symla supernode_amalgamation_regression_test: stokes128 n=" << n
            << " supernodes=" << sns.size() << " avg ncols=" << avgNcols << "\n";

  // Floor of 5 columns/supernode: comfortably above the old
  // index-adjacent-only scheme's ~3.0 (so a regression back to that
  // behavior fails loudly), comfortably below the current ~10.8 measured
  // with general merging (so ordinary AMD/ordering nondeterminism or minor
  // future retuning of the relaxation caps won't flake this test).
  SYMLA_CHECK(avgNcols > 5.0);

  std::cout << "symla supernode_amalgamation_regression_test: PASSED\n";
  return 0;
}
