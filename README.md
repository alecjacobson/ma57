# symla

`symla` is a from-scratch, header-only C++ sparse symmetric indefinite direct
solver built on [Eigen](https://eigen.tuxfamily.org), in the spirit of
Duff's MA57: multifrontal LDL^T factorization with 1x1/2x2 Bunch-Kaufman-style
threshold-pivoted stability, staged analyze/factorize/solve, multi-RHS solves,
OpenMP task-DAG parallelism, and a KKT-aware static-pivoting +
inertia-controlled-regularization mode for saddle-point/optimization systems.
See [`.claude/plans`](.) (or ask for the design doc) for the full background
and phased implementation history; this README covers building and using it.

No HSL/MA57 source was ever consulted while building this — only public
academic literature (Duff 2004, Bunch & Kaufman 1977, Ashcraft & Grimes 1989,
Vanderbei 1995, Wächter & Biegler 2006, etc.).

## Requirements

- CMake >= 3.18, a C++17 compiler
- [Eigen](https://eigen.tuxfamily.org) >= 3.4 (`libeigen3-dev` on Debian/Ubuntu)
- [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) AMD
  (`libsuitesparse-dev`) — **required**, used as a fill-reducing ordering
  backend (not for factorization)
- Optional: [METIS](https://github.com/KarypisLab/METIS) (`libmetis-dev`) for
  nested-dissection ordering
- Optional: OpenMP (usually bundled with the compiler) for parallel
  factorization
- Optional (benchmarks/tests only): SuiteSparse CHOLMOD/UMFPACK, used purely
  as head-to-head comparison baselines, and system LAPACK, used as an
  independent correctness oracle in the dense-kernel unit tests

On Ubuntu:

```sh
sudo apt-get install libeigen3-dev libsuitesparse-dev libmetis-dev
```

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j10          # pick a job count appropriate for your machine
ctest --test-dir build --output-on-failure
```

Useful CMake options (all `ON` by default when the dependency is found):

| Option | Effect |
|---|---|
| `SYMLA_WITH_METIS` | Enable METIS nested-dissection ordering |
| `SYMLA_WITH_OPENMP` | Enable OpenMP task-DAG parallel factorization |
| `SYMLA_WITH_SUITESPARSE_BENCH` | Link CHOLMOD/UMFPACK into the benchmark harness for comparison |
| `SYMLA_BUILD_TESTS` | Build the `ctest` regression suite |
| `SYMLA_BUILD_BENCH` | Build `bench/bench_main` |

`symla` itself is a header-only `INTERFACE` CMake target. To use it from
another project:

```cmake
add_subdirectory(path/to/ma57)
target_link_libraries(your_target PRIVATE symla::symla)
```

or simply add `include/` to your include path and link SuiteSparse AMD
(and METIS if you want nested dissection) yourself.

## Basic usage

```cpp
#include "symla/solver.hpp"

Eigen::SparseMatrix<double> A = ...;  // symmetric (only one triangle needs
                                       // to be stored); n x n
Eigen::MatrixXd B = ...;              // n x nrhs right-hand side(s)

symla::SymLDLT<double> solver;
solver.compute(A);                    // == analyzePattern(A) + factorize(A)
Eigen::MatrixXd X = solver.solve(B);  // supports multiple RHS columns

auto inertia = solver.inertia();      // {n_pos, n_neg, n_zero}
bool ok = !solver.isSingular();
```

### Staged API (fixed pattern / fixed system / solve)

The pattern (symbolic analysis: ordering + elimination tree + supernodes),
the numeric factorization, and the solve are three independent stages, so
you can reuse work across repeated solves:

```cpp
symla::SymLDLT<double> solver;
solver.analyzePattern(A);   // do this once per sparsity pattern
solver.factorize(A);        // do this once per set of numeric values
X1 = solver.solve(B1);      // do this as often as you like, cheaply,
X2 = solver.solve(B2);      // for any number of different RHS matrices

// New values, same pattern (e.g. a Newton iteration with fixed structure):
solver.factorize(A2);       // == solver.refactorize(A2)
X3 = solver.solve(B3);
```

### Configuration

```cpp
solver.setOrdering(symla::OrderingType::AMD);     // AMD (default), Metis, or Natural
solver.setPivotThreshold(0.01);                   // Bunch-Kaufman threshold u, in (0, 0.5]
solver.setParallel(true);                         // OpenMP task-DAG factorize (default: on if built with OpenMP)
solver.setNumThreads(0);                          // 0 = use omp_get_max_threads()
```

METIS nested dissection tends to give better (more balanced, more
parallelism-friendly) elimination trees than AMD on large, PDE/mesh-like 3D
problems; AMD is usually fine and cheaper to compute for small/unstructured
problems.

## KKT / saddle-point systems (`Mode::StaticRegularized`)

For symmetric quasidefinite (SQD) systems arising from optimization —
`K = [[-E, A^T], [A, F]]` with `E`, `F` symmetric positive (semi)definite —
Vanderbei's theory guarantees an LDL^T factorization exists for *any*
symmetric pivot order, so `symla` offers a static-pivoting mode that skips
dynamic pivot search entirely (cheaper, and more parallelism-friendly, since
it removes the delayed-pivot cross-front dependencies of dynamic pivoting)
and instead perturbs pivots that are too small or have the wrong expected
sign, escalating the perturbation and retrying until the resulting inertia
matches what the KKT block structure predicts:

```cpp
symla::SymLDLT<double> solver;
solver.setMode(symla::Mode::StaticRegularized);
solver.setKKTBlockSizes(n1, n2);   // first n1 original columns are the -E
                                    // block (expect negative pivots), the
                                    // remaining n2 are the F block (expect
                                    // positive pivots); n1 + n2 == n

solver.analyzePattern(K);
solver.factorize(K);               // throws std::runtime_error if inertia
                                    // can't be made to match after the
                                    // regularization retry budget is spent

// The factorization is of a perturbed K + Delta, not K exactly -- recover
// full accuracy with iterative refinement against the true K:
auto refined = solver.solveWithRefinement(K, B);
Eigen::MatrixXd X = refined.x;
std::cout << "refinement iterations: " << refined.diagnostics.iterations
          << ", final residual: " << refined.diagnostics.final_relative_residual << "\n";
```

`setExpectedSignPattern(std::vector<int>)` is the more general form (one
expected sign per original-index column, `-1`/`0`/`+1`) if your system isn't
a clean two-block KKT split. Diagnostics after `factorize()`:
`regularizationRetriesUsed()`, `lastRegularizationDelta()`,
`totalPerturbation()`, `numPerturbedPivots()`.

`solveWithRefinement()` also works under the default `Mode::ThresholdPivot`
as a general accuracy booster on ill-conditioned systems, not just after
static-pivoting perturbation.

## Tests

```sh
ctest --test-dir build --output-on-failure
```

The suite (`test/unit`, `test/correctness`, `test/stability`,
`test/concurrency`) covers: the dense pivoting kernel cross-checked against
LAPACK `dsytrf`; full-pipeline correctness against a dense brute-force LDL^T
oracle; solve residuals and multi-RHS consistency; genuine SQD/KKT
correctness (zero regularization needed, per Vanderbei's theory) and
regularization-retry/iterative-refinement behavior; conditioning and
near-singular/zero-diagonal stress tests; a pivot-threshold sweep; scaling
and supernode-amalgamation regression guards; and single- vs multi-thread
numerical equivalence plus repeated-run determinism checks. Several tests
also run against real matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu) (see below) and
skip cleanly if those haven't been downloaded.

## Benchmarks

```sh
python3 bench/fetch_matrices.py     # downloads a curated set of real KKT/
                                     # indefinite matrices (GHS_indef, Schenk,
                                     # HB groups) into bench/matrices/ (gitignored)
cmake --build build -j10
./build/bench/symla_bench_main      # times symla vs. Eigen SimplicialLDLT/
                                     # SparseLU and (if available) SuiteSparse
                                     # CHOLMOD/UMFPACK; writes bench/results/*.csv
```

## Project layout

```
include/symla/       public headers (header-only library)
  ordering.hpp          AMD / METIS / natural ordering wrappers
  elimination_tree.hpp  elimination tree + column counts
  symbolic.hpp           supernode partition + relaxed amalgamation
  dense_kernel.hpp      dense Bunch-Kaufman threshold-pivoted LDL^T (+ static pivoting)
  multifrontal.hpp      multifrontal numeric factorization driver (serial + task-DAG parallel)
  solve.hpp             tree-wise multi-RHS forward/diagonal/backward solve
  refine.hpp            classical iterative refinement
  parallel/task_graph.hpp  OpenMP task-DAG scheduler over the supernode tree
  solver.hpp             public SymLDLT<Scalar> API
test/unit/, test/correctness/, test/stability/, test/concurrency/   ctest suite
bench/                fetch_matrices.py, bench_main.cpp, results/ (gitignored)
```

## Known limitations

- The dense per-front kernel isn't panel-blocked, so very large individual
  fronts (which can occur with AMD ordering on some 3D/PDE-like matrices)
  don't get full BLAS-3 throughput; METIS nested dissection often avoids
  this by giving more balanced elimination trees.
- `solve()` is currently single-threaded (factorization is
  parallel; the solve's tree-wise dependency structure would support the
  same task-DAG approach in the future).
- Iterative refinement is fixed-precision, not mixed-precision GMRES-IR.
- No GPU backend.


## LLM Usage

Everything above this (and in the other files) was written by Claude code while I waited to get an academic-use license from HSL for MA57.
Obviously, I don't know if Anthropic's training has included the MA57 source code. 
I also don't know whether my Claude code instance went against my explicit direction to not look for or look at MA57 source code.
I can only say with certainty that I did not provide access to that code, nor did I have access to that code at the time this was generated.
At the time of writing this my prompts were:

>  Reimplement (without looking for or looking at their official code) the method of "MA57—A Code for the Solution of Sparse Symmetric Definite and Indefinite Systems" using Eigen in C++. The goal should be to outperform SuiteSparse and EigenLDL and LU on large sparse indefinite symmetric systems. Like those arising from KKT systems. Use a variety of regression testing for performance, accuracy, stability and robustness. A fully featured solver should enable precomputation stages for fixed pattern, fixed system,  and finally a solve stage for known rhs (which could have multiple columns). Research first. Take advantage of ideas that have emerged since the MA57 paper, but keep in mind that MA57 is still standard ldl for large sparse systems in matlab etc. So the primary goal is matching or exceeding MA57's expected bahavior in terms of performance, accuracy and robustness. Leverage parallelism. Make a plan and work independently.

> commit and push.

> Can you add a readme with usage instructions
