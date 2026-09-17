// Phase 5 performance benchmark harness: times symla::SymLDLT's
// analyze/factorize/solve phases separately against Eigen::SimplicialLDLT,
// Eigen::SparseLU, SuiteSparse CHOLMOD (Eigen::CholmodSupernodalLLT, SPD
// subset only), and SuiteSparse UMFPACK (Eigen::UmfPackLU, as an
// unsymmetric-LU baseline for comparison context) on:
//   - every real matrix found under bench/matrices/ (bench/fetch_matrices.py)
//   - a couple of large synthetic sparse indefinite/KKT matrices
//
// Records matrix name/group, n, nnz, each solver's analyze/factorize/solve
// times, residual achieved, and peak resident memory (VmHWM from
// /proc/self/status -- a whole-process high-water mark, not per-solver, but
// simple and good enough for regression tracking) into a CSV under
// bench/results/.
#include "symla/solver.hpp"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>

#ifdef SYMLA_HAVE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef SYMLA_HAVE_UMFPACK
#include <Eigen/UmfPackSupport>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

long vmHwmKb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::istringstream iss(line.substr(6));
      long kb = -1;
      iss >> kb;
      return kb;
    }
  }
  return -1;
}

struct MatrixCase {
  std::string group;
  std::string name;
  Sparse A;        // lower-triangle-authoritative (symla/MatrixMarket convention)
  bool isSPD = false;
  bool isSynthetic = false;
};

Eigen::SparseMatrix<double, Eigen::ColMajor, int> fullSymmetric(const Sparse& A) {
  // Materializes both triangles (needed by solvers, like UmfPackLU, that
  // treat the matrix as general/unsymmetric and thus need every stored
  // entry, not just the "lower triangle is authoritative" convention symla/
  // CholmodSupport/SimplicialLDLT use).
  return Sparse(A.selfadjointView<Eigen::Lower>());
}

double residualSelfAdjoint(const Sparse& A, const Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
  Eigen::MatrixXd AX = A.selfadjointView<Eigen::Lower>() * X;
  double Amax = 0.0;
  for (int k = 0; k < A.outerSize(); ++k)
    for (Sparse::InnerIterator it(A, k); it; ++it) Amax = std::max(Amax, std::abs(it.value()));
  if (Amax == 0.0) Amax = 1.0;
  return (AX - B).norm() / (Amax * std::max(1.0, X.norm()) + B.norm());
}

// --- Synthetic matrix generators (same style as test/unit/test_helpers.hpp,
// duplicated here rather than shared since bench/ is a separate executable
// target with no dependency on test/ headers) ---

Sparse randomSparseIndefinite(int n, int avgNnzPerRow, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> colDist(0, n - 1);
  std::uniform_real_distribution<double> valDist(0.1, 1.0);
  std::uniform_int_distribution<int> coin(0, 1);

  std::vector<std::vector<std::pair<int, double>>> upper(n);
  std::vector<double> rowAbsSum(n, 0.0);
  for (int i = 0; i < n; ++i) {
    const int nEntries = 1 + static_cast<int>(rng() % avgNnzPerRow);
    for (int e = 0; e < nEntries; ++e) {
      const int j = colDist(rng);
      if (j <= i) continue;
      upper[i].push_back({j, valDist(rng)});
    }
  }
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < n; ++i) {
    for (auto& pr : upper[i]) {
      trips.emplace_back(i, pr.first, pr.second);
      trips.emplace_back(pr.first, i, pr.second);
      rowAbsSum[i] += std::abs(pr.second);
      rowAbsSum[pr.first] += std::abs(pr.second);
    }
  }
  for (int i = 0; i < n; ++i) {
    double diag = rowAbsSum[i] + 1.0 + (i % 5) * 0.1;
    if (coin(rng) == 0) diag = -diag;
    trips.emplace_back(i, i, diag);
  }
  Sparse A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

Sparse randomSparseKKT(int n1, int n2, unsigned seed) {
  std::mt19937 rng(seed);
  auto spd = [&](int n, unsigned s) {
    Sparse S = randomSparseIndefinite(n, 5, s);
    // force SPD-ish: flip signs back positive by taking abs on diagonal.
    for (int k = 0; k < S.outerSize(); ++k)
      for (Sparse::InnerIterator it(S, k); it; ++it)
        if (it.row() == it.col()) it.valueRef() = std::abs(it.value());
    return S;
  };
  Sparse E = spd(n1, seed);
  Sparse F = spd(n2, seed + 1);
  std::uniform_int_distribution<int> colDist(0, n1 - 1);
  std::uniform_real_distribution<double> valDist(-0.5, 0.5);
  std::vector<Eigen::Triplet<double>> trips;
  for (int c = 0; c < E.outerSize(); ++c)
    for (Sparse::InnerIterator it(E, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(it.row(), it.col(), -it.value());
  for (int c = 0; c < F.outerSize(); ++c)
    for (Sparse::InnerIterator it(F, c); it; ++it)
      if (it.row() >= it.col()) trips.emplace_back(n1 + it.row(), n1 + it.col(), it.value());
  for (int i = 0; i < n2; ++i) {
    const int nEntries = 3;
    for (int e = 0; e < nEntries; ++e) {
      const int j = colDist(rng);
      trips.emplace_back(n1 + i, j, valDist(rng));
    }
  }
  Sparse K(n1 + n2, n1 + n2);
  K.setFromTriplets(trips.begin(), trips.end());
  return K;
}

// --- Result recording ---

struct SolverTiming {
  std::string solver;
  bool ran = false;
  bool ok = false;
  std::string note;
  double analyzeSec = 0.0;
  double factorizeSec = 0.0;
  double solveSec = 0.0;
  double residual = -1.0;
};

std::vector<std::string> csvHeader() {
  return {"group",  "name",         "n",       "nnz",     "solver",       "ok",
          "note",   "analyze_sec",  "factorize_sec", "solve_sec", "residual", "vmhwm_kb"};
}

void writeRow(std::ofstream& out, const MatrixCase& mc, const SolverTiming& t, long vmhwm) {
  out << mc.group << "," << mc.name << "," << mc.A.rows() << "," << mc.A.nonZeros() << "," << t.solver << ","
      << (t.ok ? 1 : 0) << "," << "\"" << t.note << "\"" << "," << t.analyzeSec << "," << t.factorizeSec << ","
      << t.solveSec << "," << t.residual << "," << vmhwm << "\n";
  out.flush();
}

// --- Per-solver benchmark functions ---

SolverTiming benchSymla(const MatrixCase& mc, int nrhs, unsigned seed) {
  SolverTiming t;
  t.solver = "symla::SymLDLT";
  t.ran = true;
  try {
    symla::SymLDLT<double> solver;
    const auto t0 = Clock::now();
    solver.analyzePattern(mc.A);
    t.analyzeSec = seconds_since(t0);

    const auto t1 = Clock::now();
    solver.factorize(mc.A);
    t.factorizeSec = seconds_since(t1);

    if (solver.isSingular()) {
      t.ok = false;
      t.note = "singular";
      return t;
    }

    const int n = static_cast<int>(mc.A.rows());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = mc.A.selfadjointView<Eigen::Lower>() * Xtrue;

    const auto t2 = Clock::now();
    Eigen::MatrixXd X = solver.solve(B);
    t.solveSec = seconds_since(t2);

    t.residual = residualSelfAdjoint(mc.A, X, B);
    t.ok = t.residual < 1e-6;
    if (!t.ok) t.note = "large residual";
  } catch (const std::exception& e) {
    t.ok = false;
    t.note = std::string("exception: ") + e.what();
  }
  return t;
}

SolverTiming benchSimplicialLDLT(const MatrixCase& mc, int nrhs, unsigned seed) {
  SolverTiming t;
  t.solver = "Eigen::SimplicialLDLT";
  t.ran = true;
  try {
    Eigen::SimplicialLDLT<Sparse, Eigen::Lower> solver;
    const auto t0 = Clock::now();
    solver.analyzePattern(mc.A);
    t.analyzeSec = seconds_since(t0);
    const auto t1 = Clock::now();
    solver.factorize(mc.A);
    t.factorizeSec = seconds_since(t1);
    if (solver.info() != Eigen::Success) {
      t.ok = false;
      t.note = "factorize failed (likely not SPD -- SimplicialLDLT has no pivoting)";
      return t;
    }
    const int n = static_cast<int>(mc.A.rows());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = mc.A.selfadjointView<Eigen::Lower>() * Xtrue;
    const auto t2 = Clock::now();
    Eigen::MatrixXd X = solver.solve(B);
    t.solveSec = seconds_since(t2);
    t.residual = residualSelfAdjoint(mc.A, X, B);
    t.ok = t.residual < 1e-6;
    if (!t.ok) t.note = "large residual";
  } catch (const std::exception& e) {
    t.ok = false;
    t.note = std::string("exception: ") + e.what();
  }
  return t;
}

SolverTiming benchSparseLU(const MatrixCase& mc, int nrhs, unsigned seed) {
  SolverTiming t;
  t.solver = "Eigen::SparseLU";
  t.ran = true;
  try {
    Sparse Afull = fullSymmetric(mc.A);
    Eigen::SparseLU<Sparse> solver;
    const auto t0 = Clock::now();
    solver.analyzePattern(Afull);
    t.analyzeSec = seconds_since(t0);
    const auto t1 = Clock::now();
    solver.factorize(Afull);
    t.factorizeSec = seconds_since(t1);
    if (solver.info() != Eigen::Success) {
      t.ok = false;
      t.note = "factorize failed";
      return t;
    }
    const int n = static_cast<int>(mc.A.rows());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = Afull * Xtrue;
    const auto t2 = Clock::now();
    Eigen::MatrixXd X = solver.solve(B);
    t.solveSec = seconds_since(t2);
    t.residual = (Afull * X - B).norm() / (std::max(1.0, X.norm()) + B.norm());
    t.ok = t.residual < 1e-6;
    if (!t.ok) t.note = "large residual";
  } catch (const std::exception& e) {
    t.ok = false;
    t.note = std::string("exception: ") + e.what();
  }
  return t;
}

#ifdef SYMLA_HAVE_CHOLMOD
SolverTiming benchCholmod(const MatrixCase& mc, int nrhs, unsigned seed) {
  SolverTiming t;
  t.solver = "CHOLMOD (SupernodalLLT)";
  if (!mc.isSPD) {
    t.note = "skipped (SPD-only baseline)";
    return t;
  }
  t.ran = true;
  try {
    Eigen::CholmodSupernodalLLT<Sparse, Eigen::Lower> solver;
    const auto t0 = Clock::now();
    solver.analyzePattern(mc.A);
    t.analyzeSec = seconds_since(t0);
    const auto t1 = Clock::now();
    solver.factorize(mc.A);
    t.factorizeSec = seconds_since(t1);
    if (solver.info() != Eigen::Success) {
      t.ok = false;
      t.note = "factorize failed";
      return t;
    }
    const int n = static_cast<int>(mc.A.rows());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = mc.A.selfadjointView<Eigen::Lower>() * Xtrue;
    const auto t2 = Clock::now();
    Eigen::MatrixXd X = solver.solve(B);
    t.solveSec = seconds_since(t2);
    t.residual = residualSelfAdjoint(mc.A, X, B);
    t.ok = t.residual < 1e-6;
    if (!t.ok) t.note = "large residual";
  } catch (const std::exception& e) {
    t.ok = false;
    t.note = std::string("exception: ") + e.what();
  }
  return t;
}
#endif

#ifdef SYMLA_HAVE_UMFPACK
SolverTiming benchUmfpack(const MatrixCase& mc, int nrhs, unsigned seed) {
  SolverTiming t;
  t.solver = "UMFPACK (unsymmetric LU baseline)";
  t.ran = true;
  try {
    Sparse Afull = fullSymmetric(mc.A);
    Eigen::UmfPackLU<Sparse> solver;
    const auto t0 = Clock::now();
    solver.analyzePattern(Afull);
    t.analyzeSec = seconds_since(t0);
    const auto t1 = Clock::now();
    solver.factorize(Afull);
    t.factorizeSec = seconds_since(t1);
    if (solver.info() != Eigen::Success) {
      t.ok = false;
      t.note = "factorize failed";
      return t;
    }
    const int n = static_cast<int>(mc.A.rows());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Xtrue(n, nrhs);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < nrhs; ++j) Xtrue(i, j) = dist(rng);
    Eigen::MatrixXd B = Afull * Xtrue;
    const auto t2 = Clock::now();
    Eigen::MatrixXd X = solver.solve(B);
    t.solveSec = seconds_since(t2);
    t.residual = (Afull * X - B).norm() / (std::max(1.0, X.norm()) + B.norm());
    t.ok = t.residual < 1e-6;
    if (!t.ok) t.note = "large residual";
  } catch (const std::exception& e) {
    t.ok = false;
    t.note = std::string("exception: ") + e.what();
  }
  return t;
}
#endif

void printTiming(const MatrixCase& mc, const SolverTiming& t) {
  if (!t.ran) {
    std::cout << "    " << t.solver << ": " << (t.note.empty() ? "skipped" : t.note) << "\n";
    return;
  }
  std::cout << "    " << t.solver << ": analyze=" << t.analyzeSec << "s factorize=" << t.factorizeSec
            << "s solve=" << t.solveSec << "s residual=" << t.residual << (t.ok ? "" : "  [" + t.note + "]")
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const fs::path benchDir = fs::path(__FILE__).parent_path();
  const fs::path matricesRoot = benchDir / "matrices";
  const fs::path resultsDir = benchDir / "results";
  fs::create_directories(resultsDir);

  std::time_t now = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
  const fs::path resultsPath = resultsDir / (std::string("bench_") + buf + ".csv");
  std::ofstream csv(resultsPath);
  {
    auto hdr = csvHeader();
    for (std::size_t i = 0; i < hdr.size(); ++i) csv << hdr[i] << (i + 1 < hdr.size() ? "," : "\n");
  }

  std::vector<MatrixCase> cases;

  // Real matrices, if fetched.
  struct Spec {
    std::string group, name;
    bool isSPD;
  };
  const std::vector<Spec> specs = {
      {"GHS_indef", "sit100", false},   {"GHS_indef", "tuma2", false},    {"GHS_indef", "ncvxqp1", false},
      {"GHS_indef", "tuma1", false},    {"GHS_indef", "qpband", false},   {"GHS_indef", "bratu3d", false},
      {"GHS_indef", "c-55", false},     {"GHS_indef", "aug3dcqp", false}, {"GHS_indef", "stokes128", false},
      {"GHS_indef", "dawson5", false},  {"GHS_indef", "blockqp1", false}, {"GHS_indef", "cont-201", false},
      {"GHS_indef", "turon_m", false},  {"GHS_indef", "helm2d03", false}, {"Schenk_IBMNA", "c-18", false},
      {"Schenk_IBMNA", "c-22", false},  {"Schenk_IBMNA", "c-26", false},  {"Schenk_IBMNA", "c-30", false},
      {"Schenk_IBMNA", "c-62", false},  {"Schenk_IBMNA", "c-67", false},  {"HB", "bcsstk14", true},
      {"HB", "bcsstk16", true},
  };
  // Phase 5 finding (see the final report and elimination_tree.hpp /
  // dense_kernel.hpp / multifrontal.hpp): symla's factorize() currently
  // scales very poorly on larger and/or AMD-adversarial (e.g. uniformly
  // random sparsity, which lacks the locality AMD's fill-reducing heuristic
  // relies on) matrices -- e.g. a random n=15000 indefinite matrix measured
  // ~55s in factorize() alone during this investigation, and several real
  // GHS_indef/Schenk_IBMNA matrices above n~25000-40000 did not finish
  // symbolic analysis within 90s. Rather than let a single matrix make this
  // whole benchmark run take (potentially) hours, both the real-matrix set
  // and the synthetic sizes below are capped to sizes that are known to
  // complete quickly, so this harness stays a fast, repeatable regression
  // tool; the *actual* observed scaling numbers (and which specific real
  // matrices are affected) are reported directly in the Phase 5 report as
  // the concrete evidence of this limitation for Phase 6 to fix.
  constexpr int kMaxN = 25000;
  bool anyRealFound = false;
  for (const auto& s : specs) {
    const fs::path mtxPath = matricesRoot / s.group / s.name / (s.name + ".mtx");
    if (!fs::exists(mtxPath)) continue;
    Sparse A;
    if (!Eigen::loadMarket(A, mtxPath.string()) || A.rows() != A.cols()) continue;
    if (A.rows() > kMaxN) {
      std::cout << "bench_main: skipping " << s.group << "/" << s.name << " (n=" << A.rows() << " > " << kMaxN
                << ", see the size-cap note above)\n";
      continue;
    }
    anyRealFound = true;
    MatrixCase mc;
    mc.group = s.group;
    mc.name = s.name;
    mc.A = A;
    mc.isSPD = s.isSPD;
    cases.push_back(std::move(mc));
  }
  if (!anyRealFound) {
    std::cout << "bench_main: no real matrices found under " << matricesRoot
              << " -- run `python3 bench/fetch_matrices.py` first. Continuing with synthetic matrices only.\n";
  }

  // Synthetic indefinite + KKT matrices for real perf testing, sized to
  // stay well inside symla's currently-fast regime (see size-cap note
  // above).
  {
    MatrixCase mc;
    mc.group = "synthetic";
    mc.name = "indefinite_n3000";
    mc.A = randomSparseIndefinite(3000, 6, 7777u);
    mc.isSynthetic = true;
    cases.push_back(std::move(mc));
  }
  {
    MatrixCase mc;
    mc.group = "synthetic";
    mc.name = "kkt_n3000";
    mc.A = randomSparseKKT(1800, 1200, 8888u);
    mc.isSynthetic = true;
    cases.push_back(std::move(mc));
  }
  {
    MatrixCase mc;
    mc.group = "synthetic";
    mc.name = "indefinite_n1000";
    mc.A = randomSparseIndefinite(1000, 6, 4242u);
    mc.isSynthetic = true;
    cases.push_back(std::move(mc));
  }

  const int nrhs = 3;
  for (const auto& mc : cases) {
    std::cout << "== " << mc.group << "/" << mc.name << "  n=" << mc.A.rows() << " nnz=" << mc.A.nonZeros()
              << " ==\n";

    auto tSymla = benchSymla(mc, nrhs, 111u);
    printTiming(mc, tSymla);
    writeRow(csv, mc, tSymla, vmHwmKb());

    auto tSimplicial = benchSimplicialLDLT(mc, nrhs, 222u);
    printTiming(mc, tSimplicial);
    writeRow(csv, mc, tSimplicial, vmHwmKb());

    // SparseLU / UMFPACK on very large matrices can be slow (dense-ish LU
    // fill on genuinely 2D/3D problems); still run them since they're the
    // requested baselines, but the caller can watch wall time.
    auto tSparseLU = benchSparseLU(mc, nrhs, 333u);
    printTiming(mc, tSparseLU);
    writeRow(csv, mc, tSparseLU, vmHwmKb());

#ifdef SYMLA_HAVE_CHOLMOD
    auto tCholmod = benchCholmod(mc, nrhs, 444u);
    printTiming(mc, tCholmod);
    writeRow(csv, mc, tCholmod, vmHwmKb());
#endif
#ifdef SYMLA_HAVE_UMFPACK
    auto tUmfpack = benchUmfpack(mc, nrhs, 555u);
    printTiming(mc, tUmfpack);
    writeRow(csv, mc, tUmfpack, vmHwmKb());
#endif
  }

  std::cout << "\nbench_main: results written to " << resultsPath << "\n";
  (void)argc;
  (void)argv;
  return 0;
}
