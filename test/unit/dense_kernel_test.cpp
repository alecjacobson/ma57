// Unit tests for symla::DenseLDLT (dense_kernel.hpp): the dense
// Bunch-Kaufman-style threshold-pivoted symmetric indefinite LDL^T kernel.
//
// Covers:
//  1. Hand-verified small cases (3x3-5x5) with worked-out expected pivot
//     sequence, including a matrix that forces a 2x2 pivot by construction.
//  2. Residual tests on random dense symmetric indefinite matrices at
//     several sizes, reconstructing P A P^T from L/D/perm.
//  3. Cross-check against LAPACK dsytrf/dsytrs as an independent oracle:
//     compare solve residual and inertia (not literal pivot sequence).
//  4. Growth factor sanity check.
//  5. Delayed-pivot path (degenerate/rank-deficient trailing block).
//
// LAPACK is used here purely as a black-box numerical oracle via its
// public Fortran ABI (dsytrf_/dsytrs_); no LAPACK source was read.

#include "symla/dense_kernel.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using symla::DenseLDLT;
using symla::DenseLDLTOptions;
using symla::DenseLDLTResult;
using symla::PivotKind;

extern "C" {
void dsytrf_(char* uplo, int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
void dsytrs_(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
}

namespace {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

// Reconstructs P A P^T from the factored A (strictly-lower = L multipliers),
// D_out, and perm, where the convention is A_permuted(i,j) = A_orig(perm(i),
// perm(j)) i.e. perm maps *new* index -> *original* index. Only the leading
// `nf` rows/cols (the successfully factored part) are reconstructed.
//
// Note: within a 2x2 pivot block occupying columns (k, k+1), the strictly-
// lower entry at (k+1, k) of `factored` holds D's off-diagonal d21, *not*
// an L multiplier (L's diagonal blocks are always identity), so that
// position must be excluded when extracting L.
Mat reconstructPAPT(const Mat& factored, const symla::BlockDiagonalD<double>& D, int nf,
                     const std::vector<symla::PivotBlock>& pivots) {
  Mat L = Mat::Identity(nf, nf);
  for (int j = 0; j < nf; ++j)
    for (int i = j + 1; i < nf; ++i) L(i, j) = factored(i, j);
  for (const auto& pb : pivots) {
    if (pb.kind == symla::PivotKind::TwoByTwo && pb.start + 1 < nf) L(pb.start + 1, pb.start) = 0.0;
  }
  Mat Dsub = D.head(nf).toDense();
  return L * Dsub * L.transpose();
}

// Applies the permutation to the leading nf x nf block of A: returns
// Aperm(i,j) = A(perm(i), perm(j)) for i,j < nf.
Mat applyPerm(const Mat& A, const Eigen::VectorXi& perm, int nf) {
  Mat out(nf, nf);
  for (int i = 0; i < nf; ++i)
    for (int j = 0; j < nf; ++j) out(i, j) = A(perm(i), perm(j));
  return out;
}

// Tiny forward/diag(block)/back substitution solver using the DenseLDLT
// output directly, for full-size (n_factored == n) factorizations. Solves
// A x = b given the *original* (unpermuted) A implicitly via perm.
Vec solveFromFactorization(const Mat& factored, const symla::BlockDiagonalD<double>& D, const DenseLDLTResult& res,
                            const Vec& b) {
  const int n = static_cast<int>(b.size());
  SYMLA_CHECK(res.n_factored == n);
  // Permute b: bp(i) = b(perm(i))
  Vec bp(n);
  for (int i = 0; i < n; ++i) bp(i) = b(res.perm(i));

  // Note: within a 2x2 pivot block at (k, k+1), position (k+1, k) of
  // `factored` holds D's off-diagonal d21, not an L multiplier -- skip it
  // when walking L (see reconstructPAPT's comment above for the same
  // caveat).
  std::vector<bool> isD21(n, false);
  for (const auto& pb : res.pivots)
    if (pb.kind == PivotKind::TwoByTwo) isD21[pb.start + 1] = true;

  // Forward solve L y = bp (L unit lower triangular, strictly-lower stored in `factored`).
  Vec y = bp;
  for (int j = 0; j < n; ++j)
    for (int i = j + 1; i < n; ++i) {
      if (isD21[i] && i == j + 1) continue;  // (k+1,k) slot is d21, not L
      y(i) -= factored(i, j) * y(j);
    }

  // Block-diagonal solve D z = y, per pivot.
  Vec z(n);
  for (const auto& pb : res.pivots) {
    if (pb.kind == PivotKind::OneByOne) {
      z(pb.start) = y(pb.start) / D(pb.start, pb.start);
    } else {
      const int k = pb.start;
      const double d11 = D(k, k), d21 = D(k + 1, k), d22 = D(k + 1, k + 1);
      const double det = d11 * d22 - d21 * d21;
      const double y0 = y(k), y1 = y(k + 1);
      z(k) = (d22 * y0 - d21 * y1) / det;
      z(k + 1) = (-d21 * y0 + d11 * y1) / det;
    }
  }

  // Back solve L^T w = z.
  Vec w = z;
  for (int j = n - 1; j >= 0; --j)
    for (int i = j + 1; i < n; ++i) {
      if (isD21[i] && i == j + 1) continue;
      w(j) -= factored(i, j) * w(i);
    }

  // Un-permute: x(perm(i)) = w(i).
  Vec x(n);
  for (int i = 0; i < n; ++i) x(res.perm(i)) = w(i);
  return x;
}

// Q * diag(+/- eigs) * Q^T via QR of a random matrix for Q.
Mat randomSymmetricIndefiniteEig(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  Mat G(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) G(i, j) = dist(rng);
  Eigen::HouseholderQR<Mat> qr(G);
  Mat Q = qr.householderQ();
  Vec eigs(n);
  std::uniform_real_distribution<double> mag(0.5, 5.0);
  for (int i = 0; i < n; ++i) eigs(i) = (i % 2 == 0 ? 1.0 : -1.0) * mag(rng);
  return Q * eigs.asDiagonal() * Q.transpose();
}

int lapackInertia(const Mat& Aoriginal, Vec& b, Vec& x_lapack, double& residual) {
  const int n = static_cast<int>(Aoriginal.rows());
  // LAPACK dsytrf expects column-major Fortran storage; Eigen::MatrixXd is
  // already column-major by default, and we pass 'L' (lower) uplo, so we
  // can hand it the raw data directly for the lower triangle.
  Mat A = Aoriginal;  // copy, column-major
  std::vector<int> ipiv(n);
  int info = 0;
  char uplo = 'L';
  int lwork = -1;
  double wkopt;
  int nn = n, lda = n;
  dsytrf_(&uplo, &nn, A.data(), &lda, ipiv.data(), &wkopt, &lwork, &info);
  lwork = static_cast<int>(wkopt);
  std::vector<double> work(std::max(lwork, 1));
  dsytrf_(&uplo, &nn, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
  SYMLA_CHECK(info >= 0);  // info>0 means exactly singular pivot at that index; still interpretable

  // Interpret ipiv/block-diagonal of A to get inertia (standard LAPACK
  // dsytrf output convention, documented publicly): for uplo='L', scanning
  // k = 1..n (1-based): if ipiv(k) > 0, it's a 1x1 pivot at k with value
  // A(k,k). If ipiv(k) == ipiv(k+1) < 0, it's a 2x2 pivot occupying (k,k+1)
  // with block [[A(k,k), A(k+1,k)],[A(k+1,k), A(k+1,k+1)]].
  int n_pos = 0, n_neg = 0, n_zero = 0;
  for (int k = 0; k < n;) {
    if (ipiv[k] > 0) {
      double d = A(k, k);
      if (d > 0)
        ++n_pos;
      else if (d < 0)
        ++n_neg;
      else
        ++n_zero;
      ++k;
    } else {
      // 2x2 block at k,k+1
      double d11 = A(k, k), d21 = A(k + 1, k), d22 = A(k + 1, k + 1);
      double det = d11 * d22 - d21 * d21;
      double tr = d11 + d22;
      if (det < 0) {
        ++n_pos;
        ++n_neg;
      } else if (tr > 0) {
        n_pos += 2;
      } else {
        n_neg += 2;
      }
      k += 2;
    }
  }

  // Solve A x = b using the LAPACK factorization as a residual cross-check.
  Mat Bx(n, 1);
  Bx.col(0) = b;
  int nrhs = 1, ldb = n;
  dsytrs_(&uplo, &nn, &nrhs, A.data(), &lda, ipiv.data(), Bx.data(), &ldb, &info);
  SYMLA_CHECK(info == 0);
  x_lapack = Bx.col(0);
  residual = (Aoriginal * x_lapack - b).norm() /
             (Aoriginal.cwiseAbs().maxCoeff() * std::max(1.0, x_lapack.norm()) + b.norm() + 1e-300);

  (void)n_zero;
  return n_pos * 1000 + n_neg;  // pack for easy comparison; also return via out-params if needed
}

void testHandVerified3x3Diagonal() {
  // A already diagonally dominant on the diagonal -> every step should
  // accept a 1x1 pivot with no swaps (lambda small relative to alpha*A(k,k)).
  Mat A(3, 3);
  A << 10, 1, 1,
        1, 8, 1,
        1, 1, 6;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D);
  SYMLA_CHECK(res.n_factored == 3);
  SYMLA_CHECK(static_cast<int>(res.pivots.size()) == 3);
  for (auto& pb : res.pivots) SYMLA_CHECK(pb.kind == PivotKind::OneByOne);
  for (int i = 0; i < 3; ++i) SYMLA_CHECK(res.perm(i) == i);  // no swaps expected

  Mat A0(3, 3);
  A0 << 10, 1, 1, 1, 8, 1, 1, 1, 6;
  Mat rec = reconstructPAPT(A, D, 3, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, 3);
  SYMLA_CHECK((rec - Ap).norm() / A0.norm() < 1e-12);
  SYMLA_CHECK(res.inertia.n_pos == 3 && res.inertia.n_neg == 0);
}

void testHandVerifiedForces2x2() {
  // [[0,1],[1,0]] embedded: A(0,0)=0 forces a swap/2x2 decision by
  // construction. With A(0,0)=0, lambda = |A(1,0)| = 1 (r=1). Since
  // |A(0,0)| = 0 < alpha*lambda, go to the sigma test: sigma = max over
  // column r=1 excluding row 1 itself among rows {0,1} -> only row 0,
  // |A(0,1)| = 1 (this is position (k,r)) -> per our "sigma over currently
  // relevant rows [k,n) excluding r" rule sigma = |A(0,1)| = 1. Then
  // |A(0,0)|*sigma = 0 < alpha*lambda^2 = alpha. Next: |A(1,1)| = 0 >=
  // alpha*sigma = alpha? No (0 < alpha). So we fall to accept a 2x2 pivot
  // at (0,1). By hand: D2 = [[0,1],[1,0]], det = -1 (indefinite, one +, one -).
  Mat A(3, 3);
  A << 0, 1, 0.2,
       1, 0, 0.3,
       0.2, 0.3, 5.0;
  Mat A0 = A;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D);
  SYMLA_CHECK(res.n_factored == 3);
  SYMLA_CHECK(!res.pivots.empty());
  SYMLA_CHECK(res.pivots[0].kind == PivotKind::TwoByTwo);
  SYMLA_CHECK(res.pivots[0].start == 0);
  SYMLA_CHECK(res.inertia.n_pos == 2 && res.inertia.n_neg == 1);  // 2x2 block: 1+,1-; plus trailing 5.x>0 pivot

  Mat rec = reconstructPAPT(A, D, 3, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, 3);
  SYMLA_CHECK((rec - Ap).norm() / A0.norm() < 1e-10);
}

void testHandVerified5x5Mixed() {
  // Construct a matrix with a clear dominant-diagonal leading part (1x1
  // pivots) followed by an embedded zero-diagonal 2x2-forcing pair.
  Mat A(5, 5);
  A << 20, 1, 0.5, 0.1, 0.1,
        1, 15, 0.2, 0.1, 0.1,
        0.5, 0.2, 0, 2, 0.1,
        0.1, 0.1, 2, 0, 0.2,
        0.1, 0.1, 0.1, 0.2, 12;
  Mat A0 = A;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D);
  SYMLA_CHECK(res.n_factored == 5);
  Mat rec = reconstructPAPT(A, D, 5, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, 5);
  SYMLA_CHECK((rec - Ap).norm() / A0.norm() < 1e-10);
  // Total inertia should have exactly one negative eigenvalue-sign
  // contribution from the embedded 2x2 indefinite block, rest positive-ish;
  // just check counts sum to n and are plausible (no zero pivots).
  SYMLA_CHECK(res.inertia.n_pos + res.inertia.n_neg == 5);
  SYMLA_CHECK(res.inertia.n_zero == 0);
}

void testResidualRandom(int n, unsigned seed) {
  Mat A0 = randomSymmetricIndefiniteEig(n, seed);
  Mat A = A0;
  symla::BlockDiagonalD<double> D;
  DenseLDLTOptions opts;
  auto res = DenseLDLT<double>::factor(A, D, opts);
  SYMLA_CHECK(res.n_factored == n);
  SYMLA_CHECK(static_cast<int>(res.delayed_cols.empty()));

  Mat rec = reconstructPAPT(A, D, n, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, n);
  double relErr = (rec - Ap).norm() / A0.norm();
  SYMLA_CHECK(relErr < 1e-10);

  std::mt19937 rng(seed + 777);
  std::normal_distribution<double> dist(0.0, 1.0);
  Vec b(n);
  for (int i = 0; i < n; ++i) b(i) = dist(rng);
  Vec x = solveFromFactorization(A, D, res, b);
  double resid = (A0 * x - b).norm() / (A0.cwiseAbs().maxCoeff() * x.norm() + b.norm());
  SYMLA_CHECK(resid < 1e-8);

  SYMLA_CHECK(res.max_growth < 1000.0);  // growth-factor smoke check
}

void testLapackCrossCheck(int n, unsigned seed) {
  Mat A0 = randomSymmetricIndefiniteEig(n, seed);
  Mat A = A0;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D);
  SYMLA_CHECK(res.n_factored == n);

  std::mt19937 rng(seed + 42);
  std::normal_distribution<double> dist(0.0, 1.0);
  Vec b(n);
  for (int i = 0; i < n; ++i) b(i) = dist(rng);

  Vec x_ours = solveFromFactorization(A, D, res, b);
  double resid_ours = (A0 * x_ours - b).norm() / (A0.cwiseAbs().maxCoeff() * x_ours.norm() + b.norm());

  Vec x_lapack;
  double resid_lapack;
  int packed = lapackInertia(A0, b, x_lapack, resid_lapack);
  int lapack_pos = packed / 1000;
  int lapack_neg = packed % 1000;

  SYMLA_CHECK(resid_ours < 1e-8);
  SYMLA_CHECK(resid_lapack < 1e-8);
  // Same order of magnitude.
  SYMLA_CHECK(resid_ours < 1e-6 * std::max(resid_lapack, 1e-14) + 1e-8);

  SYMLA_CHECK(res.inertia.n_pos == lapack_pos);
  SYMLA_CHECK(res.inertia.n_neg == lapack_neg);
}

void testDelayedPivotAllZeroTrailingBlock() {
  // Leading 3x3 well-conditioned SPD block, trailing 2x2 all-zero block
  // fully decoupled (no coupling entries) -> once the SPD part is
  // factored, the trailing 2x2 all-zero block has lambda==0 and
  // A(k,k)==0, which must trigger delay rather than division by zero.
  const int n = 5;
  Mat A = Mat::Zero(n, n);
  A(0, 0) = 10;
  A(1, 1) = 8;
  A(2, 2) = 6;
  A(0, 1) = A(1, 0) = 1;
  A(0, 2) = A(2, 0) = 0.5;
  A(1, 2) = A(2, 1) = 0.2;
  // rows/cols 3,4 (and their coupling to 0..2) are all zero -> structurally singular trailing block.
  Mat A0 = A;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D);

  SYMLA_CHECK(res.n_factored == 3);
  SYMLA_CHECK(static_cast<int>(res.delayed_cols.size()) == 2);
  SYMLA_CHECK(res.delayed_cols[0] == 3 && res.delayed_cols[1] == 4);

  // The successfully-factored leading 3x3 part must still be numerically
  // correct: reconstruct just that submatrix.
  Mat rec = reconstructPAPT(A, D, 3, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, 3);
  SYMLA_CHECK((rec - Ap).norm() / A0.topLeftCorner(3, 3).norm() < 1e-10);
}

void testDelayedPivotRankDeficient2x2() {
  // Force a degenerate 2x2 candidate: after a first 1x1 pivot on a strong
  // diagonal entry, leave a genuinely singular (rank-deficient, zero
  // determinant) 2x2 trailing block with zero diagonal and equal
  // off-diagonal magnitude structured so the Bunch-Kaufman test selects a
  // 2x2 pivot whose determinant is ~0.
  const int n = 3;
  Mat A(n, n);
  A << 100, 0, 0,
       0, 0, 1e-20,
       0, 1e-20, 2e-20;  // trailing 2x2 block: [[0,1e-20],[1e-20,2e-20]], det ~ -1e-40, tiny but not exactly 0.
  // Use a generous zero_tolerance to force this to be treated as degenerate.
  DenseLDLTOptions opts;
  opts.zero_tolerance = 1e-8;
  Mat A0 = A;
  symla::BlockDiagonalD<double> D;
  auto res = DenseLDLT<double>::factor(A, D, opts);

  SYMLA_CHECK(res.n_factored == 1);
  SYMLA_CHECK(static_cast<int>(res.delayed_cols.size()) == 2);
  Mat rec = reconstructPAPT(A, D, 1, res.pivots);
  Mat Ap = applyPerm(A0, res.perm, 1);
  SYMLA_CHECK(std::abs(rec(0, 0) - Ap(0, 0)) < 1e-8);
}

}  // namespace

int main() {
  testHandVerified3x3Diagonal();
  testHandVerifiedForces2x2();
  testHandVerified5x5Mixed();

  for (int n : {10, 50, 200}) {
    for (unsigned seed : {1u, 2u, 3u}) {
      testResidualRandom(n, seed * 1000u + static_cast<unsigned>(n));
    }
  }

  for (int n : {5, 10, 25, 50, 100}) {
    for (unsigned seed : {11u, 22u, 33u}) {
      testLapackCrossCheck(n, seed * 1000u + static_cast<unsigned>(n));
    }
  }

  testDelayedPivotAllZeroTrailingBlock();
  testDelayedPivotRankDeficient2x2();

  std::cout << "dense_kernel_test: all checks passed\n";
  return 0;
}
