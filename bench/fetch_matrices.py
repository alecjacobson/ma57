#!/usr/bin/env python3
"""Download a curated set of real symmetric indefinite / KKT test matrices
from the SuiteSparse Matrix Collection (https://sparse.tamu.edu) for symla's
regression/benchmark suite.

Groups covered:
  - GHS_indef: KKT matrices from the CUTEr optimization test suite (the
    canonical "real indefinite KKT/saddle-point" source). A dozen small-to-
    medium matrices (n < ~80k) plus two larger ones for real perf testing.
  - Schenk_IBMNA: PARDISO/IPOPT-family symmetric indefinite matrices from IBM
    applications.
  - HB (bcsstk*): a couple of classic SPD structural matrices, for
    CHOLMOD/SimplicialLDLT cross-checks (nice-to-have, not KKT-shaped).

Note on Mittelmann: the plan asked us to also consider the Mittelmann group
(LP/QP benchmark matrices). We checked several candidates there (nug08-3rd,
watson_1, rail507/516/582, cont1_l, ...) and every one of them turned out to
be a *rectangular* LP constraint matrix (num rows != num cols, MatrixMarket
"general" format) -- not a symmetric square matrix a symmetric indefinite
solver can consume. Rather than force a mismatched matrix into this suite, we
skip Mittelmann and get our "real indefinite KKT" coverage from GHS_indef
(which literally *is* CUTEr KKT systems) and Schenk_IBMNA instead.

Each matrix is downloaded as a Matrix Market tarball, extracted, and left in
bench/matrices/<Group>/<Name>/<Name>.mtx (plus any auxiliary files the
tarball contains, e.g. *_b.mtx). This directory is gitignored; nothing here
is meant to be committed. The script is idempotent: matrices that already
have an extracted .mtx file are skipped.

Usage:
    python3 bench/fetch_matrices.py [--dest DIR] [--only NAME [NAME ...]]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import urllib.error
import urllib.request

try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except ImportError:
    _HAVE_REQUESTS = False

# Note: plain HTTP, not HTTPS. HTTPS to sparse-files.engr.tamu.edu hangs /
# resets during the TLS handshake in this sandboxed environment (confirmed
# with both `curl` and `python3 requests`/`urllib`), while HTTP works
# reliably and quickly. sparse-files.engr.tamu.edu is a plain redirect
# target off the (also HTTP-capable) suitesparse-collection-website.
BASE_URL = "http://sparse-files.engr.tamu.edu/MM"

# (group, name)
MATRICES = [
    # --- GHS_indef: CUTEr KKT matrices, small-to-medium ---
    ("GHS_indef", "sit100"),
    ("GHS_indef", "tuma2"),
    ("GHS_indef", "ncvxqp1"),
    ("GHS_indef", "tuma1"),
    ("GHS_indef", "qpband"),
    ("GHS_indef", "bratu3d"),
    ("GHS_indef", "c-55"),
    ("GHS_indef", "aug3dcqp"),
    ("GHS_indef", "stokes128"),
    ("GHS_indef", "dawson5"),
    ("GHS_indef", "blockqp1"),
    ("GHS_indef", "cont-201"),
    # --- GHS_indef: larger, for real perf testing ---
    ("GHS_indef", "turon_m"),
    ("GHS_indef", "helm2d03"),
    # --- Schenk_IBMNA: PARDISO/IPOPT-family symmetric indefinite ---
    ("Schenk_IBMNA", "c-18"),
    ("Schenk_IBMNA", "c-22"),
    ("Schenk_IBMNA", "c-26"),
    ("Schenk_IBMNA", "c-30"),
    ("Schenk_IBMNA", "c-62"),
    ("Schenk_IBMNA", "c-67"),
    # --- HB: classic SPD structural matrices (CHOLMOD/SimplicialLDLT cross-check) ---
    ("HB", "bcsstk14"),
    ("HB", "bcsstk16"),
]


def tar_url(group: str, name: str) -> str:
    return f"{BASE_URL}/{group}/{name}.tar.gz"


def download(url: str, timeout: float = 120.0) -> bytes:
    if _HAVE_REQUESTS:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    req = urllib.request.Request(url, headers={"User-Agent": "symla-fetch-matrices/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def mtx_header_dims(path: str):
    """Return (n_rows, n_cols, nnz_stored) from a MatrixMarket file's first
    non-comment line, or None if unreadable."""
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                return None
    except OSError:
        return None
    return None


def fetch_one(group: str, name: str, dest_root: str) -> dict:
    out_dir = os.path.join(dest_root, group, name)
    mtx_path = os.path.join(out_dir, f"{name}.mtx")

    if os.path.isfile(mtx_path):
        dims = mtx_header_dims(mtx_path)
        status = "cached"
    else:
        url = tar_url(group, name)
        try:
            data = download(url)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return {"group": group, "name": name, "status": f"FAILED ({exc})", "n": None, "nnz": None}
        except Exception as exc:  # requests exceptions
            return {"group": group, "name": name, "status": f"FAILED ({exc})", "n": None, "nnz": None}

        os.makedirs(out_dir, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for member in tf.getmembers():
                    # Tarball layout is "<name>/<name>.mtx" (+ optional
                    # "<name>/<name>_b.mtx" etc). Flatten into out_dir.
                    base = os.path.basename(member.name)
                    if not base:
                        continue
                    if member.isdir():
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    with open(os.path.join(out_dir, base), "wb") as fh:
                        fh.write(extracted.read())
        except tarfile.TarError as exc:
            return {"group": group, "name": name, "status": f"FAILED (bad tar: {exc})", "n": None, "nnz": None}

        if not os.path.isfile(mtx_path):
            return {"group": group, "name": name, "status": "FAILED (no .mtx in tarball)", "n": None, "nnz": None}
        dims = mtx_header_dims(mtx_path)
        status = "downloaded"

    if dims is None:
        return {"group": group, "name": name, "status": "FAILED (unreadable .mtx)", "n": None, "nnz": None}
    n_rows, n_cols, nnz_stored = dims
    return {
        "group": group,
        "name": name,
        "status": status,
        "n": n_rows if n_rows == n_cols else f"{n_rows}x{n_cols}",
        "nnz": nnz_stored,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices"),
        help="destination directory (default: bench/matrices next to this script)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these matrix names")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    todo = MATRICES
    if args.only:
        wanted = set(args.only)
        todo = [(g, n) for (g, n) in MATRICES if n in wanted]
        if not todo:
            print(f"No matrices in the curated list match --only {args.only}", file=sys.stderr)
            return 1

    results = []
    for group, name in todo:
        print(f"[symla] fetching {group}/{name} ...", flush=True)
        r = fetch_one(group, name, args.dest)
        results.append(r)
        print(f"    -> {r['status']}", flush=True)

    print()
    print(f"{'group':<14} {'name':<12} {'n':>10} {'nnz(stored)':>12}  status")
    print("-" * 70)
    n_ok = 0
    for r in results:
        n_str = "" if r["n"] is None else str(r["n"])
        nnz_str = "" if r["nnz"] is None else str(r["nnz"])
        print(f"{r['group']:<14} {r['name']:<12} {n_str:>10} {nnz_str:>12}  {r['status']}")
        if r["status"] in ("downloaded", "cached"):
            n_ok += 1
    print("-" * 70)
    print(f"{n_ok}/{len(results)} matrices available in {args.dest}")

    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
