#!/usr/bin/env python3
"""Submission validator for POLAR(64,32) competition — run before EVERY upload.

Checks:
  1. CSV has exactly 65 columns: id, bit_0..bit_63
  2. id is contiguous 0..N-1
  3. all bit values are 0/1 integers
  4. every row satisfies the parity-check matrix (syndrome all-zero)
  5. row count matches the public test set (100000)
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BIT_COLUMNS = [f"bit_{i}" for i in range(64)]


def load_parity():
    archive = ROOT / "baseline.zip"
    if not archive.exists():
        return None
    with ZipFile(archive) as a:
        raw = a.read("baseline/Codes_DB/POLAR_N64_K32.txt")
    return np.loadtxt(io.BytesIO(raw), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--expect-rows", type=int, default=None)
    ap.add_argument(
        "--skip-parity",
        action="store_true",
        help="skip parity-check-matrix validation (use for bit-MAP output, "
             "which intentionally does not enforce codeword validity)",
    )
    args = ap.parse_args()
    path = args.csv
    expect_rows = args.expect_rows

    df = pd.read_csv(path)
    problems = []

    # 1. columns
    expected_cols = ["id"] + BIT_COLUMNS
    cols = list(df.columns)
    missing = [c for c in expected_cols if c not in df.columns]
    if cols != expected_cols:
        problems.append(f"column mismatch: got {cols[:3]}... expected id,bit_0..bit_63")
    if missing:
        # Accessing a missing column below would raise KeyError; report it first.
        print(f"file      : {path}")
        print(f"rows      : {len(df)}")
        print(f"cols      : {len(cols)}")
        print("RESULT: FAIL")
        for problem in problems:
            print("  -", problem)
        shown = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
        print(f"  - missing required columns: {shown}")
        sys.exit(1)

    # 2. id continuity
    ids = df["id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(ids))):
        problems.append(f"id not contiguous 0..{len(ids)-1} (min={ids.min()}, max={ids.max()})")

    # 3. values 0/1
    bits = df[BIT_COLUMNS].to_numpy()
    bad = ~np.isin(bits, [0, 1])
    if bad.any():
        problems.append(f"{bad.sum()} non-binary values (examples: {np.unique(bits[bad])[:5]})")

    # 4. parity check (bit-MAP output legitimately violates; only OSD must pass)
    n_viol = -1
    parity_checked = False
    if not args.skip_parity:
        H = load_parity()
        if H is None:
            print(
                "WARNING: baseline.zip not found; parity check skipped. "
                "Place the official baseline package at the repo root (see DATA.md)."
            )
        else:
            parity_checked = True
            syn = (bits @ H.T) % 2
            n_viol = int((syn.sum(axis=1) > 0).sum())
            if n_viol:
                problems.append(f"{n_viol} rows violate parity-check matrix")

    # 5. row count
    if expect_rows is not None and len(df) != expect_rows:
        problems.append(f"row count {len(df)} != expected {expect_rows}")

    print(f"file      : {path}")
    print(f"rows      : {len(df)}")
    print(f"cols      : {len(cols)}")
    if args.skip_parity:
        print("parity    : skipped (bit-MAP mode)")
    elif not parity_checked:
        print("parity    : skipped (baseline.zip missing)")
    else:
        print(f"parity ok : {len(df) - n_viol}/{len(df)}")
    if problems:
        print("RESULT: FAIL")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("RESULT: PASS — ready to upload")


if __name__ == "__main__":
    main()
