#!/usr/bin/env python3
"""
Vectorized bit-MAP list decoder (fast version of bit_map_decoder.py).

Key optimizations vs the CPU version:
  1. candidate generation: replace the per-pattern Python loop with a batched
     XOR over a (P,3) pattern-index matrix  ->  P=5489 candidates at once
  2. bit-MAP posterior does NOT depend on candidate order, so we skip the
     per-candidate cost sort entirely (only needed for the OSD top-1 baseline,
     computed via np.argmin on the cost vector)
  3. Euclidean distance / log-sum-exp posterior computed with vectorized numpy
     (float32), batched over samples

Modes:
  --validate : labeled training-set check (compare OSD top1 vs bit-MAP BER)
  --infer    : public_test.zip -> submission CSV
"""
from __future__ import annotations

import argparse
import io
import time
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BIT_COLUMNS = [f"bit_{i}" for i in range(64)]
Y_COLUMNS = [f"y_{i}" for i in range(64)]
_ZERO_ROW = np.zeros(64, dtype=np.uint8)


# --------------------------------------------------------------------------
def rref_binary(mat):
    red = mat.copy().astype(np.uint8)
    piv = []
    r = 0
    m, n = mat.shape
    for c in range(n):
        cand = np.flatnonzero(red[r:, c])
        if cand.size == 0:
            continue
        s = r + int(cand[0])
        red[[r, s]] = red[[s, r]]
        for o in np.flatnonzero(red[:, c]):
            if o != r:
                red[o] ^= red[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    return red, piv


def load_matrices():
    with ZipFile(ROOT / "baseline.zip") as a:
        raw = a.read("baseline/Codes_DB/POLAR_N64_K32.txt")
    H = np.loadtxt(io.BytesIO(raw), dtype=np.uint8)
    _, piv = rref_binary(H)
    free = [i for i in range(64) if i not in piv]
    G = np.zeros((32, 64), dtype=np.uint8)
    G[np.arange(32), free] = 1
    G[:, piv] = _[:, free].T
    return H, G


def make_pattern_matrix(list_size, order):
    """Return (P,3) int32 matrix of row indices; -1 means 'no flip'.
    Also return P (number of patterns)."""
    pats = [()]
    for size in range(1, order + 1):
        pats.extend(combinations(range(list_size), size))
    P = len(pats)
    pat = np.full((P, 3), -1, dtype=np.int32)
    for j, p in enumerate(pats):
        for k, idx in enumerate(p):
            pat[j, k] = idx
    return pat, P


def gen_candidates_fast(llr, G, pat):
    """Vectorized OSD order-3 candidate generation.
    Returns cand_orig (P,64) uint8 in original coordinate space,
            cost (P,) float64 awgn cost, hard_perm (64,) hard bits in permuted space.
    """
    co = np.argsort(-np.abs(llr))
    sys = G[:, co].copy().astype(np.uint8)
    pc = []
    pr = 0
    for col in range(64):
        cand = np.flatnonzero(sys[pr:, col])
        if cand.size == 0:
            continue
        s = pr + int(cand[0])
        sys[[pr, s]] = sys[[s, pr]]
        for o in np.flatnonzero(sys[:, col]):
            if o != pr:
                sys[o] ^= sys[pr]
        pc.append(col)
        pr += 1
        if pr == 32:
            break
    hp = (llr[co] < 0.0).astype(np.uint8)
    pc_arr = np.array(pc, dtype=np.int32)
    msg = hp[pc_arr]
    base = (msg @ sys) % 2  # (64,) in permuted space
    pr2 = np.abs(llr[co[pc_arr]])
    lr = np.argsort(pr2)[: 32]  # all 32 info rows sorted by reliability
    rows = sys[lr]  # (32,64)

    # batched flip: rows_all[32] = zero row, pat's -1 -> 32
    rows_all = np.vstack([rows, _ZERO_ROW])  # (33,64)
    p = pat.copy()
    p[p < 0] = 32
    # (P,64) = base ^ row[p0] ^ row[p1] ^ row[p2]
    cand_perm = base[None, :] ^ rows_all[p[:, 0]] ^ rows_all[p[:, 1]] ^ rows_all[p[:, 2]]

    # map back to original coordinates
    cand_orig = np.empty((cand_perm.shape[0], 64), dtype=np.uint8)
    cand_orig[:, co] = cand_perm

    # awgn cost in permuted space (|llr| weighting)
    w = np.abs(llr[co]).astype(np.float64)
    cost = ((cand_perm != hp[None, :]).astype(np.float64) * w[None, :]).sum(axis=1)
    return cand_orig, cost


def bitmap_from_candidates(y, cand, sigma2):
    """cand: (P,64) uint8. Returns bit-MAP bits (64,) and OSD top-1 (64,)."""
    mu = 1.0 - 2.0 * cand.astype(np.float32)          # (P,64)
    d2 = ((y.astype(np.float32)[None, :] - mu) ** 2).sum(axis=1)  # (P,)
    logw = -d2.astype(np.float64) / (2.0 * sigma2)
    mx = logw.max()
    w = np.exp(logw - mx)
    Z = w.sum()
    post1 = (cand.astype(np.float64) * w[:, None]).sum(axis=0) / Z
    bits = (post1 >= 0.5).astype(np.uint8)
    osd_top1 = cand[int(np.argmin(d2))]  # Euclidean-nearest = OSD top1 proxy
    return bits, osd_top1


# --------------------------------------------------------------------------
def validate(args, G, pat):
    nrows = args.validate_rows
    print(f"Loading {nrows} training rows ...")
    df_x = pd.read_csv(ROOT / "train/train_codeword_x_shard_000.csv", nrows=nrows)
    df_y = pd.read_csv(ROOT / "train/train_noisy_y_shard_000.csv", nrows=nrows)
    cx = df_x[BIT_COLUMNS].to_numpy(np.uint8)
    cy = df_y[Y_COLUMNS].to_numpy(np.float64)
    rng = np.random.default_rng(1234)
    p = rng.permutation(nrows)
    cy, cx = cy[p], cx[p]

    half = nrows // 2
    resid = cy[:half] - (1.0 - 2.0 * cx[:half].astype(np.float64))
    sigma2 = float(resid.var())
    print(f"Estimated sigma^2 = {sigma2:.4f}")

    osd_err = 0
    bm_err = 0
    total = 0
    t0 = time.time()
    for i in range(nrows):
        cand, cost = gen_candidates_fast(cy[i], G, pat)
        bits, osd_top1 = bitmap_from_candidates(cy[i], cand, sigma2)
        osd_err += int(np.sum(osd_top1 != cx[i]))
        bm_err += int(np.sum(bits != cx[i]))
        total += 64
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(
                f"  [{i+1}/{nrows}] {el:.0f}s | OSD top1 {osd_err/total:.6f} | "
                f"bitMAP {bm_err/total:.6f}"
            )
    print(f"=== VALIDATION ({nrows} samples, fast) ===")
    print(f"OSD top1 bit BER : {osd_err/total:.6f}  ({osd_err}/{total})")
    print(f"bit-MAP bit BER  : {bm_err/total:.6f}  ({bm_err}/{total})")
    print(f"(CPU version refs: OSD ~0.0071, bit-MAP ~0.0069)")
    return bm_err / total


def infer(args, G, pat, sigma2):
    with ZipFile(ROOT / "public_test.zip") as z:
        f = z.open("competition_data/test_noisy_y_public.csv")
        test = pd.read_csv(f)
    y = test[Y_COLUMNS].to_numpy(np.float64)
    ids = test["id"].to_numpy()
    print(f"Loaded test: {len(y)} rows, sigma2={sigma2}")

    out = np.empty((len(y), 64), dtype=np.uint8)
    t0 = time.time()
    for i in range(len(y)):
        cand, _ = gen_candidates_fast(y[i], G, pat)
        bits, _ = bitmap_from_candidates(y[i], cand, sigma2)
        out[i] = bits
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(y)}] {time.time()-t0:.0f}s")
    sub = pd.DataFrame(out, columns=BIT_COLUMNS)
    sub.insert(0, "id", ids)
    sub.to_csv(args.output, index=False)
    print(f"Wrote {len(sub)} rows to {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "infer"], default="validate")
    ap.add_argument("--validate-rows", type=int, default=10000)
    ap.add_argument("--list-size", type=int, default=32)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "submission_bitmap_v5.csv")
    args = ap.parse_args()

    H, G = load_matrices()
    pat, P = make_pattern_matrix(args.list_size, args.order)
    print(f"patterns: {P} (list={args.list_size}, order={args.order})")

    if args.mode == "validate":
        validate(args, G, pat)
    else:
        sigma2 = args.sigma2 if args.sigma2 else 0.4047
        infer(args, G, pat, sigma2)


if __name__ == "__main__":
    main()
