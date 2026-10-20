#!/usr/bin/env python3
"""
Production scale-mixture bit-MAP decoder -> submission CSV.

Fits p(s2) on the full 128k training rows, builds g(d2) = logsumexp over the
s2 grid, then decodes public_test.zip with logw_c = g(d2_c).

Also validates on 10000 held-out rows (same split as earlier experiments)
before writing the submission, to confirm the improvement holds.

Usage:
  python scale_mixture_submit.py --mode validate
  python scale_mixture_submit.py --mode infer --output submission_sm.csv
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
    pats = [()]
    for size in range(1, order + 1):
        pats.extend(combinations(range(list_size), size))
    P = len(pats)
    pat = np.full((P, 3), -1, dtype=np.int32)
    for j, p in enumerate(pats):
        pat[j, :len(p)] = p
    return pat, P


def gen_candidates_fast(llr, G, pat):
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
    base = (msg @ sys) % 2
    pr2 = np.abs(llr[co[pc_arr]])
    lr = np.argsort(pr2)[:32]
    rows = sys[lr]
    rows_all = np.vstack([rows, _ZERO_ROW])
    p = pat.copy()
    p[p < 0] = 32
    cand_perm = base[None, :] ^ rows_all[p[:, 0]] ^ rows_all[p[:, 1]] ^ rows_all[p[:, 2]]
    cand_orig = np.empty((cand_perm.shape[0], 64), dtype=np.uint8)
    cand_orig[:, co] = cand_perm
    return cand_orig


# --------------------------------------------------------------------------
# scale-mixture likelihood table
# --------------------------------------------------------------------------
def build_g_table(s2_samples, d2_grid, n_bins=200):
    log_s2 = np.log(np.clip(s2_samples, 1e-4, None))
    lo, hi = np.percentile(log_s2, [0.2, 99.8])
    edges = np.linspace(lo, hi, n_bins + 1)
    h, _ = np.histogram(log_s2, bins=edges)
    h = h.astype(np.float64) + 1.0
    lp = np.log(h / h.sum())
    centers = (edges[:-1] + edges[1:]) / 2
    s2_grid = np.exp(centers)
    g = np.zeros_like(d2_grid, dtype=np.float64)
    for k, d2 in enumerate(d2_grid):
        vals = -32.0 * np.log(s2_grid) - d2 / (2.0 * s2_grid) + lp
        m = vals.max()
        g[k] = m + np.log(np.exp(vals - m).sum())
    return g, d2_grid


def apply_g(d2, g, d2_grid):
    idx = np.clip(np.searchsorted(d2_grid, d2, side="right") - 1, 0,
                  len(g) - 1)
    return g[idx]


def bitmap_from_logw(y, cand, logw):
    mx = logw.max()
    w = np.exp(logw - mx)
    Z = w.sum()
    if Z <= 0:
        return (y < 0).astype(np.uint8)
    post1 = (cand.astype(np.float64) * w[:, None]).sum(axis=0) / Z
    return (post1 >= 0.5).astype(np.uint8)


# --------------------------------------------------------------------------
def fit_p_s2():
    """Fit per-sample noise power distribution from the full training set."""
    df_x = pd.read_csv(ROOT / "train/train_codeword_x_shard_000.csv",
                       usecols=BIT_COLUMNS)
    df_y = pd.read_csv(ROOT / "train/train_noisy_y_shard_000.csv",
                       usecols=Y_COLUMNS)
    cx = df_x.to_numpy(np.uint8)
    cy = df_y.to_numpy(np.float64)
    resid = cy - (1.0 - 2.0 * cx.astype(np.float64))
    s2 = (resid ** 2).mean(axis=1)
    return s2


def validate(args, G, pat, g, d2_grid, global_s2):
    nrows = args.validate_rows
    print(f"Loading {nrows} rows ...")
    df_x = pd.read_csv(ROOT / "train/train_codeword_x_shard_000.csv", nrows=nrows)
    df_y = pd.read_csv(ROOT / "train/train_noisy_y_shard_000.csv", nrows=nrows)
    cx = df_x[BIT_COLUMNS].to_numpy(np.uint8)
    cy = df_y[Y_COLUMNS].to_numpy(np.float64)
    rng = np.random.default_rng(1234)
    p = rng.permutation(nrows)
    cy, cx = cy[p], cx[p]

    err_g = 0
    err_sm = 0
    total = 0
    t0 = time.time()
    for i in range(nrows):
        y = cy[i]
        truth = cx[i]
        cand = gen_candidates_fast(y, G, pat)
        mu = 1.0 - 2.0 * cand.astype(np.float32)
        d2 = ((y.astype(np.float32)[None, :] - mu) ** 2).sum(axis=1).astype(np.float64)
        lg = -d2 / (2.0 * global_s2)
        lsm = apply_g(d2, g, d2_grid)
        err_g += int(np.sum(bitmap_from_logw(y, cand, lg) != truth))
        err_sm += int(np.sum(bitmap_from_logw(y, cand, lsm) != truth))
        total += 64
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{nrows}] {time.time()-t0:.0f}s | "
                  f"G {err_g/total:.6f} SM {err_sm/total:.6f}")
    print(f"=== VALIDATION ({nrows}) ===")
    print(f"Gaussian      : {err_g/total:.6f} ({err_g}/{total})")
    print(f"scale-mixture : {err_sm/total:.6f} ({err_sm}/{total})")


def infer(args, G, pat, g, d2_grid, global_s2):
    with ZipFile(ROOT / "public_test.zip") as z:
        f = z.open("competition_data/test_noisy_y_public.csv")
        test = pd.read_csv(f)
    y = test[Y_COLUMNS].to_numpy(np.float64)
    ids = test["id"].to_numpy()
    print(f"Loaded test: {len(y)} rows")

    out = np.empty((len(y), 64), dtype=np.uint8)
    t0 = time.time()
    for i in range(len(y)):
        cand = gen_candidates_fast(y[i], G, pat)
        mu = 1.0 - 2.0 * cand.astype(np.float32)
        d2 = ((y[i].astype(np.float32)[None, :] - mu) ** 2).sum(axis=1).astype(np.float64)
        logw = apply_g(d2, g, d2_grid)
        out[i] = bitmap_from_logw(y[i], cand, logw)
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
    ap.add_argument("--output", type=Path, default=ROOT / "submission_sm.csv")
    args = ap.parse_args()

    print("Fitting p(s2) from full training set ...")
    s2_all = fit_p_s2()
    print(f"  per-sample s2: mean={s2_all.mean():.4f} std={s2_all.std():.4f} "
          f"n={len(s2_all)}")
    global_s2 = float(s2_all.mean())

    d2_grid = np.linspace(0.0, 140.0, 1401)
    g, d2_grid = build_g_table(s2_all, d2_grid)
    print("built g(d2)")

    H, G = load_matrices()
    pat, P = make_pattern_matrix(32, 3)
    print(f"patterns: {P}")

    if args.mode == "validate":
        validate(args, G, pat, g, d2_grid, global_s2)
    else:
        infer(args, G, pat, g, d2_grid, global_s2)


if __name__ == "__main__":
    main()
