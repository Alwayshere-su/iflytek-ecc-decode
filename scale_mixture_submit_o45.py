#!/usr/bin/env python3
"""
Production order-4 scale-mixture bit-MAP decoder -> submission CSV.

Improvement over scale_mixture_submit.py: the OSD candidate list is expanded
with ALL size-4 MRB subsets (C(32,4)=35960 extra candidates/sample). This
recovers the rare samples (~0.17%) whose true codeword lies at MRB-subset
size 4 (verified on val10k: fixes sample 5391-type search failures, 7->0 bit
errors; val100k benchmark in probe5_o4_val100k.py).

Usage:
  python scale_mixture_submit_o45.py --mode validate
  python scale_mixture_submit_o45.py --mode infer --output submission_o4.csv
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
ORDER4 = np.array(list(combinations(range(32), 4)), dtype=np.int32)


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


def candidates_info(llr, G):
    """Return (co, base, rows_all) for OSD order-3/4 expansion (see ecc_validate)."""
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
    return co, base, rows_all


def candidates_o3(pat, co, base, rows_all):
    p = pat.copy()
    p[p < 0] = 32
    cand_perm = (base[None, :] ^ rows_all[p[:, 0]]
                 ^ rows_all[p[:, 1]] ^ rows_all[p[:, 2]])
    cand = np.empty((cand_perm.shape[0], 64), dtype=np.uint8)
    cand[:, co] = cand_perm
    return cand


def candidates_o4(co, base, rows_all):
    rows = rows_all[:32]
    cand_perm = (base[None, :] ^ rows[ORDER4[:, 0]] ^ rows[ORDER4[:, 1]]
                 ^ rows[ORDER4[:, 2]] ^ rows[ORDER4[:, 3]])
    cand = np.empty((cand_perm.shape[0], 64), dtype=np.uint8)
    cand[:, co] = cand_perm
    return cand


def decode_one(y, G, pat, g, d2_grid, temp=1.0):
    co, base, rows_all = candidates_info(y, G)
    c3 = candidates_o3(pat, co, base, rows_all)
    c4 = candidates_o4(co, base, rows_all)
    cand = np.vstack([c3, c4])
    mu = 1.0 - 2.0 * cand.astype(np.float32)
    d2 = ((y.astype(np.float32)[None, :] - mu) ** 2).sum(axis=1).astype(np.float64)
    idx = np.clip(np.searchsorted(d2_grid, d2, side="right") - 1, 0, len(g) - 1)
    logw = g[idx] / temp
    mx = logw.max()
    w = np.exp(logw - mx)
    Z = w.sum()
    if Z <= 0:
        return (y < 0).astype(np.uint8)
    post1 = (cand.astype(np.float64) * w[:, None]).sum(axis=0) / Z
    return (post1 >= 0.5).astype(np.uint8)


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


def fit_p_s2(shard_nos=(0, 1)):
    s2 = []
    for s in shard_nos:
        df_x = pd.read_csv(ROOT / f"train/train_codeword_x_shard_{s:03d}.csv",
                           usecols=BIT_COLUMNS)
        df_y = pd.read_csv(ROOT / f"train/train_noisy_y_shard_{s:03d}.csv",
                           usecols=Y_COLUMNS)
        cx = df_x.to_numpy(np.uint8)
        cy = df_y.to_numpy(np.float64)
        resid = cy - (1.0 - 2.0 * cx.astype(np.float64))
        s2.append((resid ** 2).mean(axis=1))
    return np.concatenate(s2)


def _worker(args):
    idx, y, G, pat, g, d2_grid, temp = args
    return np.array([decode_one(y[i], G, pat, g, d2_grid, temp) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "infer"], default="infer")
    ap.add_argument("--validate-rows", type=int, default=2000)
    ap.add_argument("--output", type=Path, default=ROOT / "submission_o4.csv")
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    print("Fitting p(s2) from train shards 0,1 ...")
    s2_all = fit_p_s2()
    print(f"  per-sample s2: mean={s2_all.mean():.4f} std={s2_all.std():.4f} n={len(s2_all)}")
    d2_grid = np.linspace(0.0, 140.0, 1401)
    g, d2_grid = build_g_table(s2_all, d2_grid)
    print("built g(d2)")

    H, G = load_matrices()
    pat, P3 = make_pattern_matrix(32, 3)
    print(f"patterns: o3={P3} o4+={len(ORDER4)}")

    if args.mode == "validate":
        nrows = args.validate_rows
        df_x = pd.read_csv(ROOT / "train/train_codeword_x_shard_000.csv", nrows=nrows)
        df_y = pd.read_csv(ROOT / "train/train_noisy_y_shard_000.csv", nrows=nrows)
        cx = df_x[BIT_COLUMNS].to_numpy(np.uint8)
        cy = df_y[Y_COLUMNS].to_numpy(np.float64)
        rng = np.random.default_rng(1234)
        p = rng.permutation(nrows)
        cy, cx = cy[p], cx[p]
        err = 0
        t0 = time.time()
        for i in range(nrows):
            out = decode_one(cy[i], G, pat, g, d2_grid, args.temp)
            err += int(np.sum(out != cx[i]))
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{nrows}] {time.time()-t0:.0f}s BER {err/((i+1)*64):.6f}")
        print(f"=== VALIDATION o3+o4 (T={args.temp}, {nrows}) BER: {err/(nrows*64):.6f} ===")
        return

    # infer
    test = pd.read_csv(ROOT / "test/competition_data/test_noisy_y_public.csv")
    y = test[Y_COLUMNS].to_numpy(np.float64)
    ids = test["id"].to_numpy()
    print(f"Loaded test: {len(y)} rows")

    import multiprocessing as mp
    idxs = np.array_split(np.arange(len(y)), args.nproc)
    t0 = time.time()
    with mp.Pool(args.nproc) as pool:
        parts = pool.map(_worker, [(idx, y, G, pat, g, d2_grid, args.temp)
                                   for idx in idxs])
    out = np.vstack(parts)
    print(f"decoded in {time.time()-t0:.0f}s")

    sub = pd.DataFrame(out, columns=BIT_COLUMNS)
    sub.insert(0, "id", ids)
    sub.to_csv(args.output, index=False)
    print(f"Wrote {len(sub)} rows to {args.output}")


if __name__ == "__main__":
    main()
