"""Spread of the Fig. 6 errors over noise realisations (Euler update; settings of run_chain.py).

Reruns both ensembles for seeds 0..19 (and 42). Fig. 6 uses for S = 9 the seed whose largest error on [0.2, 3] is
closest to the median over seeds 0..19 (SEEDS in run_chain.py).

Usage: python seed_spread.py ; writes ../data/seed_spread.json (about 1 min on 8 cores).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdhl_numerics as tn
from run_chain import N, GAMMA, N_SHOTS, T_FINAL, T_TARGET, WINDOW, POLY, DATA, probes

SEEDS = list(range(20)) + [42]
_records = {}


def records(S):
    if S not in _records:
        ch, rhos = tn.Chain(N), probes(S)
        dt = tn.notebook_dt(S, N, N_SHOTS)
        _records[S] = (ch, rhos, dt, tn.simulate_records(ch, rhos, T_FINAL, dt, np.sqrt(2 * GAMMA * dt)))
    return _records[S]


def task(args):
    S, seed = args
    ch, rhos, dt, z0 = records(S)
    np.random.seed(seed)
    z = tn.add_noise(z0, np.sqrt(2 * GAMMA * dt), N_SHOTS)
    win = tn.sg_window(WINDOW, dt, z0.shape[-1], POLY)
    K = int(np.round(T_TARGET / dt))
    t = np.arange(K) * dt
    e = tn.global_error(ch, *tn.reconstruct(ch, rhos, tn.sg_derivative(z, win, POLY, dt), dt, K, GAMMA), t)
    late = t >= 0.2
    above = np.nonzero(late & (e > 0.1))[0]
    return S, seed, {"max_err": float(e[late].max()), "err_T": float(e[-1]),
                     "first_above_0.1": float(t[above[0]]) if len(above) else None}


def main():
    t0 = time.time()
    out = {36: {}, 9: {}}
    with Pool(8) as pool:
        for S, seed, r in pool.imap_unordered(task, [(S, s) for S in (9, 36) for s in SEEDS]):
            out[S][seed] = r
    json.dump({str(S): {str(s): out[S][s] for s in SEEDS} for S in out},
              open(os.path.join(DATA, "seed_spread.json"), "w"), indent=1)
    for S in (36, 9):
        m = np.array([out[S][s]["max_err"] for s in range(20)])
        eT = np.array([out[S][s]["err_T"] for s in range(20)])
        f = [out[S][s]["first_above_0.1"] for s in range(20)]
        f = np.array([np.nan if x is None else x for x in f])
        print(f"S={S}: max error on [0.2,3] over seeds 0-19: median {np.median(m):.3f}, range {m.min():.3f}-{m.max():.3f};"
              f" error at t=3: median {np.median(eT):.3f}, range {eT.min():.3f}-{eT.max():.3f};"
              f" first time above 0.1: median {np.nanmedian(f):.2f}, range {np.nanmin(f):.2f}-{np.nanmax(f):.2f}"
              f" | seed 42: max {out[S][42]['max_err']:.3f}, t=3 {out[S][42]['err_T']:.3f},"
              f" first above 0.1 {out[S][42]['first_above_0.1']}")
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
