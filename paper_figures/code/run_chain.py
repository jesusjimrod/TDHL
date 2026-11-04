"""Reconstructions of Figs. 4, 6 and 8 (n = 6, chain.ipynb) with the Euler update.

Settings: Gamma_d = 1e-3, N_s = 1e9 per probe, signals on [0, 3.5) (0.5 buffer for the filter), window T = 3
reconstructed, dt = 0.1 [2 log(2 S n) / (N_s (D+1))]^(1/4) with D = 4 (7.019e-4 for S = 36, 6.578e-4 for S = 9),
Savitzky-Golay derivative over 0.35 time units (cubic), cutoff 5e-3.
Probes: S = 36 random Pauli products (Figs. 4, 6b, 8) and the S = 9 colouring probes (Fig. 6a).
Noise: np.random.seed(42) before the signals for S = 36. For S = 9, seed 17 is the realisation whose largest error on
[0.2, 3] is closest to the median over seeds 0..19 (seed_spread.py).

Usage: python run_chain.py ; writes ../data/chain_S36.npz and ../data/chain_S9.npz (about 20 s on 2 cores).
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdhl_numerics as tn

N, GAMMA, N_SHOTS, T_FINAL, T_TARGET, WINDOW, POLY = 6, 1e-3, 1e9, 3.5, 3.0, 0.35, 3
SEEDS = {36: 42, 9: 17}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def probes(S):
    if S == 36:
        np.random.seed(42)
        return tn.random_probes(N, 36)
    return tn.coloring_probes(N)


def task(S):
    ch = tn.Chain(N)
    rhos = probes(S)
    dt = tn.notebook_dt(S, N, N_SHOTS)
    g = np.sqrt(2 * GAMMA * dt)
    z0 = tn.simulate_records(ch, rhos, T_FINAL, dt, g)
    win = tn.sg_window(WINDOW, dt, z0.shape[-1], POLY)
    np.random.seed(SEEDS[S])
    z = tn.add_noise(z0, g, N_SHOTS)
    K = int(np.round(T_TARGET / dt))
    out = {"dt": dt, "window": win, "seed": SEEDS[S]}
    out["theta_X"], out["theta_YY"] = tn.reconstruct(ch, rhos, tn.sg_derivative(z, win, POLY, dt), dt, K, GAMMA)
    # same signals without noise: deterministic bias of the pipeline alone
    out["theta_X_noiseless"], out["theta_YY_noiseless"] = tn.reconstruct(ch, rhos, tn.sg_derivative(z0, win, POLY, dt),
                                                                         dt, K, GAMMA)
    return S, out


def main():
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    ch = tn.Chain(N)
    with Pool(2) as pool:
        for S, c in pool.imap_unordered(task, [36, 9]):
            dt = c["dt"]
            t = np.arange(c["theta_X"].shape[1]) * dt
            err = tn.global_error(ch, c["theta_X"], c["theta_YY"], t)
            err0 = tn.global_error(ch, c["theta_X_noiseless"], c["theta_YY_noiseless"], t)
            np.savez(os.path.join(DATA, f"chain_S{S}.npz"), t=t, dt=dt, window=c["window"], seed=c["seed"],
                     theta_X=c["theta_X"], theta_YY=c["theta_YY"], err=err,
                     theta_X_noiseless=c["theta_X_noiseless"], theta_YY_noiseless=c["theta_YY_noiseless"],
                     err_noiseless=err0)
            late = t >= 0.2
            print(f"S={S}: dt={dt:.6e}, window={c['window']}, K={len(t)}; max ||dmu|| on [0.2,3]: noisy {err[late].max():.4f}, "
                  f"noiseless {err0[late].max():.4f}; at t=3: {err[-1]:.4f} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
