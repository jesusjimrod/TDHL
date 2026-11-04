"""Sample-complexity sweep of Fig. 7 (complexity.ipynb) with the Euler update.

Settings: n = 2, 4, 6, 8; S = 20 (9 colouring probes + 11 random Pauli products, random.seed(42)); signals on [0, 1)
with dt = 0.02 (50 points), Gamma_d = 1e-3; Savitzky-Golay derivative (21 points, cubic); cutoff 5e-3;
N_tot = 1e6..1e10 with N_s = int(N_tot / S); 10 realisations; Err(T_sc = 0.3) summed over k = 10..14 (t = 0.20..0.28),
i.e. after the half filter window; alpha from a log-log fit over the five budgets, beta from a log-log fit over n at fixed
N_tot. Noise: np.random.seed(42) for each n and the order of draws of complexity.ipynb, which gives the same numbers.
The reconstruction stops at the last time entering Err (later steps cannot change it).

Usage: python run_complexity.py ; writes ../data/complexity.json (under a minute on 4 cores).
"""
import json
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

N_LIST = [2, 4, 6, 8]
N_TOTAL = [int(1e6), int(1e7), int(1e8), int(1e9), int(1e10)]
R_TRIALS, S_STATES = 10, 20
T_REC, DT, GAMMA, WIN, POLY = 1.0, 0.02, 1e-3, 21, 3
EVAL = np.arange(WIN // 2, int(np.round(0.3 / DT)))          # k = 10..14
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def err_tsc(ch, tX, tY):
    return float(np.sqrt(np.sum(tn.global_error(ch, tX[:, EVAL], tY[:, EVAL], EVAL * DT) ** 2)))


def sweep(n):
    ch = tn.Chain(n)
    rhos = tn.coloring_probes(n)
    np.random.seed(42)
    rhos += tn.random_probes(n, S_STATES - 9)
    g = np.sqrt(2 * GAMMA * DT)
    z0 = tn.simulate_records(ch, rhos, T_REC, DT, g)
    K = EVAL[-1] + 1

    def err(z):
        return err_tsc(ch, *tn.reconstruct(ch, rhos, tn.sg_derivative(z, WIN, POLY, DT), DT, K, GAMMA))

    errors = [[err(tn.add_noise(z0, g, int(Nt / S_STATES))) for _ in range(R_TRIALS)] for Nt in N_TOTAL]
    return n, errors, err(z0)


def main():
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    errors, noiseless = {}, {}
    with Pool(4) as pool:
        for n, e, e0 in pool.imap_unordered(sweep, N_LIST[::-1]):
            errors[n], noiseless[n] = e, e0
            print(f"n={n} done [{time.time() - t0:.0f}s]", flush=True)
    means = {n: np.mean(errors[n], axis=1) for n in N_LIST}
    out = {"N_total": N_TOTAL, "n": N_LIST, "eval_times": (EVAL * DT).tolist(),
           "errors": {n: errors[n] for n in N_LIST},
           "means": {n: means[n].tolist() for n in N_LIST},
           "stds": {n: np.std(errors[n], axis=1).tolist() for n in N_LIST},
           "noiseless": {n: noiseless[n] for n in N_LIST},
           "alphas": {n: float(-np.polyfit(np.log(N_TOTAL), np.log(means[n]), 1)[0]) for n in N_LIST},
           "betas": {Nt: float(np.polyfit(np.log(N_LIST), np.log([means[n][i] for n in N_LIST]), 1)[0])
                     for i, Nt in enumerate(N_TOTAL)},
           # statistics-dominated budgets, and the share of the deterministic floor at the largest budget
           "alphas_1e6_1e8": {n: float(-np.polyfit(np.log(N_TOTAL[:3]), np.log(means[n][:3]), 1)[0]) for n in N_LIST},
           "beta_noiseless": float(np.polyfit(np.log(N_LIST), np.log([noiseless[n] for n in N_LIST]), 1)[0]),
           "noiseless_over_mean_1e10": {n: float(noiseless[n] / means[n][-1]) for n in N_LIST}}
    json.dump(out, open(os.path.join(DATA, "complexity.json"), "w"), indent=1)
    for n in N_LIST:
        print(f"n={n}: " + "  ".join(f"{m:.4f}±{sd:.4f}" for m, sd in zip(out["means"][n], out["stds"][n]))
              + f" | alpha={out['alphas'][n]:.3f} | noiseless={noiseless[n]:.4f}")
    print("beta: " + "  ".join(f"1e{int(np.log10(Nt))}: {b:.3f}" for Nt, b in out["betas"].items()))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
