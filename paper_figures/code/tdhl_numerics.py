"""Vectorized form of the pipeline of chain.ipynb and complexity.ipynb; with the same seeds it gives the same numbers.

Model: H(t) = sum_j Omega_j(t) X_j + sum_j J_j(t) Y_j Y_{j+1} with local Z_j monitoring and dephasing (Gamma/2) D[Z_j].
Signals: on the grid t_k = k dt, each step applies the midpoint unitary exp(-i H(t_k + dt/2) dt) followed by the dephasing
map; Gaussian noise of standard deviation sqrt((g^2+1)/(g^2 N_s)) is drawn probe by probe, in the order of the notebooks.
Reconstruction: Savitzky-Golay derivative, local pseudoinverses with absolute cutoff 5e-3, averaging of couplings
estimated by two sites, explicit Euler update (Hermitised, trace renormalised). It starts at t_0 from the known probes
and pairs the derivative at t_k with the state estimate at t_k.
Sites are 0-based, as in the notebooks; the paper labels them 1-based (h_1 is true_Omega_X(0, t)).
"""
import random

import numpy as np
import scipy.linalg as la
from scipy.signal import savgol_filter

I_mat = np.array([[1, 0], [0, 1]], dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

st_z_plus = np.array([1, 0], dtype=complex)
st_z_minus = np.array([0, 1], dtype=complex)
st_y_plus = np.array([1, 1j]) / np.sqrt(2)
st_y_minus = np.array([1, -1j]) / np.sqrt(2)
st_x_plus = np.array([1, 1]) / np.sqrt(2)
st_x_minus = np.array([1, -1]) / np.sqrt(2)
pauli_states = [st_x_plus, st_x_minus, st_y_plus, st_y_minus, st_z_plus, st_z_minus]   # notebook order

ABS_CUTOFF = 5e-3


def true_Omega_X(j, t):
    return 1.0 + 0.5 * np.cos(2 * np.pi * 0.5 * t + j)


def true_J_YY(j, t):
    return 0.5 + 0.2 * t * (j % 2 == 0)


def mkron(*args):
    res = args[0]
    for op in args[1:]:
        res = np.kron(res, op)
    return res


def get_op(op, j, N):
    ops = [I_mat] * N
    ops[j] = op
    return mkron(*ops)


def state_to_dm(psi_list):
    psi_full = mkron(*psi_list)
    return np.outer(psi_full, psi_full.conj())


class Chain:
    """TFIM-like chain H(t) = sum_j Omega_j X_j + sum_j J_j Y_j Y_{j+1} with local Z_j monitoring."""

    def __init__(self, n):
        self.n, self.d = n, 2 ** n
        self.W_X = [get_op(X_mat, j, n) for j in range(n)]
        self.W_YY = [get_op(Y_mat, j, n) @ get_op(Y_mat, j + 1, n) for j in range(n - 1)]
        Z_ops = [get_op(Z_mat, j, n) for j in range(n)]
        self.Z_diag = np.array([np.real(np.diag(Z)) for Z in Z_ops])                     # (n, d)
        # active neighbourhood of site j: X_j, Y_{j-1}Y_j, Y_jY_{j+1} (notebook get_neighborhood)
        self.nbhd = [[("X", j)] + ([("YY", j - 1)] if j > 0 else []) + ([("YY", j)] if j < n - 1 else [])
                     for j in range(n)]
        terms = [(j, kind, i) for j in range(n) for kind, i in self.nbhd[j]]
        comms = [Z_ops[j] @ self.op(kind, i) - self.op(kind, i) @ Z_ops[j] for j, kind, i in terms]
        # Tr(C rho) = sum_ab C^T_ab rho_ab: one product with the flattened states gives every matrix entry
        self.CT = np.array([C.T.ravel() for C in comms]).T                                # (d^2, terms)
        self.cols = []
        c = 0
        for j in range(n):
            self.cols.append(np.arange(c, c + len(self.nbhd[j])))
            c += len(self.nbhd[j])

    def op(self, kind, i):
        return self.W_X[i] if kind == "X" else self.W_YY[i]

    def H(self, omega, J):
        H_t = sum(omega[j] * self.W_X[j] for j in range(self.n))
        return H_t + sum(J[j] * self.W_YY[j] for j in range(self.n - 1))

    def H_true(self, t):
        return self.H([true_Omega_X(j, t) for j in range(self.n)], [true_J_YY(j, t) for j in range(self.n - 1)])

    def targets(self, t):
        """True coefficients at the times t: Omega (n, len(t)), J (n-1, len(t))."""
        t = np.asarray(t, dtype=float)
        return (np.array([true_Omega_X(j, t) for j in range(self.n)]),
                np.array([true_J_YY(j, t) * np.ones_like(t) for j in range(self.n - 1)]))


def coloring_probes(n):
    """The 9 graph-colouring product probes of the notebooks (create_optimal_probe)."""
    patterns = [(st_x_plus, st_y_plus, st_x_plus), (st_y_minus, st_x_plus, st_z_plus), (st_z_plus, st_x_plus, st_y_minus)]
    rhos = []
    for color in range(3):
        for p in patterns:
            state_list = []
            for i in range(n):
                if i % 3 == color:
                    state_list.append(p[1])
                elif i % 3 == (color - 1) % 3:
                    state_list.append(p[0])
                elif i % 3 == (color + 1) % 3:
                    state_list.append(p[2])
            rhos.append(state_to_dm(state_list))
    return rhos


def random_probes(n, count):
    """Random Pauli-product probes; random.seed(42) as in the notebooks (call after np.random.seed, which it leaves alone)."""
    random.seed(42)
    return [state_to_dm([random.choice(pauli_states) for _ in range(n)]) for _ in range(count)]


def notebook_dt(S, n, N_shots, D=4, eta=0.1):
    """Time step of chain.ipynb: eta [2 log(2 S n) / (N_s (D+1))]^(1/4)."""
    return eta * ((2 * np.log(2 * S * n)) / (N_shots * (D + 1))) ** 0.25


def simulate_records(chain, rhos, T_final, dt, g):
    """Noiseless signals z_j^(s)(t_k), shape (S, n, K) with K = round(T_final/dt) and t_k = k dt.

    Each step applies exp(-i H(t_k + dt/2) dt) and then the dephasing map rho_ab -> exp(-(g^2/2) d_H(a,b)) rho_ab.
    The signal at index k is taken before step k, i.e. at time t_k."""
    K = int(np.round(T_final / dt))
    t_grid = np.arange(K) * dt
    M1 = np.array([[1.0, np.exp(-g ** 2 / 2)], [np.exp(-g ** 2 / 2), 1.0]], dtype=complex)
    M = mkron(*([M1] * chain.n))
    R = np.array(rhos, dtype=complex)
    z = np.zeros((len(rhos), chain.n, K))
    for i, t in enumerate(t_grid):
        z[:, :, i] = np.real(np.einsum("saa->sa", R)) @ chain.Z_diag.T
        U = la.expm(-1j * chain.H_true(t + dt / 2.0) * dt)
        R = M * (U @ R @ U.conj().T)
    return z


def add_noise(z, g, N_s):
    """Notebook noise: for each probe in order, np.random.normal(0, std, size=(K, n)), transposed into the records."""
    std = np.sqrt((g ** 2 + 1) / (g ** 2 * N_s))
    return z + np.stack([np.random.normal(0, std, size=(z.shape[2], z.shape[1])).T for _ in range(z.shape[0])])


def sg_derivative(z, window_length, polyorder, dt):
    return savgol_filter(z, window_length, polyorder, deriv=1, delta=dt, axis=-1)


def sg_window(target_time_window, dt, num_steps, polyorder=3):
    """Odd Savitzky-Golay window covering about target_time_window (chain.ipynb safety checks)."""
    window_length = min(int(target_time_window / dt), num_steps - 1)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > num_steps:
        window_length -= 2
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    return window_length


def euler_mask(chain, Gamma_d):
    """Dissipator of sum_j (Gamma/2) D[Z_j] acting elementwise: -Gamma d_H(a, b)."""
    zz = chain.Z_diag[:, :, None] * chain.Z_diag[:, None, :]
    return (Gamma_d / 2.0) * (zz - 1.0).sum(0).astype(complex)


def reconstruct(chain, rhos, dz, dt, K, Gamma_d, cutoff=ABS_CUTOFF):
    """Coefficient estimates at t_k = k dt, k = 0, ..., K-1, from the derivative signals dz (S, n, >= K).

    The state estimates start from the known probes; step k uses the derivative and the state estimate at t_k."""
    n, S = chain.n, len(rhos)
    theta_X, theta_YY = np.zeros((n, K)), np.zeros((n - 1, K))
    mask = euler_mask(chain, Gamma_d)
    R = np.array(rhos, dtype=complex)
    for k in range(K):
        M_all = np.imag(R.reshape(S, -1) @ chain.CT)          # Re(-i Tr([Z_j, W_m] rho_s)) for every (j, m)
        est = {"X": [[] for _ in range(n)], "YY": [[] for _ in range(n - 1)]}
        for j in range(n):
            M_j = M_all[:, chain.cols[j]]
            U, S_vals, Vh = np.linalg.svd(M_j, full_matrices=False)
            S_inv = np.zeros_like(S_vals)
            valid = S_vals > cutoff
            S_inv[valid] = 1.0 / S_vals[valid]
            theta_j = np.real(Vh.T @ np.diag(S_inv) @ U.T @ dz[:, j, k])
            for m_idx, (kind, i) in enumerate(chain.nbhd[j]):
                est[kind][i].append(theta_j[m_idx])
        theta_X[:, k] = [np.mean(e) for e in est["X"]]
        theta_YY[:, k] = [np.mean(e) for e in est["YY"]]
        if k == K - 1:
            break
        H_rec = chain.H(theta_X[:, k], theta_YY[:, k])
        R = R + (-1j * (H_rec @ R - R @ H_rec) + mask * R) * dt
        R = 0.5 * (R + R.conj().transpose(0, 2, 1))
        R = R / np.einsum("saa->s", R)[:, None, None]
    return theta_X, theta_YY


def global_error(chain, theta_X, theta_YY, t):
    """||Delta mu(t_k)||_2 for every column."""
    Om, J = chain.targets(t)
    return np.sqrt(((theta_X - Om) ** 2).sum(0) + ((theta_YY - J) ** 2).sum(0))
