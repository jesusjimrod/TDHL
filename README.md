# TDHL

Simulation code for *Rigorous Time-dependent Hamiltonian Learning via Continuous Weak Measurements*
(J. Jiménez-Rodríguez, G. Franceschetto, A. Acín, L. Pereira).

| Path | Contents |
|---|---|
| `chain.ipynb` | Reconstruction of the time-dependent TFIM-like chain with n = 6 qubits, for the minimal graph-coloring ensemble (S = 9) and the overcomplete Pauli ensemble (S = 36). Produces the data and quick-look plots of Figs. 4, 5, 6 and 8. |
| `complexity.ipynb` | Sample-complexity sweep for n = 2, 4, 6, 8 with S = 20 probes (Fig. 7). |
| `paper_figures/` | Scripts that reproduce Figs. 4, 6, 7 and 8 exactly as they appear in the paper, the data behind them, and the figure PDFs. See its README. |

## Running

```sh
pip install -r requirements.txt
jupyter notebook chain.ipynb        # about 2 minutes
jupyter notebook complexity.ipynb   # about 10 minutes, dominated by n = 8
```

All random seeds are fixed. The weak-measurement signals are simulated with a midpoint unitary followed by the
measurement-induced dephasing map at every step, plus Gaussian noise for the finite number of trajectories (App. B of
the paper). The probe states are propagated with the explicit Euler update.
