# Paper figures

Scripts that reproduce Figs. 4, 6, 7 and 8 of the paper, with the data they produce.

| Figure | PDF | Data | Script |
|---|---|---|---|
| Fig. 4 | `new_reconstruction.pdf` | `data/chain_S36.npz` | `code/run_chain.py` |
| Fig. 6 | `probe_analysis.pdf` | `data/chain_S36.npz`, `data/chain_S9.npz` | `code/run_chain.py` |
| Fig. 7 | `scaling.pdf` | `data/complexity.json` | `code/run_complexity.py` |
| Fig. 8 | `full.pdf` | `data/chain_S36.npz` | `code/run_chain.py` |

`code/tdhl_numerics.py` holds the pipeline of `chain.ipynb` and `complexity.ipynb` in vectorized form. With the same
seeds it gives the same numbers as the notebooks.

## Rebuild

From `code/` (Python 3 with numpy, scipy and matplotlib; LaTeX for the figure labels):

```sh
python run_chain.py        # data/chain_S36.npz, data/chain_S9.npz   (about 20 s)
python run_complexity.py   # data/complexity.json                     (about 20 s on 4 cores)
python seed_spread.py      # data/seed_spread.json                    (about 1.5 min on 8 cores)
python make_figures.py     # the four PDFs in this folder
```

## Notes

- **Noise realisations.** S = 36 uses seed 42. The late-time error of the S = 9 ensemble depends strongly on the noise
  realisation: over seeds 0–19, the maximum of ‖Δμ(t)‖₂ on [0.2, 3] has median 0.81 and range 0.37–2.89
  (`data/seed_spread.json`). Fig. 6 uses seed 17, the realisation whose maximum is closest to that median.
- **Fig. 7 fits.** α in the legend is fitted over all five budgets. Over the statistics-dominated budgets 10⁶–10⁸ it is
  0.45–0.50 (`alphas_1e6_1e8` in `complexity.json`). At N_tot = 10¹⁰ more than 89% of the error is the deterministic
  bias floor (`noiseless_over_mean_1e10`). β at the largest budgets therefore reflects the size dependence of that floor
  (`beta_noiseless` = 0.48).
- **Noiseless runs.** `chain_S*.npz` also stores the reconstruction from the same signals without noise
  (`theta_*_noiseless`, `err_noiseless`), which isolates the deterministic bias of the pipeline.
