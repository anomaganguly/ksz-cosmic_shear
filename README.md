# kSZ × cosmic shear: signal-to-noise forecast

Code for forecasting the signal-to-noise ratio (SNR) of the cross-correlation
between the kinetic Sunyaev–Zel'dovich (kSZ) effect and cosmic shear, using
nulled (redshift-localised) lensing kernels. Forecasts are made for Rubin Y10
and Roman (kinematic lensing, `roman_kl`) sources, combined with ACT, Simons
Observatory (SO) and CMB-HD noise.

This code accompanies Ganguly, Schaan, Krause & Eifler (2026),
[*Direct shear × kSZ correlation: controlling baryons without modeling
galaxies*](https://arxiv.org/abs/2607.29091), arXiv:2607.29091.

## Layout

```
python_scripts/ksz_cross_cov/           pipeline code
sbatch_scripts/ksz_cs_cov_submit/       SLURM submit scripts for the MPI steps
results/                                small inputs (outputs are written here too)
  source_dndz/                          source n(z): LSST Y10, Roman KL (10 tomographic bins)
  kSZ_kernel/                           kSZ weight function
  cl_kSZ_z_0.0005_*.npz                 kSZ auto-power spectra (z_max = 20, 4, 3)
forecast_curves/                        CMB theory C_ell^TT and ACT / SO / S4 / HD noise curves
```

## Requirements

Python 3 with `numpy`, `scipy`, `astropy`, `matplotlib`, `mpi4py` and
[`classy_sz`](https://github.com/CLASS-SZ/class_sz). The MPI steps also need a
working `mpirun`. The SLURM scripts load a conda environment named `myenv` and
set `PATH_TO_CLASS_SZ_DATA`; edit both, and the `#SBATCH` account and partition
lines, for your cluster.

## Pipeline

Every step takes the survey (`rubin` or `roman_kl`) and the nulled-kernel width
in comoving Mpc. The final analysis used these widths:

- Rubin: 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000
- Roman KL: 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000

Paths are resolved relative to the repository root, so scripts can be run from
any directory.

1. **Original and nulled lensing kernels.** The width is set by
   `BASE_CHI_WIDTH` at the top of each script.

   ```
   python get_original_nulled_all_kernels_eq_width_rubin_final_version.py
   python get_original_nulled_all_kernels_eq_width_roman_final_version.py
   ```

   Writes `results/lensing_kernels_full_range_{rubin_y10,roman_kl}/nulled_kernel_width_<W>_Mpc/`.

2. **kSZ × lensing cross-correlations** (MPI). Set `WIDTH` and `SURVEY` in the
   submit script and run it from `sbatch_scripts/ksz_cs_cov_submit/`:

   ```
   sbatch submit_cross.sh
   # runs: mpirun python run_cross_correlations_corrected.py --width W --survey S
   ```

   Writes `all_cross_correlations.npz` to
   `results/cross_and_auto_correlations/<survey>_all_bins/width_<W>/`.

3. **Covariance components** (MPI). These are the lensing power, shape noise,
   velocity signal and reconstruction noise, each with and without photo-z
   smearing.

   ```
   sbatch submit_cov.sh
   # runs: mpirun python run_covariance_all_variants.py --width W --survey S
   ```

   Writes `covariance_components_all_variants.npz` to the same directory.

4. **Re-project C_kappa onto the final ell grid.** Re-runs only the Limber
   projection of the saved kernels (ell = 100–40000) and patches the result
   into the covariance file from step 3. Both files are backed up first.

   ```
   python recompute_ckappa.py --survey rubin --width 300 400 600 ...
   ```

5. **SNR.** Call `compute_snr_for_width(width, survey, experiment, f_sky,
   ell_min, ell_max, n_ell_fine, get_cmb_total, flavor=...)` for each width.
   It is defined in both `snr_condition_number.py`, which returns the SNR only,
   and `snr_simple.py`, which also returns the correlation coefficient r(ell)
   against the kSZ auto-power spectrum passed as `ksz_path`. Use
   `cl_kSZ_z_0.0005_20.0.npz` for the full signal, or `_4.0` (Rubin) and
   `_3.0` (Roman KL) for the late-time signal.
   `get_cmb_total(ell, experiment)` must return the dimensionless primary
   C_ell^TT plus the experiment's noise N_ell. Build it from the files in
   `forecast_curves/`: `cltt_theory.txt` and the `AdvACT` (act), `SOV3` (so)
   and `hd_..._lmax40000_nofg` (hd) noise curves, all in muK^2, so multiply by
   1e-12 / 2.725^2. Interpolate them in log-log space. The
   `results_root` and `ksz_path` defaults point into the repository's
   `results/` directory.

   The SNR is computed for four covariance flavours:
   `no_shape_no_velnoise`, `shape_no_velnoise`, `shape_velnoise_nopz` and
   `shape_velnoise_pz`.

## Shared module

`ksz_common_with_kappa_nulling_convolve.py` holds the cosmology, kernel
loading, velocity and cross-power integrals, and input/output path logic. All
steps import from it. The velocity power and the inner K_parallel integral of
the kSZ x lensing cross-spectrum are computed by FFT convolution rather than
direct quadrature. Photo-z smearing enters through a Gaussian factor in both.

## Citation

If you use this code, please cite:

```bibtex
@article{Ganguly2026,
  author        = {Ganguly, Anoma and Schaan, Emmanuel and Krause, Elisabeth and Eifler, Tim},
  title         = {Direct shear $\times$ kSZ correlation: controlling baryons without modeling galaxies},
  year          = {2026},
  eprint        = {2607.29091},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO},
  doi           = {10.48550/arXiv.2607.29091},
  url           = {https://arxiv.org/abs/2607.29091}
}
```
