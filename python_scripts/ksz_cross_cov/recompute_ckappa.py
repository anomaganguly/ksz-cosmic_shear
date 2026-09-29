"""
Widen C_kappa to a new ell grid and propagate it into the covariance file
=========================================================================

One script, two stages, driven by --survey/--width (paths resolved via
get_input_output_dirs, the same logic the covariance script uses):

  STAGE 1 -- RE-PROJECT
    Re-run ONLY the Limber projection (calculate_aps) of the saved nulled
    kernels onto a wider ell grid [ell_min, ell_max], using CLASS_SZ (hmcode)
    matter power. Writes nulled_aps_cl.npz in the kernel dir. Nulling, SVD, the
    noise covariance, and the kernels themselves are NOT redone -- they are read
    from nulled_kernel_k*.npz.

  STAGE 2 -- PATCH COVARIANCE
    Swap C_kappa_matrix and l_vals in the covariance components file with the
    C_kappa just computed, copying velocity/noise blocks through unchanged.
    C_kappa is the only ell-dependent component; the velocity blocks
    (C_velocity_signal*, N_vrec_*) and N_matrix are (nb,nb) and ell-independent,
    so the regrid does not touch them. This is equivalent to re-running the
    covariance components script for C_kappa, without the velocity MPI integrals.

Both stages back up the file they overwrite. Use --no-cov-patch to do Stage 1
only; the patch is auto-skipped (with a warning) if the cov file is missing.

Notes
-----
  * P(k,z) is CLASS_SZ hmcode nonlinear, built from M.pk(k, z) (k in 1/Mpc,
    P in Mpc^3 -- the convention the pipeline already uses via M.pk_lin). The
    regenerated C_kappa will NOT overlay the old CAMB-takahashi file; cross-check
    instead against the --recompute-ckappa quad_vec C_kappa where Limber is valid.
  * k reach is set by Pk_matter_params['P_k_max_h/Mpc'] (~68 /Mpc), enough for
    ell=40000 unless the lowest-chi kernels demand more (a WARNING prints if so).

Usage
-----
    python recompute_ckappa.py --survey roman_kl --width 50
    python recompute_ckappa.py --survey roman_kl --width 50 100 200 \
        --ell-min 100 --ell-max 40000 --n-ell 60
"""

import os
import glob
import shutil
import argparse

import numpy as np
import scipy.interpolate
import scipy.integrate
from astropy.cosmology import Planck18
from astropy import constants as const

from ksz_common_with_kappa_nulling_convolve import get_default_params, get_input_output_dirs


# ---- cosmology constants (mirror the kernel-construction script) ------------
cosmo = Planck18
h = cosmo.h
c_kms = const.c.to_value('km/s')
ZMIN_GLOBAL = 1.E-5
ZMAX_GLOBAL = 3.0

z_grid_cosmo = np.concatenate(
    [[0.0], np.logspace(np.log10(ZMIN_GLOBAL), np.log10(ZMAX_GLOBAL + 1.0), 5000)]
)
chi_grid_cosmo = cosmo.comoving_distance(z_grid_cosmo).value
fz_chi = scipy.interpolate.interp1d(chi_grid_cosmo, z_grid_cosmo,
                                    bounds_error=False, fill_value=np.nan)
fchi_z = scipy.interpolate.interp1d(z_grid_cosmo, chi_grid_cosmo,
                                    bounds_error=False, fill_value=0.0)


# ---- CLASS_SZ P(k,z), wrapped to mimic CAMB's PKNL.P(z, k, grid=False) ------
class _PkInterp:
    """2D log-log interpolator over (k [1/Mpc], z) of nonlinear P [Mpc^3].
    P(z, k, grid=False) evaluates pointwise -- matches calculate_aps's usage."""
    def __init__(self, logk, zarr, logpk):       # logpk shape (nk, nz)
        self._spl = scipy.interpolate.RectBivariateSpline(logk, zarr, logpk)
        self.kmax_1mpc = float(np.exp(logk[-1]))
        self.kmin_1mpc = float(np.exp(logk[0]))

    def P(self, z, k, grid=False):
        z = np.atleast_1d(np.asarray(z, float))
        k = np.atleast_1d(np.asarray(k, float))
        return np.exp(self._spl.ev(np.log(k), z))


def build_pk_classy(z_max=5.0, nk=250, nz=60):
    """Nonlinear matter P(k,z) from CLASS_SZ (hmcode), same params as the rest
    of the pipeline. Returns a _PkInterp with .P(z, k, grid=False)."""
    from classy_sz import Class as Class_sz

    cosmo_params, common_params, ksz_params, Pk_matter_params = get_default_params()
    M = Class_sz()
    M.set(common_params)
    M.set(cosmo_params)
    M.set(ksz_params)
    M.set(Pk_matter_params)
    M.set({'output': 'mPk', 'z_max_pk': z_max})
    M.compute()

    hh = cosmo_params['H0'] / 100.0
    kmax_1mpc = 0.98 * Pk_matter_params['P_k_max_h/Mpc'] * hh   # stay inside the grid
    kmin_1mpc = 1.0e-4

    kgrid = np.logspace(np.log10(kmin_1mpc), np.log10(kmax_1mpc), nk)
    zgrid = np.linspace(0.0, z_max, nz)

    logpk = np.empty((nk, nz))
    for i, kk in enumerate(kgrid):
        for j, zz in enumerate(zgrid):
            logpk[i, j] = np.log(max(M.pk(float(kk), float(zz)), 1e-30))

    return _PkInterp(np.log(kgrid), zgrid, logpk)


# ---- Limber projection (identical to the kernel script, PKNL/kmax injected) -
def calculate_aps(kernel_matrix, z_grid, ell_values, PKNL, kmax_pk,
                  integration_points=2000):
    n_ell, n_tomo = len(ell_values), kernel_matrix.shape[1]

    interps = [scipy.interpolate.interp1d(z_grid, kernel_matrix[:, i],
               bounds_error=False, fill_value=0.0) for i in range(n_tomo)]

    log_z = np.linspace(np.log(max(ZMIN_GLOBAL, 1e-5)),
                        np.log(ZMAX_GLOBAL), integration_points)
    zg = np.exp(log_z)
    chi = fchi_z(zg)

    prefactor = np.where(chi > 1e-6, zg * (c_kms / cosmo.H(zg)).value / chi**2, 0.0)
    valid = chi > 1e-6

    W = np.array([f(zg) for f in interps]).T

    active = np.any(np.abs(W) > 1e-30, axis=1)
    if np.any(active):
        ai = np.where(active)[0]
        s = slice(max(0, ai[0] - 2), min(len(zg), ai[-1] + 3))
        zg, log_z, chi, prefactor, valid, W = (
            zg[s], log_z[s], chi[s], prefactor[s], valid[s], W[s])

    base = np.einsum('zi,zj->zij', W, W) * prefactor[:, None, None]

    aps = np.zeros((n_ell, n_tomo, n_tomo))
    for i, ell in enumerate(ell_values):
        k = (ell + 0.5) / chi
        pk = np.zeros_like(zg)
        m = (k <= kmax_pk) & valid
        if np.any(m):
            pk[m] = PKNL.P(zg[m], k[m], grid=False)
        aps[i] = scipy.integrate.simpson(base * pk[:, None, None], log_z, axis=0)

    return aps


def load_nulled_kernels(kernel_dir):
    """Stack saved nulled kernels into (n_z, n_nulled); assumes a shared z grid."""
    files = sorted(glob.glob(os.path.join(kernel_dir, "nulled_kernel_k*.npz")))
    if not files:
        raise FileNotFoundError(f"No nulled_kernel_k*.npz in {kernel_dir}")
    z = np.load(files[0])['z']
    cols = []
    for f in files:
        d = np.load(f)
        if not np.array_equal(d['z'], z):
            raise ValueError(f"z grid mismatch in {f}")
        cols.append(d['nulled_wf'])
    return z, np.array(cols).T


def required_kmax(z, W_mat, ell_max, active_thresh_frac=1e-3):
    """k_max = (ell_max + 0.5) / chi_min over the active kernel region."""
    chi = fchi_z(z)
    peak = np.max(np.abs(W_mat), axis=0, keepdims=True)
    active = np.any(np.abs(W_mat) > active_thresh_frac * np.maximum(peak, 1e-300), axis=1)
    active &= (chi > 1e-6)
    chi_min = chi[active].min() if np.any(active) else chi[chi > 1e-6].min()
    return (ell_max + 0.5) / chi_min, chi_min


# ---- Stage 2: patch C_kappa into the covariance components file -------------
def patch_cov_file(cov_path, ell_new, ck_new, make_backup=True):
    """Swap C_kappa_matrix and l_vals in cov_path; copy all else through."""
    cov = np.load(cov_path)
    data = {k: cov[k] for k in cov.files}

    if 'C_kappa_matrix' not in data or 'l_vals' not in data:
        raise KeyError(f"{cov_path} missing C_kappa_matrix or l_vals.")

    nb_old = data['C_kappa_matrix'].shape[1]
    if ck_new.shape[1] != nb_old:
        raise ValueError(
            f"n_bins mismatch: cov file {nb_old} vs new C_kappa {ck_new.shape[1]}.")

    ell_old = data['l_vals']
    print(f"  patch cov : old ell {ell_old.min():.0f}..{ell_old.max():.0f} "
          f"({len(ell_old)} pts) -> new {ell_new.min():.0f}..{ell_new.max():.0f} "
          f"({len(ell_new)} pts); velocity/noise blocks unchanged.")

    data['C_kappa_matrix'] = ck_new
    data['l_vals'] = ell_new

    if make_backup:
        bak = cov_path + '.bak_oldell'
        if not os.path.exists(bak):
            shutil.copy2(cov_path, bak)
            print(f"  backed up cov -> {bak}")

    np.savez(cov_path, **data)
    print(f"  wrote {cov_path}")


def main():
    p = argparse.ArgumentParser(
        description="Widen C_kappa to a new ell grid (CLASS_SZ) and patch the covariance file.")
    p.add_argument('--survey', required=True,
                   choices=['rubin', 'roman_wl', 'roman_kl', 'toy'])
    p.add_argument('--width', type=float, nargs='+', required=True,
                   help='One or more kernel widths [Mpc].')
    p.add_argument('--ell-min', type=float, default=100.0)
    p.add_argument('--ell-max', type=float, default=40000.0)
    p.add_argument('--n-ell', type=int, default=60)
    p.add_argument('--kmax', type=float, default=None,
                   help='Optional lower ceiling [1/Mpc] on the Limber k-mask '
                        '(cannot exceed the CLASS_SZ grid kmax).')
    p.add_argument('--z-max-pk', type=float, default=5.0)
    p.add_argument('--cov-name', default='covariance_components_all_variants.npz')
    p.add_argument('--no-cov-patch', action='store_true',
                   help='Stage 1 only: re-project, do not patch the cov file.')
    p.add_argument('--no-backup', action='store_true')
    args = p.parse_args()

    ell_values = np.logspace(np.log10(args.ell_min), np.log10(args.ell_max), args.n_ell)

    # Resolve paths + load kernels for every width up front; report required k.
    needed_k_global = 0.0
    jobs = []   # (width, kernel_dir, cov_path, z, W)
    for w in args.width:
        kernel_dir, output_dir = get_input_output_dirs(args.survey, w)
        z, W = load_nulled_kernels(kernel_dir)
        cov_path = os.path.join(output_dir, args.cov_name)
        k_need, chi_min = required_kmax(z, W, args.ell_max)
        needed_k_global = max(needed_k_global, k_need)
        jobs.append((w, kernel_dir, cov_path, z, W))
        print(f"width {w:g}: chi_min(active) = {chi_min:.0f} Mpc, "
              f"required k_max(ell={args.ell_max:g}) = {k_need:.1f} /Mpc")

    print("\nBuilding CLASS_SZ P(k,z) (one compute for all widths)...")
    PKNL = build_pk_classy(z_max=args.z_max_pk)
    kmax_pk = PKNL.kmax_1mpc if args.kmax is None else min(PKNL.kmax_1mpc, args.kmax)
    print(f"CLASS_SZ P(k,z) ready; grid kmax {PKNL.kmax_1mpc:.1f} /Mpc, "
          f"mask ceiling {kmax_pk:.1f} /Mpc.")
    if kmax_pk < needed_k_global:
        print(f"  WARNING: mask ceiling {kmax_pk:.1f} < required {needed_k_global:.1f} "
              f"-> high-ell C_kappa truncated. Raise P_k_max_h/Mpc in "
              f"get_default_params if you need more.")
    print("  NOTE: hmcode P(k) is physically uncertain at k >~ 10 /Mpc; "
          "high-ell C_kappa inherits that.\n")

    for w, kernel_dir, cov_path, z, W in jobs:
        print(f"=== width {w:g} ===")

        # STAGE 1: re-project and write nulled_aps_cl.npz
        cl = calculate_aps(W, z, ell_values, PKNL, kmax_pk)
        aps_path = os.path.join(kernel_dir, 'nulled_aps_cl.npz')
        if os.path.exists(aps_path) and not args.no_backup:
            bak = aps_path + '.bak_ell1000_10000'
            if not os.path.exists(bak):
                shutil.copy2(aps_path, bak)
                print(f"  backed up aps -> {bak}")
        np.savez(aps_path, ell=ell_values, cl_ij=cl)
        print(f"  wrote {aps_path}  (ell {ell_values[0]:.0f}..{ell_values[-1]:.0f}, "
              f"{args.n_ell} pts, shape {cl.shape})")

        # STAGE 2: patch the covariance components file with the same C_kappa
        if args.no_cov_patch:
            print("  (cov patch skipped: --no-cov-patch)")
        elif not os.path.exists(cov_path):
            print(f"  WARNING: cov file not found, skipping patch: {cov_path}")
        else:
            patch_cov_file(cov_path, ell_values, cl,
                           make_backup=not args.no_backup)
        print()

    print("Done.")


if __name__ == '__main__':
    main()