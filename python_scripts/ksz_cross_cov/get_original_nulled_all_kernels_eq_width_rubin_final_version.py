"""
Constructs tomographic bins and nulled lensing kernels with fixed comoving width.

The goal is to create lensing kernels that are localized in comoving distance,
enabling cross-correlation with the kSZ signal at specific redshifts.

Algorithm:
  1. Divide the comoving distance range into base tomographic bins
     (first bin = full width, rest = half width so that nulled combinations
      of 3 consecutive half-width bins produce ~full-width peaks).
  2. For each bin, compute n_i(z_true) accounting for photo-z scatter.
  3. Compute standard lensing kernels W_i(z) for each bin.
  4. Null: take linear combinations of 3 consecutive kernels to cancel
     low-z sensitivity, leaving a narrow peak at the target redshift.
  5. Compute noise covariance, angular power spectra, and save everything.
"""
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.linalg
from astropy.cosmology import Planck18
from astropy import constants as const
from astropy import units as u
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from scipy.stats import norm
import matplotlib.pyplot as plt
from ksz_common_with_kappa_nulling_convolve import get_default_params
from classy_sz import Class as Class_sz

################# INPUT PARAMETERS ##############################

PHOTO_Z_BIAS = 0.0           # Photo-z bias
PHOTO_Z_SIGMA = 0.03         # Photo-z scatter: sigma_z = PHOTO_Z_SIGMA * (1+z)
SIGMA_EPSILON = 0.37         # Intrinsic shape noise (rms ellipticity per component)  # when computing SNR finally divide by 1/sqrt{2}
N_GAL_SURVEY_DENSITY = 27.0  #27.0  # gal/arcmin^2, LSST Y10
BASE_CHI_WIDTH = 300.0      # Target nulled kernel width in comoving Mpc
FILE_DNDZ = os.path.join(REPO_ROOT, 'results/source_dndz/LSST_source_nz_Y10.txt')  # True n(z) file
ZMIN_GLOBAL = 1.E-5          # Minimum redshift for calculations
ZMAX_GLOBAL = 4.0            # Maximum redshift for calculations

################# COSMOLOGY SETUP ##############################

cosmo = Planck18
h = cosmo.h
c_kms = const.c.to_value('km/s')

# Lensing prefactor: (3/2) * H0^2 * Omega_m / c^2, converted to 1/Mpc^2
LENSING_PREFACTOR = (1.5 * cosmo.H0.to_value('1/s')**2 * cosmo.Om0 / const.c.value**2
                     * (1 * u.Mpc).to_value('m')**2)

# Interpolators: z <-> comoving distance chi(z)
z_grid_cosmo = np.concatenate([[0.0], np.logspace(np.log10(ZMIN_GLOBAL), np.log10(ZMAX_GLOBAL + 1.0), 5000)])
chi_grid_cosmo = cosmo.comoving_distance(z_grid_cosmo).value  # Mpc
fz_chi = scipy.interpolate.interp1d(chi_grid_cosmo, z_grid_cosmo, bounds_error=False, fill_value=np.nan)
fchi_z = scipy.interpolate.interp1d(z_grid_cosmo, chi_grid_cosmo, bounds_error=False, fill_value=0.0)

################# CLASS_SZ P(k,z) SETUP ##############################
print("Initializing CLASS_SZ...")

class _PkInterp:
    """2D log-log interp over (k[1/Mpc], z) of nonlinear P[Mpc^3].
    Mimics CAMB's PKNL.P(z, k, grid=False) so calculate_aps is unchanged."""
    def __init__(self, logk, zarr, logpk):          # logpk shape (nk, nz)
        self._spl = scipy.interpolate.RectBivariateSpline(logk, zarr, logpk)
        self.kmax_1mpc = float(np.exp(logk[-1]))
    def P(self, z, k, grid=False):
        z = np.atleast_1d(np.asarray(z, float))
        k = np.atleast_1d(np.asarray(k, float))
        return np.exp(self._spl.ev(np.log(k), z))

def _build_pknl_classy(z_max=5.0, nk=250, nz=60):
    """Nonlinear matter P(k,z) from CLASS_SZ (hmcode), using the pipeline's
    shared get_default_params -- so C_kappa is on the same cosmology/power as
    pk_em, C_v, and the --recompute-ckappa quad_vec C_kappa.
    k in 1/Mpc, P in Mpc^3 (the convention already used via M.pk_lin)."""
    cp, common, ksz, pkm = get_default_params()
    M = Class_sz()
    M.set(common); M.set(cp); M.set(ksz); M.set(pkm)
    M.set({'output': 'mPk', 'z_max_pk': z_max})
    M.compute()
    hh = cp['H0'] / 100.0
    kmax_1mpc = 0.98 * pkm['P_k_max_h/Mpc'] * hh     # stay inside the grid (~67 /Mpc)
    kgrid = np.logspace(np.log10(1e-4), np.log10(kmax_1mpc), nk)
    zgrid = np.linspace(0.0, z_max, nz)
    logpk = np.empty((nk, nz))
    for i, kk in enumerate(kgrid):
        for j, zz in enumerate(zgrid):
            logpk[i, j] = np.log(max(M.pk(float(kk), float(zz)), 1e-30))
    PKNL_obj = _PkInterp(np.log(kgrid), zgrid, logpk)
    M.struct_cleanup()
    M.empty()
    return PKNL_obj

PKNL = _build_pknl_classy(z_max=5.0)
KMAX_PK = PKNL.kmax_1mpc      # now in 1/Mpc; the calculate_aps mask uses this
print(f"P(k,z) ready (CLASS_SZ hmcode); k grid reaches {KMAX_PK:.1f} /Mpc.")

################# LOAD SOURCE n(z) ##############################

# Load true redshift distribution from file, sort and deduplicate
dndz_data = np.loadtxt(FILE_DNDZ).T
order = np.argsort(dndz_data[0])
z_sorted, nz_sorted = dndz_data[0][order], dndz_data[1][order] 
_, ui = np.unique(z_sorted, return_index=True)
dndz_true_interp = scipy.interpolate.interp1d(z_sorted[ui], nz_sorted[ui], bounds_error=False, fill_value=0.0)

# Find the effective redshift range where n(z) is non-negligible
z_check = np.linspace(ZMIN_GLOBAL, ZMAX_GLOBAL, 10000)
nonzero = np.where(dndz_true_interp(z_check) > 1e-3)[0]
Z_BINNING_MIN, Z_BINNING_MAX = z_check[nonzero[0]], z_check[nonzero[-1]]
print(f"Binning range: z=[{Z_BINNING_MIN:.3f}, {Z_BINNING_MAX:.3f}]")

################# CORE FUNCTIONS ##############################

def calculate_bin_properties(bin_z_min, bin_z_max):
    """
    Compute the true-z distribution n_i(z_true) of galaxies selected into
    a photometric bin [bin_z_min, bin_z_max].

    A galaxy at z_true has photo-z scatter sigma = PHOTO_Z_SIGMA*(1+z_true).
    The probability it lands in the bin is the Gaussian CDF difference.
    n_i(z_true) = n(z_true) * P(bin_z_min < z_ph < bin_z_max | z_true).

    Returns: (normed_interp, unnormed_interp, galaxy_fraction)
    """
    zt = np.linspace(ZMIN_GLOBAL, ZMAX_GLOBAL, 2000)
    sigma = PHOTO_Z_SIGMA * (1.0 + zt)

    # Selection probability: fraction of photo-z Gaussian inside the bin
    P = norm.cdf(bin_z_max, loc=zt + PHOTO_Z_BIAS, scale=sigma) - \
        norm.cdf(bin_z_min, loc=zt + PHOTO_Z_BIAS, scale=sigma)

    # Unnormalized true-z distribution for this bin
    ni = dndz_true_interp(zt) * P
    norm_val = scipy.integrate.simpson(ni, zt)

    unnormed = scipy.interpolate.interp1d(zt, ni, bounds_error=False, fill_value=0.0)
    if norm_val < 1e-10:
        normed = scipy.interpolate.interp1d([0, ZMAX_GLOBAL], [0, 0], bounds_error=False, fill_value=0.0)
        return normed, unnormed, 0.0

    normed = scipy.interpolate.interp1d(zt, ni / norm_val, bounds_error=False, fill_value=0.0)
    return normed, unnormed, norm_val


def weight_function(z, source_interps, integration_points=4000, chunk_size=200):
    """
    Compute lensing kernel W_i(z_lens) for each source bin i.

    W_i(z) = PREFACTOR * (1+z) * chi(z) * int_{z}^{inf} n_i(z_s) * [chi(z_s)-chi(z)]/chi(z_s) * d(ln z_s)

    Vectorized over lens redshifts in chunks to control memory.
    Returns: array of shape (len(z), n_bins)
    """
    z = np.atleast_1d(z)
    weights = np.zeros((len(z), len(source_interps)))
    chi_z = fchi_z(z)

    # Only compute for valid lens redshifts (below max source z, positive chi)
    valid = (z < Z_BINNING_MAX) & (chi_z > 1e-6) & np.isfinite(chi_z)
    z_v, chi_v = z[valid], chi_z[valid]

    # Source integration starts above the lens redshift
    z_src_min = np.maximum(z_v + 1e-5, Z_BINNING_MIN)
    ok = z_src_min < Z_BINNING_MAX
    z_v, chi_v, z_src_min = z_v[ok], chi_v[ok], z_src_min[ok]
    idx = np.where(valid)[0][ok]

    if len(z_v) == 0:
        return weights

    # Template grid [0,1] stretched per lens to [log(z_src_min), log(Z_BINNING_MAX)]
    t = np.linspace(0, 1, integration_points)
    log_zmin = np.log(z_src_min)
    log_zmax = np.log(Z_BINNING_MAX)

    for cs in range(0, len(z_v), chunk_size):
        sl = slice(cs, min(cs + chunk_size, len(z_v)))

        # Source redshift grid for this chunk: shape (chunk, Ns)
        log_zs = log_zmin[sl, None] + t[None, :] * (log_zmax - log_zmin[sl, None])
        zs = np.exp(log_zs)
        chi_s = fchi_z(zs)

        # Evaluate n_i(z_source) for all bins: shape (chunk, Ns, Nbins)
        nzl = np.stack([f(zs) for f in source_interps], axis=-1)

        # Lensing efficiency: (chi_source - chi_lens) / chi_source
        eff = np.zeros_like(chi_s)
        chi2d = np.broadcast_to(chi_v[sl, None], chi_s.shape)
        good = chi_s > 1e-6
        eff[good] = (chi_s[good] - chi2d[good]) / chi_s[good]

        # Integrand = z_s * n_i * eff (the z_s factor is the d(ln z) Jacobian)
        integrand = zs[:, :, None] * nzl * eff[:, :, None]

        # Integrate over source redshift axis
        integral = scipy.integrate.simpson(integrand, log_zs[:, :, None], axis=1)

        # Apply prefactor: LENSING_PREFACTOR * (1+z_lens) * chi(z_lens)
        weights[idx[sl]] = LENSING_PREFACTOR * (1 + z_v[sl, None]) * chi_v[sl, None] * integral

    return weights


def calculate_aps(kernel_matrix, z_grid, ell_values, integration_points=2000):
    """
    Angular power spectrum C_ell^ij via Limber approximation.

    C_ell^ij = int d(ln z) * z * (c/H) * W_i*W_j / chi^2 * P_NL(k=(ell+0.5)/chi, z)

    Returns: array of shape (n_ell, n_tomo, n_tomo)
    """
    n_ell, n_tomo = len(ell_values), kernel_matrix.shape[1]

    # Interpolate kernels onto integration grid
    interps = [scipy.interpolate.interp1d(z_grid, kernel_matrix[:, i],
               bounds_error=False, fill_value=0.0) for i in range(n_tomo)]

    # Log-spaced z grid for integration
    log_z = np.linspace(np.log(max(ZMIN_GLOBAL, 1e-5)), np.log(ZMAX_GLOBAL), integration_points)
    zg = np.exp(log_z)
    chi = fchi_z(zg)

    # Integrand prefactor: z * (c/H(z)) / chi(z)^2
    prefactor = np.where(chi > 1e-6, zg * (c_kms / cosmo.H(zg)).value / chi**2, 0.0)
    valid = chi > 1e-6

    # Evaluate kernels on grid
    W = np.array([f(zg) for f in interps]).T

    # Trim to z-range where at least one kernel is nonzero (saves P(k,z) evaluations)
    active = np.any(np.abs(W) > 1e-30, axis=1)
    if np.any(active):
        ai = np.where(active)[0]
        s = slice(max(0, ai[0]-2), min(len(zg), ai[-1]+3))
        zg, log_z, chi, prefactor, valid, W = zg[s], log_z[s], chi[s], prefactor[s], valid[s], W[s]

    # Pre-compute W_i * W_j * prefactor (independent of ell)
    base = np.einsum('zi,zj->zij', W, W) * prefactor[:, None, None]

    aps = np.zeros((n_ell, n_tomo, n_tomo))
    for i, ell in enumerate(ell_values):
        # Limber: k = (ell + 0.5) / chi
        k = (ell + 0.5) / chi
        pk = np.zeros_like(zg)
        m = (k <= KMAX_PK) & valid
        if np.any(m):
            pk[m] = PKNL.P(zg[m], k[m], grid=False)

        # Integrate over log(z)
        aps[i] = scipy.integrate.simpson(base * pk[:, None, None], log_z, axis=0)

    return aps


def find_coefficients(z_null, source_interps):
    """
    Find nulling coefficients via SVD.

    Evaluates the lensing kernels at z_null points (below the target bins),
    forming matrix A. The null vector of A gives coefficients c such that
    sum(c_i * W_i(z)) ~ 0 for all z < z_bins, cancelling low-z sensitivity.

    Returns: coefficient array of length len(source_interps)
    """
    valid = np.isfinite(fchi_z(z_null)) & (fchi_z(z_null) > 0) & (z_null >= ZMIN_GLOBAL)
    z_clean = z_null[valid]
    fallback = np.ones(len(source_interps))

    if len(z_clean) < len(source_interps):
        return fallback

    # Build matrix A[i,j] = W_j(z_null_i)
    A = weight_function(z_clean, source_interps)
    if not np.any(A) or not np.all(np.isfinite(A)):
        return fallback

    # SVD: null vector is the last row of V^T (smallest singular value)
    _, s, Vt = scipy.linalg.svd(A)
    if s[0] / max(s[-1], 1e-12) > 1e10:
        return fallback  # Ill-conditioned, nulling unreliable

    v = Vt[-1]
    # Normalize so last coefficient = 1 (convention)
    return v / v[-1] if abs(v[-1]) > 1e-10 else v


def compute_fwhm(kernel, z_eval):
    """
    Compute the FWHM of a kernel in comoving distance.

    Finds the peak, then searches left and right for where the kernel
    crosses half its peak value.

    Returns: (zpeak, zmin_halfmax, zmax_halfmax, chi_fwhm)
    """
    zpeak = zmin_hw = zmax_hw = chi_fwhm = np.nan
    if np.max(np.abs(kernel)) == 0:
        return zpeak, zmin_hw, zmax_hw, chi_fwhm

    # Find peak (by absolute value, to handle both positive and negative kernels)
    pi = np.argmax(np.abs(kernel))
    zpeak = z_eval[pi]

    # Half-max preserves sign: ensures we find crossings within the main lobe
    hm = kernel[pi] / 2.0

    # Helper lambda for linear interpolation at index i
    get_z = lambda i: z_eval[i] + (hm - kernel[i]) * (z_eval[i+1] - z_eval[i]) / (kernel[i+1] - kernel[i])

    # Find where the kernel drops below half-max
    left_idx = np.where(kernel[:pi] <= hm)[0]
    right_idx = np.where(kernel[pi:] <= hm)[0]

    # Interpolate left crossing (last index below hm before peak)
    if left_idx.size > 0:
        zmin_hw = get_z(left_idx[-1])

    # Interpolate right crossing (first index below hm after peak)
    if right_idx.size > 0:
        zmax_hw = get_z(pi + right_idx[0] - 1)

    # Calculate Comoving Distance Width
    chi_fwhm = fchi_z(zmax_hw) - fchi_z(zmin_hw)

    return zpeak, zmin_hw, zmax_hw, chi_fwhm


################# MAIN EXECUTION ##############################

if __name__ == "__main__":
    output_dir = f'{REPO_ROOT}/results/lensing_kernels_full_range_rubin_y10/nulled_kernel_width_{int(BASE_CHI_WIDTH)}_Mpc'
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Convert galaxy density from arcmin^-2 to sr^-1
    N_GAL_SR = N_GAL_SURVEY_DENSITY * (60.0 * 180.0 / np.pi)**2

    # =====================================================================
    # BIN EDGES
    # First bin = full BASE_CHI_WIDTH (becomes the k=0 kernel directly).
    # Remaining bins = half width, so that pairs of adjacent bins span
    # one full width when combined in the nulling procedure.
    # =====================================================================
    chi_min = fchi_z(Z_BINNING_MIN)
    chi_max = fchi_z(Z_BINNING_MAX)
    chi_after_first = chi_max - (chi_min + BASE_CHI_WIDTH)

    if chi_after_first <= 0:
        # Entire range fits in one bin
        num_base_bins = 1
        chi_edges = np.array([chi_min, chi_max])
        dchi_rem = 0.0
    else:
        n_rem = int(np.ceil(chi_after_first / (BASE_CHI_WIDTH / 2.0)))
        dchi_rem = chi_after_first / n_rem  # Actual half-width (adjusted to fill range exactly)
        num_base_bins = 1 + n_rem
        chi_edges = np.concatenate([[chi_min], [chi_min + BASE_CHI_WIDTH],
                                     chi_min + BASE_CHI_WIDTH + dchi_rem * np.arange(1, n_rem + 1)])

    # Convert chi edges to redshift edges
    z_edges = np.clip(fz_chi(chi_edges), Z_BINNING_MIN, Z_BINNING_MAX)
    z_edges[0], z_edges[-1] = Z_BINNING_MIN, Z_BINNING_MAX
    bin_defs = list(zip(z_edges[:-1], z_edges[1:]))
    print(f"Bins: {num_base_bins} (first={chi_edges[1]-chi_edges[0]:.0f} Mpc, rest={2*dchi_rem:.0f} Mpc)")

    # =====================================================================
    # COMPUTE n_i(z_true) FOR EACH BIN (with photo-z scatter)
    # =====================================================================
    all_normed, all_unnormed, all_fracs = [], [], []
    for zlo, zhi in bin_defs:
        normed, unnormed, frac = calculate_bin_properties(zlo, zhi)
        all_normed.append(normed)
        all_unnormed.append(unnormed)
        all_fracs.append(frac)

    # Galaxy fractions and shape noise per bin
    # N_ii = sigma_e^2 / n_gal_i (diagonal noise power spectrum)
    all_fracs = np.array(all_fracs)
    bin_fracs = all_fracs / all_fracs.sum()
    n_gal_sr = bin_fracs * N_GAL_SR
    N_base_diag = SIGMA_EPSILON**2 / n_gal_sr

    # =====================================================================
    # COMPUTE ORIGINAL LENSING KERNELS W_i(z)
    # =====================================================================
    z_eval = np.linspace(0.0, ZMAX_GLOBAL, 3000)
    W_orig = weight_function(z_eval, all_normed)

    # =====================================================================
    # NULLING LOOP
    # k=0: single bin (no nulling)
    # k=1: difference of first two bins (partial nulling)
    # k>=2: SVD null-space of 3 consecutive bins (full nulling)
    # =====================================================================
    all_nulled, all_solutions = [], []

    for k in range(num_base_bins):
        solution = np.zeros(3)

        if k == 0:
            # No nulling: just the first bin's kernel
            solution[0] = 1.0
            kernel = W_orig[:, 0]

        elif k == 1:
            # Simple difference: W_1 - W_0 (cancels some low-z response)
            solution[0], solution[1] = -1.0, 1.0
            kernel = W_orig[:, 1] - W_orig[:, 0]

        else:
            # Full nulling: combine bins (k-2, k-1, k)
            i0, i1, i2 = k-2, k-1, k

            # Choose nulling points below the lowest bin in the triplet
            z0 = bin_defs[i0][0]
            z_null_max = z0 + 1e-2 if k == 2 else 0.7 * z0
            z_pts = np.linspace(ZMIN_GLOBAL, max(ZMIN_GLOBAL, z_null_max), 1000)
            if z_pts[-1] <= z_pts[0]:
                z_pts = np.linspace(ZMIN_GLOBAL, z0, 3)

            # Find coefficients that null the kernel at low z
            c = find_coefficients(z_pts, [all_normed[i0], all_normed[i1], all_normed[i2]])
            solution[:] = c

            # Apply nulling: linear combination of 3 kernels
            kernel = W_orig[:, [i0, i1, i2]] @ solution

        all_nulled.append(kernel)
        all_solutions.append(solution)

        # Measure the width of the nulled kernel
        zpeak, zmin_hw, zmax_hw, chi_fwhm = compute_fwhm(kernel, z_eval)

        # Propagate noise: N_nulled_kk = sum(c_j^2 * N_base_jj)
        if k == 0:
            noise_diag = solution[0]**2 * N_base_diag[0]
        elif k == 1:
            noise_diag = solution[0]**2 * N_base_diag[0] + solution[1]**2 * N_base_diag[1]
        else:
            noise_diag = sum(solution[j]**2 * N_base_diag[k-2+j] for j in range(3))

        # Save per-kernel results (file structure matches what the MPI code expects)
        fname = os.path.join(output_dir, f'nulled_kernel_k{k+1:03d}.npz')
        orig = W_orig[:, [0]] if k == 0 else W_orig[:, [0, 1]] if k == 1 else W_orig[:, [k-2, k-1, k]]

        np.savez(fname, z=z_eval, nulled_wf=kernel, coeff=solution,
                 zmin=zmin_hw, zmax=zmax_hw, zpeak=zpeak,
                 chi_FWHM=chi_fwhm, original_kernels=orig,
                 nulled_noise_power_diag=noise_diag)
        print(f"  Kernel {k+1}: z_peak={zpeak:.3f}, FWHM={chi_fwhm:.0f} Mpc, N_kk={noise_diag:.2e}")

    # =====================================================================
    # FULL NOISE COVARIANCE MATRIX
    # N_nulled = C * diag(N_base) * C^T
    # C[k,j] = coefficient of base bin j in nulled kernel k.
    # Cross-terms arise because consecutive nulled kernels share base bins.
    # =====================================================================
    C_mat = np.zeros((num_base_bins, num_base_bins))
    for k in range(num_base_bins):
        if k == 0:
            C_mat[k, 0] = all_solutions[k][0]
        elif k == 1:
            C_mat[k, 0] = all_solutions[k][0]
            C_mat[k, 1] = all_solutions[k][1]
        else:
            for j in range(3):
                C_mat[k, k-2+j] = all_solutions[k][j]

    N_nulled = C_mat @ np.diag(N_base_diag) @ C_mat.T
    np.savez(os.path.join(output_dir, 'nulled_noise_power_covariance_matrix.npz'),
             nulled_noise_power_matrix=N_nulled, C_coeff_matrix=C_mat, N_base_diag_power=N_base_diag)

    # =====================================================================
    # ANGULAR POWER SPECTRA (Limber approximation)
    # Computed for both original and nulled kernels.
    # =====================================================================
    ell_values = np.logspace(np.log10(100), np.log10(40000), 60)

    # Original kernels C_ell
    cl_orig = calculate_aps(W_orig, z_eval, ell_values)
    np.savez(os.path.join(output_dir, 'original_aps_cl.npz'), ell=ell_values, cl_ij=cl_orig)

    # Nulled kernels C_ell
    W_nulled_mat = np.array(all_nulled).T  # Shape: (n_z, n_nulled)
    cl_nulled = calculate_aps(W_nulled_mat, z_eval, ell_values)
    np.savez(os.path.join(output_dir, 'nulled_aps_cl.npz'), ell=ell_values, cl_ij=cl_nulled)

    # =====================================================================
    # SUMMARY PLOT: 3 panels
    # Left: original kernels W_i(z)
    # Center: nulled kernels vs comoving distance
    # Right: per-bin n_i(z) distributions with full n(z) overlay
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_base_bins))

    for i in range(num_base_bins):
        axes[0].plot(z_eval, W_orig[:, i], color=colors[i], alpha=0.8)
    axes[0].set(xlabel='z', ylabel='W(z)', title='Original Kernels')

    for i in range(num_base_bins):
        axes[1].plot(fchi_z(z_eval), all_nulled[i], color=colors[i], alpha=0.8)
    axes[1].axhline(0, color='grey', lw=0.5)
    axes[1].set(xlabel='χ (Mpc)', ylabel='W_nulled(χ)', title='Nulled Kernels')

    axes[2].plot(z_eval, dndz_true_interp(z_eval), 'k--', lw=2, label='Full n(z)')
    for i in range(num_base_bins):
        axes[2].plot(z_eval, all_unnormed[i](z_eval), color=colors[i], alpha=0.8)
    axes[2].set(xlabel='z', ylabel='n(z)', title='Bin distributions')
    axes[2].legend()

    for ax in axes:
        ax.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'summary.png'), dpi=150)
    plt.close()

    # =====================================================================
    # SAVE PLOT DATA
    # All arrays needed to reproduce every panel of summary.png.
    # Keep them in a single file so a downstream plotting script can
    # load once and rebuild the figure without re-running the pipeline.
    # =====================================================================
    chi_eval = fchi_z(z_eval)                                   # Panel 2 x-axis
    nz_full = dndz_true_interp(z_eval)                          # Panel 3 black dashed
    nz_per_bin_unnormed = np.array(                             # Panel 3 colored curves
        [f(z_eval) for f in all_unnormed]
    ).T                                                          # shape (n_z, n_bins)
    nz_per_bin_normed = np.array(                               # also save normed for completeness
        [f(z_eval) for f in all_normed]
    ).T                                                          # shape (n_z, n_bins)
    W_nulled_array = np.array(all_nulled).T                     # Panel 2 y-values, shape (n_z, n_bins)
    coeff_array = np.array(all_solutions)                       # nulling coeffs per kernel, shape (n_bins, 3)

    np.savez(
        os.path.join(output_dir, 'plot_data.npz'),
        # axes
        z_eval=z_eval,
        chi_eval=chi_eval,
        # panel 1: original kernels
        W_original=W_orig,                  # shape (n_z, n_bins)
        # panel 2: nulled kernels
        W_nulled=W_nulled_array,            # shape (n_z, n_bins)
        # panel 3: n(z) distributions
        nz_full=nz_full,                    # shape (n_z,)
        nz_per_bin_unnormed=nz_per_bin_unnormed,
        nz_per_bin_normed=nz_per_bin_normed,
        # bin metadata (useful for labels, colors, sanity checks)
        z_edges=z_edges,
        chi_edges=chi_edges,
        bin_fractions=bin_fracs,
        n_gal_sr_per_bin=n_gal_sr,
        N_base_diag=N_base_diag,
        coeff_array=coeff_array,
        num_base_bins=num_base_bins,
    )
    print(f"  Saved plot_data.npz with {num_base_bins} bins on {len(z_eval)} z-points.")
    print("Done.")