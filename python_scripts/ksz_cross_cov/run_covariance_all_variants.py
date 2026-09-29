"""
kSZ Covariance Components Calculator (MPI Parallelized)
=======================================================

Computes and saves the *components* of the covariance
    Cov[C_ell^{ij}, C_ell^{i'j'}] = (C_kappa^{ii'}(ell) + N_shape^{ii'}) * C_v^{jj'}
so that the assembled covariances can be built later in any combination by
build_covariances.py.

Components saved:
    C_kappa_matrix        : lensing cross-power C_kappa^{ii'}(ell)  (n_ell,nb,nb)
    N_matrix              : shape-noise matrix N_shape^{ii'}        (nb,nb)
    C_velocity_signal     : velocity signal C_v (plain Pvv), no photo-z   (nb,nb)
    C_velocity_signal_pz  : velocity signal C_v (plain Pvv), photo-z on    (nb,nb)
    N_vrec_nopz           : reconstruction-noise term, photo-z off         (nb,nb)
    N_vrec_pz             : reconstruction-noise term, photo-z on          (nb,nb)

Velocity-factor pieces (C_v^{jj'}) are signal + reconstruction noise:
    <v_hat v_hat> = C_v_signal + N_vrec.
When photo-z smearing is on it is applied to BOTH signal and noise (the *_pz
arrays). The signal is the existing growth-corrected kernel C_v; the noise is
the kernel-overlap shot-noise term (raw velocity kernels):

    N^{V_a V_b} = (a H f / c)^2|_{z_ref} * (1/4pi) * int dK_par/(2pi)
                  Htilde^v_a(K_par) Htilde^v_b(-K_par) S^2(K_par, sigma_chi)
    S^2 = exp(-K_par^2 sigma_chi^2),   sigma_chi = c sigma_z / H(z)

with 1/(nbar_g b^2) folded into the velocity-kernel transforms Htilde^v.
(a H f / c)^2 is evaluated at z_ref (= z_nulled for auto, midpoint for cross)
as (Dv / (D c))^2 since Dv = a D f H.

Survey galaxy model (for nbar_g, b):
    Built in: rubin (b = 0.95/D, n_eff = 27) and roman (piecewise b, n_eff = 4).
    roman_wl reuses roman_kl's n_eff (verify). Only roman_kl's dN/dz path is
    known here; pass --dndz-file for the others.

Usage with MPI:
    mpirun -n 4 python run_covariance_all_variants.py --width 200 --survey roman_kl
    python build_covariances.py <component_file.npz>   # assemble the 4 covariances
"""

import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
import argparse
import itertools

import numpy as np
import scipy.interpolate
import scipy.integrate
from mpi4py import MPI

from ksz_common_with_kappa_nulling_convolve import (
    setup_mpi, get_default_params, setup_cosmology_interpolators,
    compute_growth_factor, compute_velocity_growth_factor,
    load_ksz_kernel, load_kernel_data,
    load_noise_matrix, load_precomputed_ckappa, get_input_output_dirs,
    get_sigma_z_over_1plusz,
    MultiBinPowerSpectrumCalculator,
    c_kms,
)


# -----------------------------------------------------------------------------
# Galaxy sample-property factories (ROMAN-specific -- edit for other surveys)
# -----------------------------------------------------------------------------
# Survey galaxy-model defaults (only values we have actually been given).
#   n_eff [arcmin^-2]: rubin 27, roman_kl 4 (pass --n-eff to override).
#   dN/dz file: rubin and roman_kl are in results/source_dndz/; pass
#               --dndz-file for anything else.
_N_EFF_BY_SURVEY = {'rubin': 27.0, 'roman_kl': 4.0, 'roman_wl': 51.0}
_DNDZ_BY_SURVEY = {'rubin': os.path.join(REPO_ROOT, 'results/source_dndz/LSST_source_nz_Y10.txt'), 'roman_kl':os.path.join(REPO_ROOT, 'results/source_dndz/dndz_roman_kl_ntomo10.txt')}


def make_bias_func(survey, fchi_z, Dz_interp_chi):
    """
    Linear galaxy bias b(z), survey-specific:
      rubin             : b(z) = 0.95 / D(z)
      roman_wl/roman_kl : b(z) = 0.88 z + 0.49 (z < 2), 0.98 z + 0.49 (z >= 2)
    Other surveys raise -- add a model before using them.
    """
    if survey == 'rubin':
        def bias_closure(z):
            z_arr = np.atleast_1d(z).astype(float)
            chi = fchi_z(z_arr)                       # z -> comoving distance
            D_z = np.maximum(Dz_interp_chi(chi), 1e-10)
            b = 0.95 / D_z
            return float(b[0]) if np.isscalar(z) or z_arr.ndim == 0 else b
        return bias_closure

    if survey in ('roman_wl', 'roman_kl'):
        def bias_closure(z):
            z_arr = np.atleast_1d(z).astype(float)
            b = np.where(z_arr < 2.0,
                         0.88 * z_arr + 0.49,
                         0.98 * z_arr + 0.49)
            return float(b[0]) if np.isscalar(z) or z_arr.ndim == 0 else b
        return bias_closure

    raise ValueError(
        f"No galaxy bias model defined for survey '{survey}'. "
        f"Add one in make_bias_func() before running this survey."
    )


def make_nbar_func(fchi_z, astropy_cosmo, file_dndz, n_arcmin2):
    """
    Comoving galaxy number density nbar_g(z) [(1/Mpc)^3].

    Reads normalized n(z) from file, multiplies by n_arcmin2 (converted to
    sr^-1), and divides by the comoving volume element dV/(dz dOmega).
    """
    if not os.path.exists(file_dndz):
        raise FileNotFoundError(f"Could not find n(z) file: {file_dndz}")

    z_data, nz_data = np.loadtxt(file_dndz, unpack=True)
    order = np.argsort(z_data)  # Roman KL file is not sorted in z
    z_data, nz_data = z_data[order], nz_data[order]
    nz_norm = nz_data / np.trapezoid(nz_data, z_data)
    nz_interp = scipy.interpolate.interp1d(
        z_data, nz_norm, bounds_error=False, fill_value=0.0
    )

    # 1 sr = (180*60/pi)^2 arcmin^2
    n_sr = n_arcmin2 * (180.0 * 60.0 / np.pi) ** 2

    def nbar_closure(z):
        z_arr = np.atleast_1d(z).astype(float)
        nbar_out = np.zeros_like(z_arr)
        for i, zi in enumerate(z_arr):
            if zi <= 0.0:
                continue
            dndz_domega = n_sr * nz_interp(zi)        # galaxies / sr / dz
            Hz = astropy_cosmo.H(zi).value            # km/s/Mpc
            chi_z = fchi_z(zi)                         # Mpc
            dV_dzdomega = c_kms * (chi_z ** 2) / Hz    # Mpc^3 / sr / dz
            if dV_dzdomega > 0:
                nbar_out[i] = dndz_domega / dV_dzdomega
        return float(nbar_out[0]) if np.isscalar(z) or z_arr.ndim == 0 else nbar_out

    return nbar_closure


def sigma_chi_of_z(z, astropy_cosmo, sigma_z_over_1plusz):
    """
    Photo-z radial smearing scale sigma_chi(z) [Mpc].
    sigma_chi = c * sigma_z / H(z),  sigma_z = (1+z) * (sigma_z/(1+z)).
    With sigma_z_over_1plusz = 0 this returns 0 (photo-z smearing off).
    """
    z = np.asarray(z, dtype=float)
    sigma_z = sigma_z_over_1plusz * (1.0 + z)
    if z.ndim == 0:
        Hz = float(astropy_cosmo.H(float(z)).value)
    else:
        Hz = np.array([astropy_cosmo.H(zi).value for zi in z])
    return c_kms * sigma_z / Hz


# -----------------------------------------------------------------------------
# Velocity-kernel Fourier transform weighted by sqrt(1 / (nbar_g b^2))
# -----------------------------------------------------------------------------
class WeightedKernelFT:
    """
    Fourier transform of a velocity kernel H^v(chi) multiplied by
    sqrt(1 / (nbar_g(chi) b(chi)^2)).

    This mirrors NulledKernelFT's get_transform(k, chi_center) interface but
    applies NO growth correction -- it transforms the RAW kernel (times the
    sqrt weight). With both bins carrying sqrt(1/(nbar b^2)), the product
    htilde_i(K) htilde_j*(K) is the FT pair whose Parseval inverse is
    int dchi H^v_i(chi) H^v_j(chi) / (nbar_g(chi) b(chi)^2), so the K_par
    integral with sigma_chi = 0 reproduces the real-space overlap exactly.

    Parameters
    ----------
    kernel_interp : callable
        Interpolator for H^v(Delta chi) with Delta chi = chi - chi_nulled.
    chi_nulled : float
        Comoving distance to the kernel peak [Mpc] (phase center).
    weight_interp : callable
        Interpolator for the weight w(chi) = 1 / (nbar_g(chi) b(chi)^2),
        evaluated at ABSOLUTE comoving distance.
    chi_FWHM : float, optional
        Kernel FWHM in chi [Mpc]; sets the FFT window to +/- 20 * FWHM.
    chi_min, chi_max : float
        Manual FFT window if chi_FWHM is None [Mpc].
    num_samples : int
        Number of FFT samples.
    """

    def __init__(self, kernel_interp, chi_nulled, weight_interp,
                 chi_FWHM=None, chi_min=-6000., chi_max=6000.,
                 num_samples=2**16):
        if chi_FWHM is not None:
            padding_factor = 20.0
            self.chi_min = -chi_FWHM * padding_factor
            self.chi_max = chi_FWHM * padding_factor
        else:
            self.chi_min = chi_min
            self.chi_max = chi_max

        self.num_samples = num_samples
        self.chi_nulled = chi_nulled

        chi_arr = np.linspace(self.chi_min, self.chi_max, num_samples, endpoint=False)
        d_chi = chi_arr[1] - chi_arr[0]

        # H^v(Delta chi) * sqrt(w(chi)) with chi = Delta chi + chi_nulled.
        kern = kernel_interp(chi_arr)
        w_abs = np.maximum(weight_interp(chi_arr + chi_nulled), 0.0)
        weighted = kern * np.sqrt(w_abs)

        fft_raw = np.fft.fft(np.fft.ifftshift(weighted)) * d_chi
        k_raw = np.fft.fftfreq(num_samples, d_chi)

        self.k_axis_shifted = 2 * np.pi * np.fft.fftshift(k_raw)
        self.fft_shifted = np.fft.fftshift(fft_raw)

        self.fft_interp_real = scipy.interpolate.interp1d(
            self.k_axis_shifted, self.fft_shifted.real,
            kind='linear', bounds_error=False, fill_value=0.0
        )
        self.fft_interp_imag = scipy.interpolate.interp1d(
            self.k_axis_shifted, self.fft_shifted.imag,
            kind='linear', bounds_error=False, fill_value=0.0
        )

    def get_transform(self, k, chi_center):
        """Get FFT at k with phase shift to chi_center."""
        base = self.fft_interp_real(k) + 1j * self.fft_interp_imag(k)
        phase = np.exp(-1j * k * chi_center)
        return base, phase


# -----------------------------------------------------------------------------
# Calculator with kernel-overlap velocity reconstruction noise
# -----------------------------------------------------------------------------
class ReconstructionNoiseCalculator(MultiBinPowerSpectrumCalculator):
    """
    Adds the velocity reconstruction-NOISE term to the base calculator.

    The signal velocity cross-power C_v is provided unchanged by the parent
    (C_velocity_cross_approx_quad, optionally photo-z smeared). The noise term
    is the kernel-overlap shot-noise contribution:

        N^{V_a V_b} = (a H f / c)^2|_{z_ref} (1/4pi) int dK_par/(2pi)
                      htilde_a(K_par) htilde_b(-K_par) S^2(K_par, sigma_chi),
        S^-2 = exp(K_par^2 sigma_chi^2),

    with 1/(nbar_g b^2) folded into the htilde via WeightedKernelFT. The
    reconstructed velocity variance is C_v_signal + N^{V_a V_b}.

    Empty / sparse bins: nbar_g is floored at nbar_floor, so the weight
    1/(nbar_g b^2) -- and hence the noise -- is large (but finite) where there
    are no galaxies. Lower --nbar-floor to make empty-bin noise larger.
    """

    def __init__(self, *args, bias_func, nbar_func, nbar_floor=1e-6,
                 z_weight_max=8.0, n_weight_samples=3000, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_func = bias_func
        self.nbar_func = nbar_func
        self.nbar_floor = float(nbar_floor)

        # Precompute w(chi) = 1 / (nbar_g(chi) b(chi)^2) on a modest grid once,
        # then interpolate (cheap, vectorized). Avoids per-FFT-sample astropy
        # H(z) calls when building the kernel transforms below.
        z_samp = np.linspace(1e-3, z_weight_max, n_weight_samples)
        nbar_samp = np.maximum(np.asarray(self.nbar_func(z_samp), dtype=float),
                               self.nbar_floor)
        b_samp = np.asarray(self.bias_func(z_samp), dtype=float)
        denom = nbar_samp * np.maximum(b_samp ** 2, 1e-30)
        w_samp = 1.0 / denom
        chi_samp = np.asarray(self.fchi_z(z_samp), dtype=float)
        self.weight_interp = scipy.interpolate.interp1d(
            chi_samp, w_samp, kind='linear', bounds_error=False, fill_value=0.0
        )

        # Velocity-kernel transforms with the sqrt-weight folded in (raw kernel,
        # no growth correction), one per bin.
        self.noise_fts = [
            WeightedKernelFT(
                kd['kernel_interp'], kd['chi_nulled'], self.weight_interp,
                chi_FWHM=kd['chi_FWHM']
            )
            for kd in self.kernel_data_list
        ]

    def _vel_prefactor(self, z_ref):
        """(a H f / c)^2 |_{z_ref} = (Dv / (D c))^2, since Dv = a D f H."""
        chi_ref = float(self.fchi_z(z_ref))
        D = self.Dz_interp_chi(chi_ref)
        Dv = self.Dv_interp_chi(chi_ref)
        return (Dv / (D * c_kms)) ** 2

    def N_vrec(self, bin_i, bin_j, z_ref, sigma_chi):
        """
        Velocity reconstruction-noise N^{V_i V_j}, DIAGONAL ONLY.
        Off-diagonal pairs (bin_i != bin_j) return 0; only the same-bin (auto)
        reconstruction noise is computed.

        sigma_chi = 0 gives the no-photo-z noise (S^2 = 1); sigma_chi > 0 the
          photo-z-smeared noise.
        """
        # Diagonal only: no reconstruction-noise correlation between distinct bins.
        if bin_i != bin_j:
            return 0.0
    
        kd_i = self.kernel_data_list[bin_i]
        kd_j = self.kernel_data_list[bin_j]            # == kd_i here
        chi_FWHM = min(kd_i['chi_FWHM'], kd_j['chi_FWHM'])
        vel_prefactor = self._vel_prefactor(z_ref)
    
        def integrand(Kpar):
            h_i_base, h_i_phase = self.noise_fts[bin_i].get_transform(
                Kpar, kd_i['chi_nulled']
            )
            h_j_base, h_j_phase = self.noise_fts[bin_j].get_transform(
                Kpar, kd_j['chi_nulled']
            )
            h_product = (h_i_base * h_i_phase) * np.conj(h_j_base * h_j_phase)
            S2 = np.exp(Kpar ** 2 * sigma_chi ** 2)   # S^-2 = exp(Kpar^2 sigma_chi^2)
            return np.real(h_product) * S2 / (4.0 * np.pi)
    
        result, _ = scipy.integrate.quad_vec(
            integrand, 1.E-6, 20 * (1. / chi_FWHM),
            limit=100000, epsabs=0., epsrel=1e-4
        )
        return vel_prefactor * 2 * result * (1 / (2 * np.pi))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compute_cv_matrices(calc, sigma_chi_pz, n_bins, zmin, zmax,
                        comm, rank, size, clear_caches=True):
    """
    Compute the four velocity-factor components in one MPI-distributed pass over
    upper-triangular bin pairs:

        C_v_signal_nopz : plain Pvv signal, no photo-z smearing
        C_v_signal_pz   : plain Pvv signal, photo-z smeared
        N_vrec_nopz     : reconstruction noise, sigma_chi = 0
        N_vrec_pz       : reconstruction noise, survey sigma_chi

    sigma_chi_pz(z) returns the photo-z radial-smearing scale [Mpc]. The signal
    methods come from the parent; the noise term from ReconstructionNoiseCalculator.
    Returns four (n_bins, n_bins) arrays on rank 0, (None,)*4 elsewhere.
    """
    all_pairs = list(itertools.combinations_with_replacement(range(n_bins), 2))
    pair_array = (np.array(all_pairs, dtype=int) if all_pairs
                  else np.zeros((0, 2), dtype=int))
    my_pairs = np.array_split(pair_array, size)[rank]

    local = {}
    for row in my_pairs:
        j, jp = int(row[0]), int(row[1])
        z_j = calc.kernel_data_list[j]['z_nulled']
        z_jp = calc.kernel_data_list[jp]['z_nulled']
        z_ref = 0.5 * (z_j + z_jp)            # matches C_v / prefactor convention
        sig = float(sigma_chi_pz(z_ref))      # photo-z smearing scale at z_ref

        # Signal (plain Pvv): without and with photo-z smearing. The z arg to
        # C_velocity_cross_approx_quad is ignored internally (uses its own z_ref).
        cv_sig_nopz = float(calc.C_velocity_cross_approx_quad(j, jp, z_ref, zmin, zmax))
        cv_sig_pz = float(calc.C_velocity_cross_approx_quad(
            j, jp, z_ref, zmin, zmax, sigma_chi=sig))

        chi_j   = calc.kernel_data_list[j]['chi_nulled']
        chi_jp  = calc.kernel_data_list[jp]['chi_nulled']
        chi_mid = 0.5 * (chi_j + chi_jp)
        norm_v = (calc.Dz_interp_chi(chi_j) * calc.Dz_interp_chi(chi_jp)
                  / calc.Dz_interp_chi(chi_mid) ** 2)
        cv_sig_nopz *= norm_v
        cv_sig_pz   *= norm_v
        
        # Reconstruction noise: sigma_chi = 0 (no pz) and survey sigma (pz).
        n_nopz = float(calc.N_vrec(j, jp, z_ref, sigma_chi=0.0))
        n_pz = float(calc.N_vrec(j, jp, z_ref, sigma_chi=sig))

        local[(j, jp)] = (cv_sig_nopz, cv_sig_pz, n_nopz, n_pz)

        print(f"[Rank {rank}] C_v bins ({j:2d},{jp:2d}): "
              f"sig_nopz={cv_sig_nopz: .4e}  sig_pz={cv_sig_pz: .4e}  "
              f"N_nopz={n_nopz: .4e}  N_pz={n_pz: .4e}")

        if clear_caches:
            # Keep memory flat across pairs (CLASS_SZ instances are large).
            # The kernel transforms (noise_fts) are built once and kept.
            calc._class_instance_cache.clear()
            calc._results_cache.clear()
            calc._C_vv_interp_cache.clear()

    gathered = comm.gather(local, root=0)

    if rank != 0:
        return None, None, None, None

    C_v_signal_nopz = np.zeros((n_bins, n_bins))
    C_v_signal_pz = np.zeros((n_bins, n_bins))
    N_vrec_nopz = np.zeros((n_bins, n_bins))
    N_vrec_pz = np.zeros((n_bins, n_bins))
    for d in gathered:
        for (j, jp), (a, b, c, e) in d.items():
            C_v_signal_nopz[j, jp] = C_v_signal_nopz[jp, j] = a
            C_v_signal_pz[j, jp] = C_v_signal_pz[jp, j] = b
            N_vrec_nopz[j, jp] = N_vrec_nopz[jp, j] = c
            N_vrec_pz[j, jp] = N_vrec_pz[jp, j] = e
    return C_v_signal_nopz, C_v_signal_pz, N_vrec_nopz, N_vrec_pz


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    comm, rank, size = setup_mpi()

    parser = argparse.ArgumentParser(
        description="Compute kSZ-lensing covariance for all velocity/noise variants."
    )
    parser.add_argument('--width', type=float, required=True,
                        help='Chi width of lensing kernel [Mpc]')
    parser.add_argument('--survey', type=str, default='rubin',
                        choices=['rubin', 'roman_wl', 'roman_kl'])
    parser.add_argument('--recompute-ckappa', action='store_true')
    parser.add_argument('--n_ell', type=int, default=40)
    parser.add_argument('--ell_min', type=float, default=100)
    parser.add_argument('--ell_max', type=float, default=40000)
    # --- reconstruction-noise (galaxy) model: ROMAN defaults ---
    parser.add_argument('--dndz-file', type=str, default=None,
                        help='Source dN/dz file for nbar(z). Default: survey '
                             'lookup (rubin, roman_kl); pass explicitly for '
                             'other surveys.')
    parser.add_argument('--n-eff', type=float, default=None,
                        help='Effective galaxy density [arcmin^-2]. Default: '
                             'survey value (rubin 27, roman 4).')
    parser.add_argument('--sigma-z-over-1plusz', type=float, default=None,
                        help='Photo-z sigma_z/(1+z) for the photo-z variant. '
                             'Default: survey value (rubin 0.03, roman 0.002).')
    parser.add_argument('--nbar-floor', type=float, default=1e-6,
                        help='Floor on nbar_g [Mpc^-3] in the noise weight '
                             '1/(nbar_g b^2). Smaller -> larger empty-bin noise.')
    parser.add_argument('--output-name', type=str,
                        default='covariance_components_all_variants.npz')
    args = parser.parse_args()

    # Photo-z scatter for the photo-z (pz) variant.
    sigma_z_pz = (args.sigma_z_over_1plusz if args.sigma_z_over_1plusz is not None
                  else get_sigma_z_over_1plusz(args.survey))

    # Resolve galaxy-model inputs from the survey (CLI overrides win). These are
    # deterministic on every rank; abort collectively if a value is unknown so
    # no rank proceeds with None.
    n_eff = args.n_eff if args.n_eff is not None else _N_EFF_BY_SURVEY.get(args.survey)
    dndz_file = (args.dndz_file if args.dndz_file is not None
                 else _DNDZ_BY_SURVEY.get(args.survey))
    if n_eff is None or dndz_file is None:
        if rank == 0:
            if n_eff is None:
                print(f"ERROR: no default n_eff for survey '{args.survey}'. "
                      f"Pass --n-eff.")
            if dndz_file is None:
                print(f"ERROR: no default dN/dz file for survey '{args.survey}'. "
                      f"Pass --dndz-file.")
        comm.Abort(1)

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    input_kernel_dir, output_dir = get_input_output_dirs(args.survey, args.width)

    if rank == 0:
        if not os.path.exists(input_kernel_dir):
            print(f"ERROR: Input directory not found: {input_kernel_dir}")
            comm.Abort(1)
        if not os.path.exists(dndz_file):
            print(f"ERROR: dN/dz file not found: {dndz_file}")
            comm.Abort(1)
        os.makedirs(output_dir, exist_ok=True)
        print("=" * 70)
        print(f"kSZ Covariance -- ALL VARIANTS (MPI size: {size})")
        print("=" * 70)
        print(f"Survey           : {args.survey}, width {args.width} Mpc")
        print(f"dN/dz            : {dndz_file}")
        print(f"n_eff            : {n_eff} arcmin^-2")
        print(f"sigma_z/(1+z) pz : {sigma_z_pz}")
        print(f"Output           : {os.path.join(output_dir, args.output_name)}")

    # -------------------------------------------------------------------------
    # [1/4] Pre-computed quantities (all ranks load; avoids broadcasting)
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n[1/4] Loading pre-computed quantities...")
    N_matrix, C_coeff = load_noise_matrix(input_kernel_dir)
    n_bins = N_matrix.shape[0]

    if args.recompute_ckappa:
        ell_values = np.geomspace(args.ell_min, args.ell_max, num=args.n_ell)
        C_kappa_matrix_precomputed = None
    else:
        try:
            ell_values, C_kappa_matrix_precomputed = load_precomputed_ckappa(input_kernel_dir)
        except FileNotFoundError:
            if rank == 0:
                print("C_kappa not found. Falling back to recompute mode.")
            ell_values = np.geomspace(args.ell_min, args.ell_max, num=args.n_ell)
            C_kappa_matrix_precomputed = None

    # -------------------------------------------------------------------------
    # [2/4] Cosmology + calculators
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n[2/4] Initializing cosmology and CLASS_SZ...")
    cosmo_params, common_params, ksz_params, Pk_matter_params = get_default_params()
    z_grid, chi_grid, fz_chi, fchi_z = setup_cosmology_interpolators()

    Dz_interp_chi = compute_growth_factor(
        z_grid, chi_grid, common_params, cosmo_params, ksz_params, Pk_matter_params
    )
    Dv_interp_chi = compute_velocity_growth_factor(
        z_grid, chi_grid, common_params, cosmo_params, ksz_params, Pk_matter_params
    )
    ksz_kernel_interp, chi_ksz_peak = load_ksz_kernel(fchi_z)
    kernel_data_full = load_kernel_data(input_kernel_dir, fchi_z)

    from astropy.cosmology import Planck18 as astropy_cosmo

    bias_closure = make_bias_func(args.survey, fchi_z, Dz_interp_chi)
    if rank == 0:
        print("[2/4] Loading source dN/dz and normalizing...")
    nbar_closure = make_nbar_func(fchi_z, astropy_cosmo, dndz_file, n_eff)

    def sigma_chi_pz(z):     # photo-z ON
        return sigma_chi_of_z(z, astropy_cosmo, sigma_z_pz)

    # Shared positional args for the parent constructor.
    base_args = (cosmo_params, common_params, ksz_params, Pk_matter_params,
                 kernel_data_full, ksz_kernel_interp, Dz_interp_chi, Dv_interp_chi,
                 fchi_z, fz_chi, sigma_z_pz)

    # One calculator: signal C_v from the parent (with/without photo-z via the
    # sigma_chi arg), reconstruction noise from N_vrec (sigma_chi = 0 or survey).
    calc = ReconstructionNoiseCalculator(
        *base_args, bias_func=bias_closure, nbar_func=nbar_closure,
        nbar_floor=args.nbar_floor,
    )

    if calc.n_bins != n_bins:
        if rank == 0:
            print(f"ERROR: kernel n_bins ({calc.n_bins}) != "
                  f"noise n_bins ({n_bins}).")
        comm.Abort(1)

    # -------------------------------------------------------------------------
    # [3/4] Compute matrices in parallel
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n[3/4] Computing matrices in parallel...")
    zmin, zmax = 1e-6, 5.0

    all_pairs = list(itertools.combinations_with_replacement(range(n_bins), 2))

    # --- C_kappa (only if recomputing); identical logic to the original ---
    local_C_k = {}
    if C_kappa_matrix_precomputed is None:
        my_ck_pairs = np.array_split(np.array(all_pairs, dtype=int), size)[rank]
        for row in my_ck_pairs:
            i, ip = int(row[0]), int(row[1])
            z_i = calc.kernel_data_list[i]['z_nulled']
            z_ip = calc.kernel_data_list[ip]['z_nulled']
            z_eval = 0.5 * (z_i + z_ip)
            chi_eval = fchi_z(z_eval)
            kperp_vals = ell_values / chi_eval
            C_k_arr = np.array([
                calc.C_kappa_cross_approx_quad(kp, i, ip, z_eval, zmin, zmax)
                for kp in kperp_vals
            ])
            local_C_k[(i, ip)] = C_k_arr * (1 / chi_eval) ** 2
            print(f"[Rank {rank}] Computed C_kappa for bins ({i}, {ip})")
    all_C_k_dicts = comm.gather(local_C_k, root=0)

    # --- C_velocity: signal (no pz / pz) + reconstruction noise (no pz / pz) ---
    C_v_signal_nopz, C_v_signal_pz, N_vrec_nopz, N_vrec_pz = compute_cv_matrices(
        calc, sigma_chi_pz, n_bins, zmin, zmax, comm, rank, size,
    )

    # -------------------------------------------------------------------------
    # [4/4] Assemble and save on rank 0
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n[4/4] Assembling C_kappa and saving components...")

        # Assemble C_kappa
        if C_kappa_matrix_precomputed is not None:
            C_kappa_matrix = C_kappa_matrix_precomputed
        else:
            C_kappa_matrix = np.zeros((len(ell_values), n_bins, n_bins))
            for res_dict in all_C_k_dicts:
                for (i, ip), val_arr in res_dict.items():
                    C_kappa_matrix[:, i, ip] = val_arr
                    C_kappa_matrix[:, ip, i] = val_arr

        # Diagnostic: diagonal noise-to-signal ratio (no pz), should be > 0.
        d_sig = np.diag(C_v_signal_nopz)
        nz = d_sig > 0
        if np.any(nz):
            print("        Diagonal N_vrec / C_v_signal (no pz):")
            print(f"          median = "
                  f"{np.median(np.diag(N_vrec_nopz)[nz] / d_sig[nz]):.4e}")
            print("        Diagonal photo-z signal suppression "
                  "C_v_signal_pz / C_v_signal_nopz:")
            print(f"          median = "
                  f"{np.median(np.diag(C_v_signal_pz)[nz] / d_sig[nz]):.4f}")

        outpath = os.path.join(output_dir, args.output_name)
        np.savez(
            outpath,
            l_vals=ell_values,
            # covariance components (assemble with build_covariances.py)
            C_kappa_matrix=C_kappa_matrix,        # (n_ell, n_bins, n_bins)
            N_matrix=N_matrix,                    # shape noise (n_bins, n_bins)
            C_velocity_signal=C_v_signal_nopz,    # signal, no photo-z
            C_velocity_signal_pz=C_v_signal_pz,   # signal, photo-z on
            N_vrec_nopz=N_vrec_nopz,              # reconstruction noise, no photo-z
            N_vrec_pz=N_vrec_pz,                  # reconstruction noise, photo-z on
            # photo-z metadata
            sigma_z_over_1plusz=sigma_z_pz,
            z_bins=np.array([kd['z_nulled'] for kd in kernel_data_full]),
        )
        print(f"\nSaved components to {outpath}")
        print("Component keys: l_vals, C_kappa_matrix, N_matrix, "
              "C_velocity_signal, C_velocity_signal_pz, N_vrec_nopz, N_vrec_pz, "
              "sigma_z_over_1plusz, z_bins")


if __name__ == '__main__':
    main()