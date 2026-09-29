"""
kSZ - Shared Classes and Utilities
===================================

Common infrastructure for kSZ cross-correlation and covariance calculations.
Contains: NulledKernelFT, KSZKernelFT, MultiBinPowerSpectrumCalculator,
and shared setup/loading functions.

# ell range: 100, 40000
"""

import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.signal
import glob
import argparse
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from classy_sz import Class as Class_sz

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
c_kms = const.c.to('km/s').value  # ~ 299792.458 km/s


def sigma_chi_of_z(z, astropy_cosmo, sigma_z_over_1plusz):
    """
    Photo-z radial smearing scale sigma_chi(z) [Mpc].
    sigma_chi = c * sigma_z / H(z),  sigma_z = (1+z) * (sigma_z/(1+z)).
    """
    z = np.asarray(z, dtype=float)
    sigma_z = sigma_z_over_1plusz * (1.0 + z)
    # astropy H(z).value returns km/s/Mpc; handles scalar or array.
    if z.ndim == 0:
        Hz = float(astropy_cosmo.H(float(z)).value)
    else:
        Hz = np.array([astropy_cosmo.H(zi).value for zi in z])
    return c_kms * sigma_z / Hz


# =============================================================================
# CLASS: Nulled Kernel Fourier Transform
# =============================================================================
class NulledKernelFT:
    """
    Computes and stores the Fourier transform of a nulled lensing kernel.
    
    The nulled kernel W(χ) is transformed to k-space via FFT:
        W̃(k) = ∫ W(χ - χ_nulled) × [D(χ)/D(χ_nulled)]^p × e^{-ikχ_nulled} dχ
    
    Parameters
    ----------
    lensing_kernel_interp : callable
        Interpolator for W(Δχ) where Δχ = χ - χ_nulled
    Dz_interp : callable  
        Interpolator for D(χ), the growth factor as function of comoving distance
    chi_nulled : float
        Comoving distance to the kernel peak [Mpc]
    chi_FWHM : float, optional
        Full width at half maximum of kernel in χ [Mpc]
    chi_min, chi_max : float
        Manual FFT window bounds if chi_FWHM not provided [Mpc]
    num_samples : int
        Number of FFT samples (should be power of 2 for efficiency)

    Notes
    -----
    The growth correction factor D(χ)/D(χ_nulled) is always applied at power 1.
    """
    
    def __init__(self, lensing_kernel_interp, Dz_interp, chi_nulled, 
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
        
        chi_arr = np.linspace(self.chi_min, self.chi_max, self.num_samples, endpoint=False)
        d_chi = chi_arr[1] - chi_arr[0]

        weight_function_values = lensing_kernel_interp(chi_arr)

        D_chi_nulled = Dz_interp(chi_nulled)
        if D_chi_nulled == 0:
            raise ValueError("D(chi_nulled) is zero, cannot normalize kernel.")
        
        chi_abs_arr = chi_arr + chi_nulled
        D_chi_arr = Dz_interp(chi_abs_arr)
        correction_factor = D_chi_arr / D_chi_nulled
        weight_function_values_corrected = weight_function_values * correction_factor

        fft_raw = np.fft.fft(np.fft.ifftshift(weight_function_values_corrected)) * d_chi
        k_raw = np.fft.fftfreq(self.num_samples, d_chi)

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
        real_part = self.fft_interp_real(k)
        imag_part = self.fft_interp_imag(k)
        fft_base = real_part + 1j * imag_part
        phase_shift = np.exp(-1j * k * chi_center)
        return fft_base, phase_shift


# =============================================================================
# CLASS: kSZ Kernel Fourier Transform
# =============================================================================
class KSZKernelFT:
    """
    Computes and stores the Fourier transform of the kSZ kernel, with a
    growth-factor correction applied for a specific (kappa, velocity) bin pair.

    The correction factor applied to the kernel weight, evaluated at absolute
    comoving distance chi = chi_arr + chi_ksz_peak, is:

        correction = [D(chi) / D(chi_nulled_kappa)]
                     * [Dv(chi) / Dv(chi_nulled_velocity)]

    where D is the (density) growth factor and Dv = a * D * f * H is the
    velocity growth factor. The density piece is anchored at the kappa-kernel
    peak and the velocity piece at the velocity-kernel peak, so this object is
    specific to one bin pair and should be rebuilt per pair.

    Parameters
    ----------
    ksz_kernel_interp : callable
        Interpolator for W_kSZ(Δχ) where Δχ = χ - χ_ksz_peak
    Dz_interp : callable
        Interpolator for D(χ), the (density) growth factor vs comoving distance
    Dv_interp : callable
        Interpolator for Dv(χ) = a D f H, the velocity growth factor vs χ
    chi_ksz_peak : float
        Comoving distance to the kSZ kernel peak [Mpc]
    chi_nulled_kappa : float
        Comoving distance to the kappa (density) bin peak [Mpc]
    chi_nulled_velocity : float
        Comoving distance to the velocity bin peak [Mpc]
    chi_min, chi_max : float
        FFT window bounds [Mpc]
    num_samples : int
        Number of FFT samples
    """
    
    def __init__(self, ksz_kernel_interp, Dz_interp, Dv_interp,
                 chi_ksz_peak, chi_nulled_kappa, chi_nulled_velocity,
                 chi_min=-50000., chi_max=50000., num_samples=2**16):
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.num_samples = num_samples
        
        chi_arr = np.linspace(self.chi_min, self.chi_max, self.num_samples, endpoint=False)
        d_chi = chi_arr[1] - chi_arr[0]

        weight_function_values = ksz_kernel_interp(chi_arr)

        # Growth-factor correction (density x velocity) for this bin pair.
        # Evaluated at absolute comoving distance chi_arr + chi_ksz_peak.
        chi_abs_arr = chi_arr + chi_ksz_peak
        D_chi_arr = Dz_interp(chi_abs_arr)
        Dv_chi_arr = Dv_interp(chi_abs_arr)

        D_chi_nulled_kappa = Dz_interp(chi_nulled_kappa)
        Dv_chi_nulled_velocity = Dv_interp(chi_nulled_velocity)
        if D_chi_nulled_kappa == 0 or Dv_chi_nulled_velocity == 0:
            raise ValueError(
                "D(chi_nulled_kappa) or Dv(chi_nulled_velocity) is zero, "
                "cannot normalize kSZ kernel."
            )

        correction_factor = ((D_chi_arr / D_chi_nulled_kappa) *
                             (Dv_chi_arr / Dv_chi_nulled_velocity))
        weight_function_values = weight_function_values * correction_factor

        fft_raw = np.fft.fft(np.fft.ifftshift(weight_function_values)) * d_chi
        k_raw = np.fft.fftfreq(self.num_samples, d_chi)

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

    def get_transform(self, k, chi_peak):
        """Get FFT at k with phase shift to chi_peak."""
        real_part = self.fft_interp_real(k)
        imag_part = self.fft_interp_imag(k)
        fft_base = real_part + 1j * imag_part
        phase_shift = np.exp(-1j * k * chi_peak)
        return fft_base, phase_shift


# =============================================================================
# CLASS: Multi-Bin Power Spectrum Calculator
# =============================================================================
class MultiBinPowerSpectrumCalculator:
    """
    Calculator for kSZ-lensing cross-correlations with multiple redshift bins.
    """
    
    def __init__(self, cosmo_params, common_params, ksz_params, Pk_matter_params,
                 kernel_data_list, ksz_kernel_interp, Dz_interp_chi, Dv_interp_chi,
                 fchi_z, fz_chi, sigma_z_over_1plusz):
        
        self.cosmo_params = cosmo_params
        self.common_params = common_params
        self.ksz_params = ksz_params
        self.Pk_matter_params = Pk_matter_params
        self.astropy_cosmo = cosmo
        self.fchi_z = fchi_z
        self.fz_chi = fz_chi
        self.Dz_interp_chi = Dz_interp_chi
        self.Dv_interp_chi = Dv_interp_chi
        self.ksz_kernel_interp = ksz_kernel_interp
        self.sigma_z_over_1plusz = sigma_z_over_1plusz
        
        self.n_bins = len(kernel_data_list)
        self.kernel_data_list = kernel_data_list
        
        self._class_instance_cache = {}
        self._results_cache = {}
        self._C_vv_interp_cache = {}

        self.kernel_fts_auto = []
        for kd in kernel_data_list:
            ft = NulledKernelFT(
                kd['kernel_interp'],
                Dz_interp=Dz_interp_chi,
                chi_nulled=kd['chi_nulled'],
                chi_FWHM=kd['chi_FWHM']
            )
            self.kernel_fts_auto.append(ft)

        # Cross uses the same FTs (identical construction); alias, don't rebuild.
        self.kernel_fts_cross = self.kernel_fts_auto

    
    # =========================================================================
    # CLASS_SZ Interface Methods
    # =========================================================================
    
    def _get_computed_class_instance(self, z_pk, z_min, z_max):
        rounded_z_pk = round(z_pk, 4)
        cache_key = (rounded_z_pk, z_min, z_max)
        
        if cache_key in self._class_instance_cache:
            return self._class_instance_cache[cache_key]
        
        M = Class_sz()
        M.set(self.common_params)
        M.set(self.cosmo_params)
        M.set(self.ksz_params)
        M.set(self.Pk_matter_params)
        M.set({
            'output': 'mPk, pk, pk_em_at_z_1h, pk_em_at_z_2h, pk_at_z_1h, pk_at_z_2h, D',
            'use_fft_for_profiles_transform': 1,
            'z_pk': z_pk,
            'z_for_pk_hm': z_pk,
            'z_min': z_min,
            'z_max': z_max,
        })
        M.compute()
        self._class_instance_cache[cache_key] = M
        return M

    def get_power_spectra(self, z_pk, z_min, z_max):
        rounded_z_pk = round(z_pk, 4)
        cache_key = (rounded_z_pk, z_min, z_max)
        
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]
        
        M = self._get_computed_class_instance(z_pk, z_min, z_max)
        h = self.cosmo_params['H0'] / 100.0
        Om0 = self.astropy_cosmo.Om0 
        Ob0 = self.astropy_cosmo.Ob0
        
        pks_em = M.pk_em_at_z_hm()
        pks_mm = M.pk_at_z_hm()
        # cl_ksz = M.cl_ksz()
        
        k_h_by_mpc = np.array(pks_mm['k'])
        k_by_mpc = k_h_by_mpc * h
        
        pk_em_1h = np.array(pks_em['1h']) * (Om0/Ob0) / (h**3)
        pk_em_2h = np.array(pks_em['2h']) * (Om0/Ob0) / (h**3)
        pk_mm_1h = np.array(pks_mm['1h']) / (h**3)
        pk_mm_2h = np.array(pks_mm['2h']) / (h**3)
        pk_lin = np.array([M.pk_lin(k, z_pk) for k in k_by_mpc])
        
        log_k = np.log(k_by_mpc)
        interp_opts = {'kind': 'linear', 'bounds_error': False, 'fill_value': "extrapolate"}
        
        def create_interp(pk_array):
            safe_pk = np.maximum(pk_array, 1e-30)
            return interp1d(log_k, np.log(safe_pk), **interp_opts)
            
        power_spectra = {
            'z_pk': z_pk,
            'k_by_mpc': k_by_mpc,
            'pk_em_total_interp': create_interp(pk_em_1h + pk_em_2h),
            'pk_mm_total_interp': create_interp(pk_mm_1h + pk_mm_2h),
            'pk_lin_interp': create_interp(pk_lin),
            # 'ell': cl_ksz['ell'],
            # 'cl_ksz_total': np.array(cl_ksz['1h']) + np.array(cl_ksz['2h']),
        }
        
        self._results_cache[cache_key] = power_spectra
        return power_spectra

    # =========================================================================
    # Velocity Power Spectrum
    # =========================================================================
    
    def Pvv(self, z, zmin, zmax, kpar, kperp):
        M = self._get_computed_class_instance(z_pk=z, z_min=zmin, z_max=zmax)
        
        h = self.cosmo_params['H0'] / 100.0
        pks_mm = M.pk_at_z_hm()
        k_h_by_mpc = np.array(pks_mm['k'])
        k_by_mpc = k_h_by_mpc * h
        
        pk_lin = np.array([M.pk_lin(k, z) for k in k_by_mpc])
        
        k_anchor = k_by_mpc[0]
        pk_anchor = pk_lin[0]

        log_k = np.log(k_by_mpc)
        safe_pk = np.maximum(pk_lin, 1e-30)
        pk_lin_interp_func = interp1d(log_k, np.log(safe_pk), 
                                       bounds_error=False, fill_value="extrapolate")

        kpar = np.atleast_1d(kpar)
        kperp = np.atleast_1d(kperp)
        k = np.sqrt(kpar**2 + kperp**2)
        
        Hz_val = self.astropy_cosmo.H(z).value
        fz = M.scale_independent_growth_factor_f(z)
        a = 1. / (1 + z)
        
        prefactor = np.divide((a * Hz_val * fz), (k * c_kms), where=k!=0)**2
        
        valid_mask = (k > 0) & (k <= 67)
        
        result = np.zeros_like(k)
        if np.any(valid_mask):
            k_valid = k[valid_mask]
            Pk_lin_val = np.exp(pk_lin_interp_func(np.log(k_valid)))
            
            extrap_mask = k_valid < k_anchor
            if np.any(extrap_mask):
                scaling = k_valid[extrap_mask] / k_anchor
                Pk_lin_val[extrap_mask] = pk_anchor * scaling

            result[valid_mask] = (prefactor[valid_mask] * Pk_lin_val * 
                                  np.divide(kpar[valid_mask], k_valid, where=k_valid!=0)**2)
        
        return result

    def _precompute_C_vv_interpolator(self, z_nulled, zmin, zmax, chi_FWHM):
        cache_key = (z_nulled, zmin, zmax, chi_FWHM)
        if cache_key in self._C_vv_interp_cache:
            return self._C_vv_interp_cache[cache_key]
        
        Kpararr = np.logspace(np.log10(1.E-6), np.log10(20*(1./chi_FWHM)), 500)
        
        Cvv_vals = np.array([
            self._C_vv_ksz_approx_quad(Kpar=k, z_nulled=z_nulled, zmin=zmin, zmax=zmax) 
            for k in Kpararr
        ]).flatten()
        
        interp = scipy.interpolate.interp1d(
            Kpararr, Cvv_vals, kind="linear", bounds_error=False, fill_value=0.0
        )
        self._C_vv_interp_cache[cache_key] = interp
        return interp

    def _C_vv_ksz_approx_integrand(self, Kperp, Kpar, z_nulled, zmin, zmax):
        return self.Pvv(z=z_nulled, zmin=zmin, zmax=zmax, kpar=Kpar, kperp=Kperp) * Kperp**2

    def _C_vv_ksz_approx_quad(self, Kpar, z_nulled, zmin, zmax):
        result, _ = scipy.integrate.quad_vec(
            lambda log_Kperp: self._C_vv_ksz_approx_integrand(
                np.exp(log_Kperp), Kpar, z_nulled, zmin, zmax
            ), 
            -10.0, 4.5, limit=100000, epsabs=0., epsrel=1e-4
        )
        return result * (1 / (2 * np.pi))

    # =========================================================================
    # Lensing/Matter Power Spectrum Integration
    # =========================================================================

    def kernels_overlap(self, bin_i, bin_j, overlap_factor=5.0):
        """
        Check if two nulled kernels have significant overlap in χ-space.
        
        Kernels overlap if their centers are within (FWHM_i + FWHM_j) × overlap_factor / 2.
        
        Parameters
        ----------
        bin_i, bin_j : int
            Bin indices
        overlap_factor : float
            Number of (half-widths) beyond which overlap is negligible.
            Default 5.0
            
        Returns
        -------
        bool
            True if kernels overlap, False if signal should be set to zero.
        """
        kd_i = self.kernel_data_list[bin_i]
        kd_j = self.kernel_data_list[bin_j]
        
        chi_i = kd_i['chi_nulled']
        chi_j = kd_j['chi_nulled']
        fwhm_i = kd_i['chi_FWHM']
        fwhm_j = kd_j['chi_FWHM']
        
        separation = np.abs(chi_i - chi_j)
        overlap_threshold = overlap_factor * (fwhm_i + fwhm_j) / 2.0
    
        return separation < overlap_threshold
    
    def C_kappa_cross_approx_quad(self, kperp, bin_i, bin_j, z_eval, zmin, zmax):

        # Early exit if kernels don't overlap
        if not self.kernels_overlap(bin_i, bin_j):
            return 0.0
        pk_results = self.get_power_spectra(z_pk=z_eval, z_min=zmin, z_max=zmax)
        pk_mm_interp = pk_results['pk_mm_total_interp']
        
        kd_i = self.kernel_data_list[bin_i]
        kd_j = self.kernel_data_list[bin_j]
        chi_FWHM = 0.5 * (kd_i['chi_FWHM'] + kd_j['chi_FWHM'])
        
        def integrand(kpar):
            k_tot = np.sqrt(kpar**2 + kperp**2)
            
            wk_i_base, wk_i_phase = self.kernel_fts_auto[bin_i].get_transform(
                kpar, kd_i['chi_nulled']
            )
            wk_j_base, wk_j_phase = self.kernel_fts_auto[bin_j].get_transform(
                kpar, kd_j['chi_nulled']
            )
            
            wk_product = (wk_i_base * wk_i_phase) * np.conj(wk_j_base * wk_j_phase)
            
            pk_mm = np.exp(pk_mm_interp(np.log(k_tot)))
            return np.real(wk_product) * pk_mm
        
        result, _ = scipy.integrate.quad_vec(
            integrand, 1.E-6, 20*(1./chi_FWHM), limit=100000, epsabs=0., epsrel=1e-4
        )
        return 2 * result * (1 / (2 * np.pi))

    # def C_velocity_cross_approx_quad(self, bin_i, bin_j, z_eval, zmin, zmax, sigma_chi=None):
    #     if not self.kernels_overlap(bin_i, bin_j):
    #         return 0.0
    #     kd_i = self.kernel_data_list[bin_i]
    #     kd_j = self.kernel_data_list[bin_j]
    #     chi_FWHM = 0.5 * (kd_i['chi_FWHM'] + kd_j['chi_FWHM'])
        
    #     z_ref = 0.5 * (kd_i['z_nulled'] + kd_j['z_nulled'])
    #     C_vv_interp = self._precompute_C_vv_interpolator(z_ref, zmin, zmax, chi_FWHM)
        
    #     def integrand(Kpar):
    #         wv_i_base, wv_i_phase = self.kernel_fts_auto[bin_i].get_transform(
    #             Kpar, kd_i['chi_nulled']
    #         )
    #         wv_j_base, wv_j_phase = self.kernel_fts_auto[bin_j].get_transform(
    #             Kpar, kd_j['chi_nulled']
    #         )
            
    #         wv_product = (wv_i_base * wv_i_phase) * np.conj(wv_j_base * wv_j_phase)
            
    #         Cvv = C_vv_interp(np.abs(Kpar))
    #         # Photo-z radial smearing on the velocity power
    #         # (ThatThat: factor of 1 in the exponent).
    #         if sigma_chi is not None:
    #             Cvv = Cvv * np.exp(-Kpar**2 * sigma_chi**2)
            
    #         return np.real(wv_product) * Cvv
        
    #     result, _ = scipy.integrate.quad_vec(
    #         integrand, 1.E-6, 20*(1./chi_FWHM), limit=100000, epsabs=0., epsrel=1e-4
    #     )
    #     return 2 * result * (1 / (2 * np.pi))

    def C_velocity_cross_approx_quad(self, bin_i, bin_j, z_eval, zmin, zmax, sigma_chi=None):
        if not self.kernels_overlap(bin_i, bin_j):
            return 0.0
        kd_i = self.kernel_data_list[bin_i]
        kd_j = self.kernel_data_list[bin_j]
        chi_FWHM = 0.5 * (kd_i['chi_FWHM'] + kd_j['chi_FWHM'])
    
        z_ref = 0.5 * (kd_i['z_nulled'] + kd_j['z_nulled'])
        C_vv_interp = self._precompute_C_vv_interpolator(z_ref, zmin, zmax, chi_FWHM)
    
        delta_chi = kd_i['chi_nulled'] - kd_j['chi_nulled']
        Kmax = 20.0 / chi_FWHM
    
        # Anti-aliasing: chi window 2*pi/dK must exceed |delta_chi| + kernel supports.
        chi_window = 3.0 * (np.abs(delta_chi) + 6.0 * chi_FWHM)
        n = int(2 ** np.ceil(np.log2(2 * Kmax * chi_window / (2 * np.pi))))
        Kg = np.linspace(-Kmax, Kmax, n, endpoint=False)
        dK = Kg[1] - Kg[0]
    
        ft_i = self.kernel_fts_auto[bin_i]
        ft_j = self.kernel_fts_auto[bin_j]
        wv_i = ft_i.fft_interp_real(Kg) + 1j * ft_i.fft_interp_imag(Kg)
        wv_j = ft_j.fft_interp_real(Kg) + 1j * ft_j.fft_interp_imag(Kg)
    
        h = wv_i * np.conj(wv_j) * C_vv_interp(np.abs(Kg))
        if sigma_chi is not None:
            h = h * np.exp(-Kg**2 * sigma_chi**2)
    
        # h_tilde(chi) = int dK/(2*pi) h(K) exp(-i K chi), evaluated at delta_chi.
        # Grid starts at -Kmax, so multiply by the phase that re-centers the FFT.
        h_tilde = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(h))) * dK / (2 * np.pi)
        chi_axis = np.fft.fftshift(np.fft.fftfreq(n, dK)) * 2 * np.pi
        return float(np.interp(delta_chi, chi_axis, h_tilde.real))

    # =========================================================================
    # Cross-correlation Calculation
    # =========================================================================
    
    def _inner_integrand_cross(self, Kpar, kpar, kperp, pkem_interp, C_vv_interp,
                                density_bin, velocity_bin, sigma_chi=None):
        delta_k = kpar - Kpar
        ktot = np.sqrt(delta_k**2 + kperp**2)
        
        kd_i = self.kernel_data_list[density_bin]
        kd_j = self.kernel_data_list[velocity_bin]

        _, wk_phase = self.kernel_fts_cross[density_bin].get_transform(
            Kpar, kd_i['chi_nulled']
        )
        
        wk_base, _ = self.kernel_fts_cross[density_bin].get_transform(
            delta_k, kd_i['chi_nulled']
        )
        wv_base, wv_phase = self.kernel_fts_cross[velocity_bin].get_transform(
            Kpar, kd_j['chi_nulled']
        )
        net_phase = wv_phase * np.conj(wk_phase)
        pkem = np.exp(pkem_interp(np.log(ktot)))
        Cvv = C_vv_interp(np.abs(Kpar))
        
        # Photo-z radial smearing on the velocity power (TThat: 1/2 in exponent).
        if sigma_chi is not None:
            Cvv = Cvv * np.exp(-Kpar**2 * sigma_chi**2 / 2.0)
        
        return wk_base * wv_base * net_phase * pkem * Cvv

    def _hatT_T_full_integrand_cross(self, kpar, kperp, inner_interp, 
                                      density_bin, chi_ksz_peak, ksz_ft):
        wksz_base, wksz_phase = ksz_ft.get_transform(-kpar, chi_ksz_peak)
        
        kd_i = self.kernel_data_list[density_bin]
        _, wk_phase = self.kernel_fts_cross[density_bin].get_transform(
            kpar, kd_i['chi_nulled']
        )
        
        net_phase = wksz_phase * wk_phase
        inner_result = inner_interp((kpar, kperp))
        
        return wksz_base * net_phase * inner_result

    def calculate_hatT_T_full_cross(self, l_vals, density_bin, velocity_bin, 
                                     chi_ksz_peak, zmin, zmax):
        kd_i = self.kernel_data_list[density_bin]   # density / kappa bin
        kd_j = self.kernel_data_list[velocity_bin]  # velocity bin
        
        # Evaluate at the chi-midpoint of the kappa and velocity peaks,
        # then map back to redshift via fz_chi (Pk_em is computed at this z).
        chi_i = kd_i['chi_nulled']       # kappa (density) peak
        chi_alpha = kd_j['chi_nulled']   # velocity peak
        chi_eval = 0.5 * (chi_i + chi_alpha)
        z_eval = float(self.fz_chi(chi_eval))
        chi_FWHM = 0.5 * (kd_i['chi_FWHM'] + kd_j['chi_FWHM'])
        
        # Photo-z radial smearing for the velocity field, evaluated at the
        # velocity-bin redshift (matches where C_vv is precomputed below).
        sigma_chi = sigma_chi_of_z(
            kd_j['z_nulled'], self.astropy_cosmo, self.sigma_z_over_1plusz
        )

        # Non-overlapping kernel pairs contribute no signal (matches the
        # early-exit in C_kappa/C_velocity, putting signal and covariance on
        # the same footing).
        if not self.kernels_overlap(density_bin, velocity_bin):
            zeros = np.zeros(len(l_vals), dtype=complex)
            return {
                'C_l_photoz': zeros,
                'C_l_nopz': zeros.copy(),
                'sigma_chi': sigma_chi,
            }
        # kSZ kernel FT built fresh for this (kappa, velocity) bin pair.
        ksz_ft = KSZKernelFT(
            self.ksz_kernel_interp,
            Dz_interp=self.Dz_interp_chi,
            Dv_interp=self.Dv_interp_chi,
            chi_ksz_peak=chi_ksz_peak,
            chi_nulled_kappa=chi_i,
            chi_nulled_velocity=chi_alpha,
        )
        
        C_vv_interp = self._precompute_C_vv_interpolator(
            z_eval, zmin, zmax, chi_FWHM
        )
        
        pk_results = self.get_power_spectra(z_pk=z_eval, z_min=zmin, z_max=zmax)
        pkem_interp = pk_results['pk_em_total_interp']
        
        klower, kupper = -20./chi_FWHM, 20./chi_FWHM
        
        kperp_vals = l_vals / chi_eval
        n_kpar = max(1000, int(30 * (kupper-klower) * abs(chi_i-chi_alpha) / (2*np.pi)))
        kpar_vals = np.linspace(klower, kupper, n_kpar)
        kpar_grid, kperp_grid = np.meshgrid(kpar_vals, kperp_vals, indexing="ij")
        
        # Growth normalization for evaluating at the chi-midpoint:
        # D(chi_i) D(chi_alpha) / D^2((chi_i + chi_alpha)/2)
        D_norm = 1.0 #(self.Dz_interp_chi(chi_i) * self.Dz_interp_chi(chi_alpha)/ self.Dz_interp_chi(chi_eval)**2)
        norm = (1 / (2 * np.pi))**2 * (1 / chi_eval)**2 * D_norm
        
        # The photo-z smearing S enters ONLY the inner Kpar integral (via
        # _inner_integrand_cross). Everything above (kSZ FT, P_em, C_vv, grids,
        # normalization) is shared, so the two cases differ only here.
        #   sig = sigma_chi -> photo-z smearing S
        #   sig = None       -> S = 1 (no smearing)
        def run_variant(sig):
            # def inner_quad(kpar, kperp):
            #     res, _ = scipy.integrate.quad_vec(
            #         lambda Kpar: self._inner_integrand_cross(
            #             Kpar, kpar, kperp, pkem_interp, C_vv_interp,
            #             density_bin, velocity_bin, sigma_chi=sig
            #         ),
            #         klower, kupper, limit=100000, epsabs=0., epsrel=1e-4
            #     )
            #     return res
            
            # values_flat = inner_quad(kpar_grid.ravel(), kperp_grid.ravel())
            # values_vec = np.array(values_flat, dtype=complex).reshape(
            #     len(kpar_vals), len(kperp_vals)
            # )
            # inner_interp = scipy.interpolate.RegularGridInterpolator(
            #     (kpar_vals, kperp_vals), values_vec, bounds_error=False, fill_value=0.0
            # )
            delta_chi = kd_j['chi_nulled'] - kd_i['chi_nulled']

            Kmax = 20.0 / chi_FWHM
            # dK must resolve kernel structure AND the phase; 2*pi/dK (chi-space extent)
            # must exceed |delta_chi| + kernel supports or g~ wraps around (aliasing).
            chi_window = 3.0 * (abs(delta_chi) + 6.0 * chi_FWHM + 500.0)

            # n from the two constraints: extent 2*Kmax, spacing 2*pi/chi_window
            n = int(2 ** np.ceil(np.log2(2 * Kmax * chi_window / (2 * np.pi))))
            Kg = np.linspace(-Kmax, Kmax, n, endpoint=False)
            dK = Kg[1] - Kg[0]
            
            ft_k = self.kernel_fts_cross[density_bin]
            ft_v = self.kernel_fts_cross[velocity_bin]
            wk_b = ft_k.fft_interp_real(Kg) + 1j * ft_k.fft_interp_imag(Kg)
            wv_b = ft_v.fft_interp_real(Kg) + 1j * ft_v.fft_interp_imag(Kg)
            
            S = np.exp(-Kg**2 * sig**2 / 2.0) if sig is not None else 1.0
            g = wv_b * C_vv_interp(np.abs(Kg)) * S * np.exp(-1j * Kg * delta_chi)
            
            inner_vals = np.empty((n, len(kperp_vals)), dtype=complex)
            for m, kperp in enumerate(kperp_vals):
                f = wk_b * np.exp(pkem_interp(np.log(np.sqrt(Kg**2 + kperp**2))))
                inner_vals[:, m] = scipy.signal.fftconvolve(f, g, mode='same') * dK 
            
            inner_interp = scipy.interpolate.RegularGridInterpolator(
                (Kg, kperp_vals), inner_vals, bounds_error=False, fill_value=0.0)
            def final_quad(kperp):
                res, _ = scipy.integrate.quad_vec(
                    lambda kpar: self._hatT_T_full_integrand_cross(
                        kpar, kperp, inner_interp, density_bin, chi_ksz_peak, ksz_ft
                    ),
                    klower, kupper, limit=100000, epsabs=0., epsrel=1e-4
                )
                return res
            
            final_results = np.array([final_quad(kp) for kp in kperp_vals])
            return final_results * norm
        
        return {
            'C_l_photoz': run_variant(sigma_chi),  # with photo-z S
            'C_l_nopz': run_variant(None),         # S = 1
            'sigma_chi': sigma_chi,
        }

    # =========================================================================
    # Auto-correlation (Diagonal) Calculation
    # =========================================================================
    
    def hatT_hatT_approx(self, l_vals, bin_idx, zmin, zmax):
        kd = self.kernel_data_list[bin_idx]
        z_nulled = kd['z_nulled']
        chi_FWHM = kd['chi_FWHM']
        Nell = kd['Nell']
        
        chi_at_z = self.fchi_z(z_nulled)
        
        C_vv_interp = self._precompute_C_vv_interpolator(z_nulled, zmin, zmax, chi_FWHM)
        
        kperp_vals = l_vals / chi_at_z
        C_kappa = np.array([
            self.C_kappa_cross_approx_quad(kp, bin_idx, bin_idx, z_nulled, zmin, zmax) 
            for kp in kperp_vals
        ])
        
        C_kappa_angular = C_kappa * (1 / chi_at_z)**2
        C_kappa_with_noise = C_kappa_angular + Nell
        
        # Photo-z radial smearing on the velocity power (ThatThat).
        sigma_chi = sigma_chi_of_z(
            z_nulled, self.astropy_cosmo, self.sigma_z_over_1plusz
        )
        C_velocity = self.C_velocity_cross_approx_quad(
            bin_idx, bin_idx, z_nulled, zmin, zmax, sigma_chi=sigma_chi
        )
        
        hatT_hatT_with_noise = C_kappa_with_noise * C_velocity
        hatT_hatT_without_noise = C_kappa_angular * C_velocity
        
        return {
            'C_kappa': C_kappa_angular,
            'C_kappa_with_noise': C_kappa_with_noise,
            'C_velocity': C_velocity,
            'hatT_hatT_with_noise': hatT_hatT_with_noise,
            'hatT_hatT_without_noise': hatT_hatT_without_noise
        }

    # =========================================================================
    # Covariance Matrix Calculation
    # =========================================================================
    
    def compute_covariance_matrix(self, l_vals, zmin, zmax, N_matrix=None, 
                                   C_kappa_matrix_precomputed=None):
        """
        Compute the full covariance matrix for all bin combinations.
        
        Formula:
            Cov[C_ℓ^{ij}, C_ℓ^{i'j'}] = (C_κ^{ii'}(ℓ) + N^{ii'}) × C_v^{jj'}
        
        where:
            - i, i' are density bin indices
            - j, j' are velocity bin indices
            - C_κ^{ii'}(ℓ) is the lensing cross-power (cosmic variance)
            - N^{ii'} is the FULL nulled noise matrix (with off-diagonal correlations)
            - C_v^{jj'} is the velocity cross-variance
        
        No factors of f_sky or (2ℓ+1) are included.
        
        Parameters
        ----------
        l_vals : array
            Multipole values
        zmin, zmax : float
            Redshift bounds for CLASS computations
        N_matrix : ndarray, optional
            Full nulled noise covariance matrix, shape (n_bins, n_bins).
            If None, uses diagonal noise from kernel_data_list (less accurate).
        C_kappa_matrix_precomputed : ndarray, optional
            Pre-computed C_κ^{ii'}(ℓ), shape (n_ell, n_bins, n_bins).
            If provided, skips C_kappa computation (faster).
            
        Returns
        -------
        cov_matrix : ndarray
            Shape (n_ells, n_bins, n_bins, n_bins, n_bins)
            Indices: [ell_idx, i_density, j_velocity, ip_density, jp_velocity]
        C_kappa_matrix : ndarray
            Shape (n_ells, n_bins, n_bins)
        C_velocity_matrix : ndarray
            Shape (n_bins, n_bins)
        N_matrix_used : ndarray
            Shape (n_bins, n_bins) - the noise matrix that was used
        """
        n_bins = self.n_bins
        n_ells = len(l_vals)

        # Pre-compute overlap mask based on kernel geometry
        overlap_mask = np.zeros((n_bins, n_bins), dtype=bool)
        for i in range(n_bins):
            for j in range(n_bins):
                overlap_mask[i, j] = self.kernels_overlap(i, j)
        
        print(f"  Kernel overlap matrix: {np.sum(overlap_mask)} / {n_bins**2} pairs overlap")

        # ---------------------------------------------------------------------
        # Step 1: C_kappa matrix (cosmic variance)
        # ---------------------------------------------------------------------
        if C_kappa_matrix_precomputed is not None:
            C_kappa_matrix = C_kappa_matrix_precomputed
            print(f"  Using pre-computed C_kappa matrix")
        else:
            print(f"  Computing C_kappa for {n_bins*(n_bins+1)//2} bin pairs...")
            C_kappa_matrix = np.zeros((n_ells, n_bins, n_bins))
            
            for i in range(n_bins):
                for ip in range(i, n_bins):
                    # Skip if kernels don't overlap
                    if not overlap_mask[i, ip]:
                        # C_kappa remains zero (already initialized)
                        continue
                    # Average in comoving distance, then map back to redshift.
                    chi_i = self.kernel_data_list[i]['chi_nulled']
                    chi_ip = self.kernel_data_list[ip]['chi_nulled']
                    chi_eval = 0.5 * (chi_i + chi_ip)
                    z_eval = float(self.fz_chi(chi_eval))
                    
                    kperp_vals = l_vals / chi_eval
                    
                    C_kappa = np.array([
                        self.C_kappa_cross_approx_quad(kp, i, ip, z_eval, zmin, zmax)
                        for kp in kperp_vals
                    ])
                    
                    C_kappa_angular = C_kappa * (1 / chi_eval)**2
                    
                    C_kappa_matrix[:, i, ip] = C_kappa_angular
                    C_kappa_matrix[:, ip, i] = C_kappa_angular
                
                print(f"    Row {i+1}/{n_bins} done")
        
        # ---------------------------------------------------------------------
        # Step 1b: growth normalization of C_kappa (kappa pair i, i')
        #   norm_kappa[i,i'] = D(chi_i) D(chi_i') / D^2((chi_i + chi_i')/2)
        # Applied to BOTH freshly computed and precomputed C_kappa for a
        # uniform covariance formula. NOTE: if a precomputed C_kappa already
        # includes this growth weighting, disable this block to avoid double
        # counting.
        # ---------------------------------------------------------------------
        # Step 1b: only for FRESHLY computed C_kappa. A precomputed C_kappa
        # (nulled_aps_cl.npz) already carries this growth weighting, so applying
        # it here would double-count.
        if C_kappa_matrix_precomputed is None:
            for i in range(n_bins):
                for ip in range(n_bins):
                    if not overlap_mask[i, ip]:
                        continue
                    chi_i = self.kernel_data_list[i]['chi_nulled']
                    chi_ip = self.kernel_data_list[ip]['chi_nulled']
                    chi_mid = 0.5 * (chi_i + chi_ip)
                    norm_kappa = 1.0 #(self.Dz_interp_chi(chi_i) * self.Dz_interp_chi(chi_ip)/ self.Dz_interp_chi(chi_mid)**2)
                    C_kappa_matrix[:, i, ip] = C_kappa_matrix[:, i, ip] * norm_kappa

        # ---------------------------------------------------------------------
        # Step 2: C_velocity matrix (velocity cross-variance, ℓ-independent)
        # ---------------------------------------------------------------------
        print(f"  Computing C_velocity for {n_bins*(n_bins+1)//2} bin pairs...")
        C_velocity_matrix = np.zeros((n_bins, n_bins))
        
        for j in range(n_bins):
            for jp in range(j, n_bins):
                # Skip if kernels don't overlap
                if not overlap_mask[j, jp]:
                    # C_kappa remains zero (already initialized)
                    continue
                # Photo-z radial smearing (ThatThat convention, no 1/2 in the
                # exponent), evaluated at the average redshift of the two
                # velocity bins, matching the z_ref C_vv is precomputed at.
                z_ref = 0.5 * (self.kernel_data_list[j]['z_nulled']
                               + self.kernel_data_list[jp]['z_nulled'])
                sigma_chi = sigma_chi_of_z(
                    z_ref, self.astropy_cosmo, self.sigma_z_over_1plusz
                )
                C_v = self.C_velocity_cross_approx_quad(
                    j, jp, self.kernel_data_list[j]['z_nulled'], zmin, zmax,
                    sigma_chi=sigma_chi
                )
                # Growth normalization (velocity pair j, j'):
                #   norm_v[j,j'] = D(chi_j) D(chi_j') / D^2((chi_j + chi_j')/2)
                chi_j = self.kernel_data_list[j]['chi_nulled']
                chi_jp = self.kernel_data_list[jp]['chi_nulled']
                chi_mid = 0.5 * (chi_j + chi_jp)
                norm_velocity = 1.0 #(self.Dz_interp_chi(chi_j) * self.Dz_interp_chi(chi_jp) / self.Dz_interp_chi(chi_mid)**2)
                C_v = C_v * norm_velocity
                C_velocity_matrix[j, jp] = C_v
                C_velocity_matrix[jp, j] = C_v
            
            print(f"    Row {j+1}/{n_bins} done")
        
        # ---------------------------------------------------------------------
        # Step 3: Noise matrix
        # ---------------------------------------------------------------------
        if N_matrix is not None:
            # Use provided full noise matrix (includes off-diagonal correlations)
            N_matrix_used = N_matrix
            print(f"  Using provided full noise matrix (with off-diagonals)")
        else:
            # Fall back to diagonal-only noise (less accurate)
            print(f"  WARNING: Using diagonal-only noise matrix. "
                  f"For accurate results, load full N_matrix from "
                  f"nulled_noise_power_covariance_matrix.npz")
            N_matrix_used = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                N_matrix_used[i, i] = self.kernel_data_list[i]['Nell']
        
        # ---------------------------------------------------------------------
        # Step 4: Assemble covariance (vectorized for speed)
        # ---------------------------------------------------------------------
        # Cov[C_ℓ^{ij}, C_ℓ^{i'j'}] = (C_κ^{ii'}(ℓ) + N^{ii'}) × C_v^{jj'}
        print(f"  Assembling covariance matrix...")
        
        # Add noise to lensing: (C_κ + N) has shape (n_ell, n_bins, n_bins)
        C_kappa_plus_N = C_kappa_matrix + N_matrix_used[np.newaxis, :, :]
        
        # cov[ℓ, i, j, i', j'] = (C_κ + N)[ℓ, i, i'] × C_v[j, j']
        cov_matrix = np.zeros((n_ells, n_bins, n_bins, n_bins, n_bins), dtype=np.float32)
        for j in range(n_bins):
            for jp in range(n_bins):
                # Broadcasts across ell, i, and ip only (very memory cheap)
                cov_matrix[:, :, j, :, jp] = C_kappa_plus_N * C_velocity_matrix[j, jp]
        
        print(f"  Covariance shape: {cov_matrix.shape}")
        
        return cov_matrix, C_kappa_matrix, C_velocity_matrix, N_matrix_used


# =============================================================================
# SHARED SETUP FUNCTIONS
# =============================================================================

def setup_mpi():
    """Initialize MPI and return comm, rank, size."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        print("Error: mpi4py not installed. Please install it to run this script.")
        sys.exit(1)
    
    os.environ['OMP_NUM_THREADS'] = "1"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)
    
    return comm, rank, size


def get_default_params():
    """Return the default cosmological and CLASS_SZ parameter dictionaries."""
    cosmo_params = {
        'H0': cosmo.H0.value,
        'omega_b': cosmo.Ob0 * cosmo.h**2,
        'omega_cdm': cosmo.Odm0 * cosmo.h**2,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'tau_reio': 0.0561
    }
    
    common_params = {
        'ndim_redshifts': 100,
        'M_min': 1.0e10,
        'M_max': 3.5e15,
        'ndim_masses': 100,
        'mass function': 'T08M200c',
        'concentration parameter': 'B13',
        'delta_for_galaxies': "200c",
        'delta_for_matter_density': "200c",
        'redshift_epsabs': 1.0e-40,
        'redshift_epsrel': 0.0005,
        'mass_epsabs': 1.0e-40,
        'mass_epsrel': 0.0005,
        'ell_max': 40000.0,
        'ell_min': 2.0,
        'dell': 10,
        'non_linear': 'hmcode',
        'hm_consistency': 1
    }
    
    ksz_params = {
        "ell_min": 2,
        "ell_max": 40000,
        'dell': 0,
        'dlogell': 0.2,
        'gas profile': 'B16',
        'gas profile mode': 'agn',
        'normalize_gas_density_profile': 0,
        'use_xout_in_density_profile_from_enclosed_mass': 1,
        'n_m_density_profile': 100,
        'n_z_density_profile': 100,
        'N_samp_fftw': 1024,
        'use_fft_for_profiles_transform': 1,
        'x_min_gas_density_fftw': 1e-8,
        'x_max_gas_density_fftw': 1e8
    }
    
    Pk_matter_params = {
        'k_min_for_pk_class_sz': 0.00001,
        'k_max_for_pk_class_sz': 100.0,
        'k_per_decade_class_sz': 500.0,
        'P_k_max_h/Mpc': 100.0
    }
    
    return cosmo_params, common_params, ksz_params, Pk_matter_params


def setup_cosmology_interpolators():
    """Create z(χ) and χ(z) interpolators and growth factor D(χ)."""
    ZMIN, ZMAX = 1.E-6, 12.0
    z_grid = np.logspace(np.log10(ZMIN), np.log10(ZMAX), num=10000)
    chi_grid = cosmo.comoving_distance(z_grid).value
    
    fz_chi = scipy.interpolate.interp1d(
        chi_grid, z_grid, bounds_error=False, fill_value=0., kind='linear'
    )
    fchi_z = scipy.interpolate.interp1d(
        z_grid, chi_grid, bounds_error=False, fill_value=0.
    )
    
    return z_grid, chi_grid, fz_chi, fchi_z


def compute_growth_factor(z_grid, chi_grid, common_params, cosmo_params, 
                           ksz_params, Pk_matter_params):
    """Compute growth factor D(z) and return interpolator D(χ)."""
    M_init = Class_sz()
    M_init.set(common_params)
    M_init.set(cosmo_params)
    M_init.set(ksz_params)
    M_init.set(Pk_matter_params)
    M_init.set({'output': 'mPk', 'z_max_pk': 12.0})
    M_init.compute()
    
    Dz_grid = np.array([M_init.scale_independent_growth_factor(z) for z in z_grid])
    Dz_interp_chi = scipy.interpolate.interp1d(
        chi_grid, Dz_grid, bounds_error=False, fill_value=0.0
    )
    return Dz_interp_chi


def compute_velocity_growth_factor(z_grid, chi_grid, common_params, cosmo_params,
                                   ksz_params, Pk_matter_params):
    """Compute velocity growth factor Dv(z) = a * D(z) * f(z) * H(z) and return
    interpolator Dv(χ).

    Here a = 1/(1+z), D is the (density) growth factor, f the growth rate, and
    H(z) the Hubble rate in km/s/Mpc (from astropy). Dv therefore carries units
    of km/s/Mpc, but it is only ever used as a ratio Dv(χ)/Dv(χ_nulled), so the
    units (and any common prefactors) cancel.
    """
    M_init = Class_sz()
    M_init.set(common_params)
    M_init.set(cosmo_params)
    M_init.set(ksz_params)
    M_init.set(Pk_matter_params)
    M_init.set({'output': 'mPk', 'z_max_pk': 12.0})
    M_init.compute()

    a_grid = 1.0 / (1.0 + z_grid)
    D_grid = np.array([M_init.scale_independent_growth_factor(z) for z in z_grid])
    f_grid = np.array([M_init.scale_independent_growth_factor_f(z) for z in z_grid])
    H_grid = np.array([cosmo.H(z).value for z in z_grid])
    Dv_grid = a_grid * D_grid * f_grid * H_grid

    Dv_interp_chi = scipy.interpolate.interp1d(
        chi_grid, Dv_grid, bounds_error=False, fill_value=0.0
    )
    return Dv_interp_chi


def load_ksz_kernel(fchi_z):
    """Load kSZ kernel data and return interpolator and peak position."""
    filename_ksz = os.path.join(REPO_ROOT, 'results/kSZ_kernel/kSZ_weight_function_zmax_8.npz')
    if not os.path.exists(filename_ksz):
        raise FileNotFoundError(f"kSZ kernel file not found: {filename_ksz}")
    
    ksz_kernel_data = np.load(filename_ksz)
    chi_ksz_values = fchi_z(ksz_kernel_data['z'])
    chi_ksz_peak = float(ksz_kernel_data['chi_ksz_peak'][0])
    
    ksz_kernel_interp = scipy.interpolate.interp1d(
        chi_ksz_values - chi_ksz_peak,
        ksz_kernel_data['kSZ_wf'],
        kind='quadratic', bounds_error=False, fill_value=0.0
    )
    return ksz_kernel_interp, chi_ksz_peak


def load_kernel_data(input_kernel_dir, fchi_z):
    """Load all nulled kernel files and return list of kernel data dicts."""
    kernel_files = sorted(glob.glob(os.path.join(input_kernel_dir, "nulled_kernel_k*.npz")))
    
    kernel_data_list = []
    for kf in kernel_files:
        data = np.load(kf)
        chi_values = fchi_z(data['z'])
        z_nulled = float(data['zpeak'])
        chi_nulled = float(fchi_z(z_nulled))
        
        kernel_interp = scipy.interpolate.interp1d(
            chi_values - chi_nulled,
            data['nulled_wf'],
            kind='cubic', bounds_error=False, fill_value=0.0
        )
        
        kernel_data_list.append({
            'filename': kf,
            'z_nulled': z_nulled,
            'chi_nulled': chi_nulled,
            'chi_FWHM': float(data['chi_FWHM']),
            'Nell': float(data['nulled_noise_power_diag']),
            'kernel_interp': kernel_interp
        })
    
    return kernel_data_list


def load_noise_matrix(input_kernel_dir):
    """
    Load the full nulled noise covariance matrix N^{ii'}.
    
    This matrix has OFF-DIAGONAL elements because consecutive nulled kernels
    share base tomographic bins in their linear combinations:
        N_nulled = C @ diag(N_base) @ C^T
    
    Returns
    -------
    N_matrix : ndarray
        Full noise covariance matrix, shape (n_bins, n_bins)
    C_coeff : ndarray
        Coefficient matrix mapping base bins to nulled kernels
    """
    noise_file = os.path.join(input_kernel_dir, 'nulled_noise_power_covariance_matrix.npz')
    
    if not os.path.exists(noise_file):
        raise FileNotFoundError(
            f"Noise covariance matrix not found: {noise_file}\n"
            f"This should be generated by the kernel construction code."
        )
    
    data = np.load(noise_file)
    N_matrix = data['nulled_noise_power_matrix']  # Full matrix with off-diagonals
    C_coeff = data['C_coeff_matrix']
    
    return N_matrix, C_coeff
    
def load_precomputed_ckappa(input_kernel_dir):
    """
    Load pre-computed lensing angular power spectrum C_κ^{ii'}(ℓ).
    
    Returns
    -------
    ell_values : ndarray
        Multipole values, shape (n_ell,)
    C_kappa_matrix : ndarray
        Lensing cross-power, shape (n_ell, n_bins, n_bins)
    """
    aps_file = os.path.join(input_kernel_dir, 'nulled_aps_cl.npz')
    
    if not os.path.exists(aps_file):
        raise FileNotFoundError(
            f"Pre-computed C_kappa not found: {aps_file}\n"
            f"Run with --recompute-ckappa or regenerate kernels."
        )
    
    data = np.load(aps_file)
    ell_values = data['ell']
    C_kappa_matrix = data['cl_ij']  # Shape: (n_ell, n_bins, n_bins)
    
    return ell_values, C_kappa_matrix


def get_sigma_z_over_1plusz(survey):
    """Photo-z scatter sigma_z/(1+z) for each survey.

    rubin/roman_wl -> 0.03, roman_kl -> 0.002. The 'toy' case is not specified;
    it defaults to 0.0, which makes sigma_chi = 0 and disables photo-z smearing.
    """
    mapping = {
        'rubin': 0.03,
        'roman_wl': 0.03,
        'roman_kl': 0.002,
        'toy': 0.0,  # FLAG: not specified; 0.0 disables smearing
    }
    if survey not in mapping:
        raise ValueError(f"Unknown survey: {survey}")
    return mapping[survey]


def get_input_output_dirs(survey, width):
    """Return input kernel directory and output directory for a given survey/width."""
    width_str = f'{width:g}'
    
    if survey == 'rubin':
        input_kernel_dir = f'{REPO_ROOT}/results/lensing_kernels_full_range_rubin_y10/nulled_kernel_width_{width_str}_Mpc'
        output_dir = f'{REPO_ROOT}/results/cross_and_auto_correlations/rubin_all_bins/width_{width_str}'
    elif survey == 'roman_wl':
        input_kernel_dir = f'{REPO_ROOT}/results/lensing_kernels_full_range_roman_wl/nulled_kernel_width_{width_str}_Mpc'
        output_dir = f'{REPO_ROOT}/results/cross_and_auto_correlations/roman_wl_all_bins/width_{width_str}'
    elif survey == 'roman_kl':
        input_kernel_dir = f'{REPO_ROOT}/results/lensing_kernels_full_range_roman_kl/nulled_kernel_width_{width_str}_Mpc'
        output_dir = f'{REPO_ROOT}/results/cross_and_auto_correlations/roman_kl_all_bins/width_{width_str}'
    else:
        input_kernel_dir = f'{REPO_ROOT}/results/lensing_kernels_constant_dndz/nulled_kernel_width_{width_str}_Mpc'
        output_dir = f'{REPO_ROOT}/results/cross_and_auto_correlations/toy_all_bins/width_{width_str}'
    
    return input_kernel_dir, output_dir


def build_argparser(description):
    """Create the common argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--width', type=float, required=True,
                        help='Chi width of lensing kernel [Mpc]')
    parser.add_argument('--survey', type=str, default='rubin',
                        choices=['rubin', 'roman_wl', 'roman_kl', 'toy'],
                        help='Survey to use (rubin or roman or toy problem (const dndz))')
    return parser