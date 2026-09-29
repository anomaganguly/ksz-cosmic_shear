"""
snr_simple.py  --  minimal SNR via full covariance + correlation transform.

Per ell:
  1. Build full covariance C = A_ell (x) B.
  2. Report cond(C).
  3. Transform to correlation matrix R = D^{-1/2} C D^{-1/2}, D = diag(C).
     Transform signal: d' = D^{-1/2} d.
  4. Report cond(R).
  5. Solve R x = d', SNR^2 contribution = d' . x.

Also computes per-ell correlation coefficient
  r(l) = <T That> / sqrt(<That That> <TT>)
for the Wiener-combined template That = w^T Thatvec, w = C^{-1} <T Thatvec>.
Algebraically, r^2(l) = (d^T C^{-1} d) / C_ell^TT.

That's it. No fallbacks, no mode-truncation, no Kronecker shortcuts.
"""

import os
import numpy as np
import scipy.linalg
import scipy.interpolate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# Covariance loading (matches snr.py format)
# =============================================================================

_FLAVORS = {
    'no_shape_no_velnoise': (False, ['C_velocity_signal'],                 'C_l_matrix_nopz'),
    'shape_no_velnoise':    (True,  ['C_velocity_signal'],                 'C_l_matrix_nopz'),
    'shape_velnoise_nopz':  (True,  ['C_velocity_signal', 'N_vrec_nopz'],  'C_l_matrix_nopz'),
    'shape_velnoise_pz':    (True,  ['C_velocity_signal_pz', 'N_vrec_nopz'], 'C_l_matrix'),
}


def load_cov_blocks(cov_data, flavor):
    """Return (l_vals_cov, C_kappa(ell), N_kappa, B, signal_key)."""
    keys = set(cov_data.files)
    l_vals = cov_data['l_vals']
    C_kappa = cov_data['C_kappa_matrix']
    N_kappa = cov_data['N_matrix'] * (1./2.) ## noise component (1/2) because sigma epsilon was used in quadrature

    use_shape, vel_keys, signal_key = _FLAVORS[flavor]
    # N_vrec enters diagonal-only (shot-noise treatment); signal keeps full
    # off-diagonal structure.
    B = np.zeros_like(np.asarray(cov_data[vel_keys[0]], float))
    for k in vel_keys:
        arr = np.asarray(cov_data[k], float)
        if k.startswith('N_vrec'):
            arr = np.diag(np.diag(arr))
        B = B + arr
    N_used = N_kappa if use_shape else np.zeros_like(N_kappa)
    return l_vals, C_kappa, N_used, B, signal_key


def interp_signed_loglog(l_coarse, y_coarse, l_fine):
    """Linear-in-log-log interpolation preserving sign of y."""
    out = np.zeros(len(l_fine))
    y_coarse = np.asarray(y_coarse, dtype=np.float64)
    if not np.any(y_coarse != 0):
        return out
    sign = np.sign(y_coarse)
    absy = np.abs(y_coarse)
    pos = absy > 0
    if not np.any(pos):
        return out
    floor_val = absy[pos].min() * 1e-10
    absy_safe = np.where(pos, absy, floor_val)
    log_interp = scipy.interpolate.interp1d(
        np.log(l_coarse), np.log(absy_safe),
        kind='linear', fill_value='extrapolate', bounds_error=False)
    sign_interp = scipy.interpolate.interp1d(
        l_coarse, sign, kind='nearest', fill_value='extrapolate', bounds_error=False)
    return sign_interp(l_fine) * np.exp(log_interp(np.log(l_fine)))


# =============================================================================
# Main
# =============================================================================

def compute_snr_for_width(width, survey, experiment, f_sky, ell_min, ell_max,
                          n_ell_fine, get_cmb_total,
                          flavor='shape_velnoise_pz',
                          results_root=os.path.join(REPO_ROOT, 'results'),
                          cov_name='covariance_components_all_variants.npz',
                          ksz_path=os.path.join(REPO_ROOT, 'results/cl_kSZ_z_0.0005_20.0.npz'),
                          skip_r=False,
                          verbose='summary'):
    """
    Returns (snr_same, snr_all, n_bins, worst_condC, worst_condR,
             l_fine_r, r_same_of_ell, r_all_of_ell).

    r_same_of_ell, r_all_of_ell: arrays of length n_ell_fine, evaluated on the
      fixed grid l_fine_r = geomspace(100, 40000, n_ell_fine), independent of
      the CMB experiment.
      r(l) = sqrt( d^T C^{-1} d / C_ell^kSZ ) for the Wiener-combined template,
      where C_ell^kSZ is the kSZ auto-power spectrum (1h+2h) loaded from
      ksz_path. NaN at ells where l_fine_r falls outside the kSZ file's ell
      range.

    SNR values (snr_same, snr_all) are computed on a separate ell grid set by
    ell_min/ell_max/n_ell_fine and are experiment-dependent via get_cmb_total.

    verbose: 'off' / 'summary' / 'full'  (or True == 'full', False == 'off').
    """
    v = {True: 'full', False: 'off'}.get(verbose, verbose)

    base = os.path.join(results_root, 'cross_and_auto_correlations',
                        f'{survey}_all_bins', f'width_{width:g}')
    # load cov data
    cov_data = np.load(os.path.join(base, cov_name))
    # load signal data
    signal_data = np.load(os.path.join(base, 'all_cross_correlations.npz'))

    # load ell, lensing ps, lensing shape noise, vel cosmic var + noise(if any), cross-signal
    l_cov, C_kappa_coarse, N_kappa, B, signal_key = load_cov_blocks(cov_data, flavor) 
    z_bins = signal_data['z_bins']
    S_coarse = np.real(signal_data[signal_key])
    n_bins = len(z_bins)

    # Common ell range, fine grid
    l_cov = np.asarray(l_cov, float)
    l_sig = np.asarray(signal_data['l_vals'], float)
    print(min(l_cov), max(l_cov), min(l_sig), max(l_sig))
    lo = max(ell_min, float(l_cov.min()), float(l_sig.min()))
    hi = min(ell_max, float(l_cov.max()), float(l_sig.max()))
    l_fine = np.geomspace(lo, hi, n_ell_fine)
    n_ell = len(l_fine)

    # Interpolate signal and C_kappa to the fine grid
    S_fine = np.zeros((n_ell, n_bins, n_bins))
    Ck_fine = np.zeros((n_ell, n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            S_fine[:, i, j] = interp_signed_loglog(l_sig, S_coarse[:, i, j], l_fine)
            Ck_fine[:, i, j] = interp_signed_loglog(l_cov, C_kappa_coarse[:, i, j], l_fine)

    A_fine = 0.5 * ((Ck_fine + N_kappa[None]) + (Ck_fine + N_kappa[None]).transpose(0, 2, 1)) # fixed l (index 0) and swap i and j (kappa bins)
    B = 0.5 * (B + B.T) # swap alpha and beta velocity bins

    # Mode-counting / CMB-noise weight
    delta_ell = l_fine * np.gradient(np.log(l_fine))
    modes_factor = (2.0 * l_fine + 1.0) * f_sky * delta_ell
    cmb_noise = get_cmb_total(l_fine, experiment=experiment)

    snr2_all = 0.0
    snr2_same = 0.0
    worst_condC = 0.0
    worst_condR = 0.0

    for li in range(n_ell):
        noise = cmb_noise[li]
        if (not np.isfinite(noise)) or noise <= 0:
            continue
        scale = modes_factor[li] / noise

        # ---------- all-pairs: full Kronecker covariance ----------
        A = A_fine[li]
        C = np.kron(A, B)
        C = 0.5 * (C + C.T)
        d = S_fine[li].reshape(-1)

        # condition number
        wC = np.linalg.eigvalsh(C)
        condC = np.linalg.cond(C) #wC.max() / wC.min() if wC.min() > 0 else np.inf

        # correlation transform
        s = 1.0 / np.sqrt(np.diag(C))
        R = (C * s[:, None]) * s[None, :]
        R = 0.5 * (R + R.T)
        d_prime = s * d

        wR = np.linalg.eigvalsh(R)
        condR = np.linalg.cond(R) #wR.max() / wR.min() if wR.min() > 0 else np.inf

        # solve and accumulate
        x = scipy.linalg.solve(R, d_prime, assume_a='sym', check_finite=False)
        q_all = float(np.dot(d_prime, x))
        snr2_all += scale * q_all

        worst_condC = max(worst_condC, condC)
        worst_condR = max(worst_condR, condR)

        if v == 'full':
            print(f"  [ell={l_fine[li]:8.1f}] cond(C)={condC:.3e}  "
                  f"cond(R)={condR:.3e}  snr2={scale * q_all:.4e}")

        # ---------- same-bin: smaller (n_bins x n_bins) block ----------
        cov_same = 0.5 * (A * B + (A * B).T)
        s_same = 1.0 / np.sqrt(np.diag(cov_same))
        R_same = (cov_same * s_same[:, None]) * s_same[None, :]
        R_same = 0.5 * (R_same + R_same.T)
        d_same_p = s_same * np.diag(S_fine[li])
        x_same = scipy.linalg.solve(R_same, d_same_p, assume_a='sym',
                                    check_finite=False)
        q_same = float(np.dot(d_same_p, x_same))
        snr2_same += scale * q_same

    # ---------------- r(ell) on fixed grid [100, 40000], CMB-independent ----
    l_fine_r = np.geomspace(100.0, 40000.0, n_ell_fine)
    n_ell_r = len(l_fine_r)
    r_all_of_ell = np.full(n_ell_r, np.nan)
    r_same_of_ell = np.full(n_ell_r, np.nan)

    if not skip_r:
        # Re-interpolate S and C_kappa to the r grid
        S_r = np.zeros((n_ell_r, n_bins, n_bins))
        Ck_r = np.zeros((n_ell_r, n_bins, n_bins))
        for i in range(n_bins):
            for j in range(n_bins):
                S_r[:, i, j] = interp_signed_loglog(l_sig, S_coarse[:, i, j], l_fine_r)
                Ck_r[:, i, j] = interp_signed_loglog(l_cov, C_kappa_coarse[:, i, j], l_fine_r)
        A_r = 0.5 * ((Ck_r + N_kappa[None]) + (Ck_r + N_kappa[None]).transpose(0, 2, 1))

        # kSZ auto-power for r(ell) denominator (file stores ell(ell+1)/(2pi)*C_ell).
        ksz_data = np.load(ksz_path)
        l_ksz = np.asarray(ksz_data['cl_ksz_ell'], float)
        dl_ksz = np.asarray(ksz_data['cl_ksz_total'], float)
        cl_ksz = dl_ksz / (l_ksz * (l_ksz + 1.0) / (2.0 * np.pi))
        log_ksz_interp = scipy.interpolate.interp1d(
            np.log(l_ksz), np.log(cl_ksz),
            kind='linear', bounds_error=False, fill_value=np.nan)
        C_ksz_r = np.exp(log_ksz_interp(np.log(l_fine_r)))

        for li in range(n_ell_r):
            C_ksz_li = C_ksz_r[li]
            if not (np.isfinite(C_ksz_li) and C_ksz_li > 0):
                continue

            A = A_r[li]
            C = np.kron(A, B)
            C = 0.5 * (C + C.T)
            d = S_r[li].reshape(-1)

            s = 1.0 / np.sqrt(np.diag(C))
            R = (C * s[:, None]) * s[None, :]
            R = 0.5 * (R + R.T)
            d_prime = s * d
            x = scipy.linalg.solve(R, d_prime, assume_a='sym', check_finite=False)
            q_all = float(np.dot(d_prime, x))
            r_all_of_ell[li] = np.sqrt(q_all / C_ksz_li)

            cov_same = 0.5 * (A * B + (A * B).T)
            s_same = 1.0 / np.sqrt(np.diag(cov_same))
            R_same = (cov_same * s_same[:, None]) * s_same[None, :]
            R_same = 0.5 * (R_same + R_same.T)
            d_same_p = s_same * np.diag(S_r[li])
            x_same = scipy.linalg.solve(R_same, d_same_p, assume_a='sym',
                                        check_finite=False)
            q_same = float(np.dot(d_same_p, x_same))
            r_same_of_ell[li] = np.sqrt(q_same / C_ksz_li)

    if v in ('summary', 'full'):
        print(f"  width={width} flavor={flavor}: "
              f"worst cond(C)={worst_condC:.3e}, "
              f"worst cond(R)={worst_condR:.3e}")
        # Warn if r > 1 (would indicate covariance inconsistency)
        if np.any(r_all_of_ell[np.isfinite(r_all_of_ell)] > 1.0):
            print(f"    WARNING: r_all(ell) > 1 at some ells; "
                  f"max r_all = {np.nanmax(r_all_of_ell):.4f}")
        if np.any(r_same_of_ell[np.isfinite(r_same_of_ell)] > 1.0):
            print(f"    WARNING: r_same(ell) > 1 at some ells; "
                  f"max r_same = {np.nanmax(r_same_of_ell):.4f}")

    return (float(np.sqrt(max(snr2_same, 0.0))),
            float(np.sqrt(max(snr2_all, 0.0))),
            n_bins,
            worst_condC,
            worst_condR,
            l_fine_r,
            r_same_of_ell,
            r_all_of_ell)