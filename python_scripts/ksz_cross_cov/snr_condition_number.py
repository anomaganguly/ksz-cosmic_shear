"""
snr_condition_number_effect.py  --  minimal SNR via full covariance + correlation transform.

Per ell:
  1. Build full covariance C = A_ell (x) B.
  2. Report cond(C).
  3. Transform to correlation matrix R = D^{-1/2} C D^{-1/2}, D = diag(C).
     Transform signal: d' = D^{-1/2} d.
  4. Report cond(R).
  5. Solve R x = d', SNR^2 contribution = d' . x.

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
    N_kappa = cov_data['N_matrix'] / 2 # because shape noise per component matters!

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
    if not np.any(y_coarse != 0): # if all elements of y_coarse array are zero, it returns 0 array of size l_fine
        return out
    sign = np.sign(y_coarse)
    absy = np.abs(y_coarse)
    pos = absy > 0
    if not np.any(pos):
        return out # if all elements of y_coarse array are zero, it returns 0 array of size l_fine
    floor_val = absy[pos].min() * 1e-5
    absy_safe = np.where(pos, absy, floor_val) # replace zeros with a tiny floor 1.e-10 smaller than the smallest positive value
    log_interp = scipy.interpolate.interp1d(
        np.log(l_coarse), np.log(absy_safe),
        kind='linear', fill_value='extrapolate', bounds_error=False)
    sign_interp = scipy.interpolate.interp1d(
        l_coarse, sign, kind='nearest', fill_value='extrapolate', bounds_error=False)
    return sign_interp(l_fine) * np.exp(log_interp(np.log(l_fine))) # linearly interpolate sign and log interpolate the ell wrt to cov element


class _PerturbableNpz(dict):
    """Mutable stand-in for an NpzFile: supports d[k] and d.files."""
    @property
    def files(self):
        return list(self.keys())


def perturb_components(cov_data, epsrel, rng):
    """
    Return a mutable copy of cov_data with the quad_vec-integrated blocks
    perturbed by symmetric relative Gaussian noise at level `epsrel`.

    Only C_kappa_matrix and the velocity *signal* blocks carry the
    epsrel integration error, so only those are perturbed. The noise blocks
    (N_matrix shape noise, N_vrec_* reconstruction noise) are copied through.
    """
    out = _PerturbableNpz((k, cov_data[k]) for k in cov_data.files)

    def sym_rel(M):
        M = np.asarray(M, float)
        P = epsrel * M * rng.standard_normal(M.shape)
        if M.ndim == 3:        # (n_ell, n_bins, n_bins): symmetrize bin axes
            P = 0.5 * (P + P.transpose(0, 2, 1))
        else:                  # (n_bins, n_bins)
            P = 0.5 * (P + P.T)
        return M + P

    for key in ('C_kappa_matrix', 'C_velocity_signal', 'C_velocity_signal_pz'):
        if key in out:
            out[key] = sym_rel(out[key])
    return out
    
# =============================================================================
# Main
# =============================================================================

def compute_snr_for_width(width, survey, experiment, f_sky, ell_min, ell_max,
                          n_ell_fine, get_cmb_total,
                          flavor='shape_velnoise_pz',
                          results_root=os.path.join(REPO_ROOT, 'results'),
                          cov_name='covariance_components_all_variants.npz',
                          verbose='summary',
                          perturb_epsrel=0.0, perturb_rng=None):   # NEW
    """
    Returns (snr_same, snr_all, n_bins).

    verbose: 'off' / 'summary' / 'full'  (or True == 'full', False == 'off').
    """
    v = {True: 'full', False: 'off'}.get(verbose, verbose)

    base = os.path.join(results_root, 'cross_and_auto_correlations',
                        f'{survey}_all_bins', f'width_{width:g}')
    # load cov data
    cov_data = np.load(os.path.join(base, cov_name))
    if perturb_epsrel > 0:                                         # NEW
        cov_data = perturb_components(                             # NEW
            cov_data, perturb_epsrel,                              # NEW
            perturb_rng if perturb_rng is not None                # NEW
            else np.random.default_rng())                         # NEW
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
        C = np.kron(A, B) # kappa kappa v v
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
        snr2_same += scale * float(np.dot(d_same_p, x_same))

    if v in ('summary', 'full'):
        print(f"  width={width} flavor={flavor}: "
              f"worst cond(C)={worst_condC:.3e}, "
              f"worst cond(R)={worst_condR:.3e}")

    return (float(np.sqrt(max(snr2_same, 0.0))),
            float(np.sqrt(max(snr2_all, 0.0))),
            n_bins,
            worst_condC,       # NEW
            worst_condR)       # NEW
    
def snr_integration_stability(width, survey, experiment, f_sky, ell_min,
                              ell_max, n_ell_fine, get_cmb_total,
                              flavor='shape_velnoise_pz',
                              epsrel=1e-4, n_draws=200, seed=0, **kw):
    """Move the SNR with epsrel-level noise on C_kappa & C_velocity_signal,
    and compare the spread to the per-ell condition numbers."""
    s0, a0, n_bins, condC, condR = compute_snr_for_width(
        width, survey, experiment, f_sky, ell_min, ell_max, n_ell_fine,
        get_cmb_total, flavor=flavor, verbose='off', **kw)

    rng = np.random.default_rng(seed)
    same = np.empty(n_draws); alls = np.empty(n_draws)
    for d in range(n_draws):
        s, a, *_ = compute_snr_for_width(
            width, survey, experiment, f_sky, ell_min, ell_max, n_ell_fine,
            get_cmb_total, flavor=flavor, verbose='off',
            perturb_epsrel=epsrel, perturb_rng=rng, **kw)
        same[d], alls[d] = s, a

    def report(name, base, draws):
        rel = draws.std() / abs(draws.mean()) * 100 if draws.mean() else np.inf
        print(f"  {name:9s}: base={base:.5e}  mean={draws.mean():.5e}  "
              f"spread={rel:.3f}%  [min {draws.min():.5e}, max {draws.max():.5e}]")

    print(f"width={width} flavor={flavor}  epsrel={epsrel}  n_draws={n_draws}")
    print(f"  worst cond(C)={condC:.3e}  worst cond(R)={condR:.3e}")
    report('snr_same', s0, same)
    report('snr_all',  a0, alls)
    return dict(snr_same=s0, snr_all=a0, condC=condC, condR=condR,
                same_draws=same, all_draws=alls)