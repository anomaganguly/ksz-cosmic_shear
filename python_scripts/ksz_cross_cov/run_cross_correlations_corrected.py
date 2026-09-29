"""
kSZ Cross-Correlation Calculator
=================================

Computes C_ℓ^{ij,kSZ} for all density bin (i) × velocity bin (j) combinations.
Uses MPI for parallelization across bin pairs.

Usage:
    mpirun -n <nprocs> python run_cross_correlations.py --width <Mpc> --survey <rubin|roman>
"""

import os
import sys
import gc
import numpy as np
from itertools import product

from ksz_common_with_kappa_nulling_convolve import (
    setup_mpi, get_default_params, setup_cosmology_interpolators,
    compute_growth_factor, compute_velocity_growth_factor,
    load_ksz_kernel, load_kernel_data,
    get_input_output_dirs, build_argparser, get_sigma_z_over_1plusz,
    MultiBinPowerSpectrumCalculator
)


def main():
    comm, rank, size = setup_mpi()

    parser = build_argparser("Calculate kSZ-lensing cross-correlations for all bin combinations.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    try:
        path_to_class_sz = os.environ['PATH_TO_CLASS_SZ_DATA'] + '/class_sz/class-sz/'
    except KeyError:
        if rank == 0:
            print("ERROR: Environment variable 'PATH_TO_CLASS_SZ_DATA' is not set.")
        sys.exit(1)

    input_kernel_dir, output_dir = get_input_output_dirs(args.survey, args.width)

    if not os.path.exists(input_kernel_dir):
        if rank == 0:
            print(f"ERROR: Input directory not found: {input_kernel_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cosmo_params, common_params, ksz_params, Pk_matter_params = get_default_params()

    if rank == 0:
        print("--- Cross-Correlation Calculation ---")
        print(f"Survey: {args.survey}, Kernel width: {args.width} Mpc")

    # -------------------------------------------------------------------------
    # Cosmology
    # -------------------------------------------------------------------------
    z_grid, chi_grid, fz_chi, fchi_z = setup_cosmology_interpolators()

    if rank == 0:
        print("Computing growth factor...")
    Dz_interp_chi = compute_growth_factor(
        z_grid, chi_grid, common_params, cosmo_params, ksz_params, Pk_matter_params
    )

    if rank == 0:
        print("Computing velocity growth factor...")
    Dv_interp_chi = compute_velocity_growth_factor(
        z_grid, chi_grid, common_params, cosmo_params, ksz_params, Pk_matter_params
    )

    # -------------------------------------------------------------------------
    # Load kernels
    # -------------------------------------------------------------------------
    ksz_kernel_interp, chi_ksz_peak = load_ksz_kernel(fchi_z)
    kernel_data_list = load_kernel_data(input_kernel_dir, fchi_z)
    n_kernels = len(kernel_data_list)

    if rank == 0:
        print(f"Loaded {n_kernels} kernels, kSZ kernel ready.")

    # Photo-z scatter sigma_z/(1+z) for the chosen survey.
    sigma_z_over_1plusz = get_sigma_z_over_1plusz(args.survey)

    # NOTE: the kSZ kernel FT is now built per (density, velocity) bin pair
    # inside calculate_hatT_T_full_cross, so we pass the raw interpolator
    # (plus Dv and the survey photo-z scatter) rather than a prebuilt KSZKernelFT.
    calculator = MultiBinPowerSpectrumCalculator(
        cosmo_params, common_params, ksz_params, Pk_matter_params,
        kernel_data_list, ksz_kernel_interp, Dz_interp_chi, Dv_interp_chi,
        fchi_z, fz_chi, sigma_z_over_1plusz
    )

    # -------------------------------------------------------------------------
    # Distribute bin combinations across MPI ranks
    # -------------------------------------------------------------------------
    l_vals = np.geomspace(100, 40000, num=40)
    zmin, zmax = 1e-6, 5.0

    all_combinations = list(product(range(n_kernels), range(n_kernels)))
    my_combinations = np.array_split(all_combinations, size)[rank]

    if rank == 0:
        print(f"Total bin combinations: {len(all_combinations)}")

    # -------------------------------------------------------------------------
    # Compute cross-correlations
    # -------------------------------------------------------------------------
    results = {}

    for density_bin, velocity_bin in my_combinations:
        print(f"[Rank {rank}] Computing (density={density_bin}, velocity={velocity_bin})")
        try:
            cross = calculator.calculate_hatT_T_full_cross(
                l_vals, density_bin, velocity_bin, chi_ksz_peak, zmin, zmax
            )
            results[(density_bin, velocity_bin)] = {
                'l_vals': l_vals,
                # 'C_l_cross' = photo-z case (unchanged key, so existing
                # downstream readers keep working); '_nopz' is the new S=1 case.
                'C_l_cross': cross['C_l_photoz'],
                'C_l_cross_nopz': cross['C_l_nopz'],
                'z_density': kernel_data_list[density_bin]['z_nulled'],
                'z_velocity': kernel_data_list[velocity_bin]['z_nulled'],
            }
            # Clear caches to free memory
            calculator._class_instance_cache.clear()
            calculator._results_cache.clear()
            calculator._C_vv_interp_cache.clear()
            gc.collect()
        except Exception as e:
            print(f"[Rank {rank}] Error for ({density_bin}, {velocity_bin}): {e}")
            continue

    # -------------------------------------------------------------------------
    # Gather and save
    # -------------------------------------------------------------------------
    comm.Barrier()
    all_results = comm.gather(results, root=0)

    if rank == 0:
        merged_results = {}
        for r in all_results:
            merged_results.update(r)

        for (i, j), data in merged_results.items():
            np.savez(
                os.path.join(output_dir, f'cross_correlation_i{i:03d}_j{j:03d}.npz'),
                **data
            )

        C_l_matrix = np.zeros((len(l_vals), n_kernels, n_kernels), dtype=complex)
        C_l_matrix_nopz = np.zeros((len(l_vals), n_kernels, n_kernels), dtype=complex)
        for (i, j), data in merged_results.items():
            C_l_matrix[:, i, j] = data['C_l_cross']            # photo-z S
            C_l_matrix_nopz[:, i, j] = data['C_l_cross_nopz']  # S = 1

        np.savez(
            os.path.join(output_dir, 'all_cross_correlations.npz'),
            l_vals=l_vals,
            C_l_matrix=C_l_matrix,            # photo-z (unchanged key)
            C_l_matrix_nopz=C_l_matrix_nopz,  # S = 1 (new)
            z_bins=np.array([kd['z_nulled'] for kd in kernel_data_list])
        )

        print(f"Saved all {len(merged_results)} cross-correlations to {output_dir}")

    comm.Barrier()
    if rank == 0:
        print("\n=== Cross-correlation calculations completed ===")


if __name__ == '__main__':
    main()