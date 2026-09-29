#!/bin/bash
#SBATCH --job-name=ksz_cov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=4:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
##SBATCH --account=krausee
##SBATCH --partition=standard
##SBATCH --mem=64G
#SBATCH --output=logs/cov_roman_3000_%j.out
##SBATCH --error=logs/cov_%j.err

module purge > /dev/null 2>&1
module load anaconda
conda init bash
source ~/.bashrc
conda activate myenv

export PATH_TO_CLASS_SZ_DATA=/path/to/your/class_sz_data
export OMP_NUM_THREADS=1

mkdir -p logs

# SCRIPT is relative to this directory: submit from sbatch_scripts/ksz_cs_cov_submit/

WIDTH=3000
SURVEY=roman_kl
SCRIPT="../../python_scripts/ksz_cross_cov/run_covariance_all_variants.py"

mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --mca pml ob1 \
      python ${SCRIPT} \
    --width ${WIDTH} \
    --survey ${SURVEY} \
