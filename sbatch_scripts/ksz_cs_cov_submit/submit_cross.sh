#!/bin/bash
#SBATCH --job-name=ksz_cross
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=94
#SBATCH --time=48:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
##SBATCH --account=krausee
##SBATCH --partition=standard
##SBATCH --mem=64G
#SBATCH --output=logs/cross_toy_600_%j.out
##SBATCH --error=logs/cross_%j.err

module purge > /dev/null 2>&1
module load anaconda 
conda init bash
source ~/.bashrc
conda activate myenv

export PATH_TO_CLASS_SZ_DATA=/path/to/your/class_sz_data
export OMP_NUM_THREADS=1

mkdir -p logs

# SCRIPT is relative to this directory: submit from sbatch_scripts/ksz_cs_cov_submit/

WIDTH=600
SURVEY=rubin
SCRIPT="../../python_scripts/ksz_cross_cov/run_cross_correlations_corrected.py"

mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --mca pml ob1 \
      python ${SCRIPT} \
    --width ${WIDTH} \
    --survey ${SURVEY}
