#!/bin/bash
#SBATCH --job-name=dynesty
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=3-0 #3 days
#SBATCH -o Outputs/fit_BOSS_updated_chi2_dynesty_flat_%a.out
#SBATCH -e Errors/fit_BOSS_updated_chi2_dynesty_flat_%a.err

source activate pybirdz1z2z3

#original config- gives flat fit, with wide error bars/bimodality in h
#python -u fit_BOSS_pybird_updated_chi_2_P4.py ../config/tbird_NGCSGC_curved.ini $SLURM_ARRAY_TASK_ID

#with_resum=True, optiresum=False, order 4, smaller h, smaller Om_k- worked well!!
python -u fit_BOSS_pybird_updated_chi_2_P4_flat.py ../config/tbird_NGCSGC_with_resum_4th_order.ini $SLURM_ARRAY_TASK_ID
