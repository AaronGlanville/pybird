#!/bin/bash
#SBATCH --job-name=dynesty
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=1-0 #1 day
#SBATCH -o Outputs/nelder_mead_%a.out
#SBATCH -e Errors/nelder_mead_%a.err

source activate pybirdz1z2z3

#with_resum=True, optiresum=False, order 4, smaller h, smaller Om_k
python -u fit_BOSS_pybird_updated_chi_2_P4_nelder_mead_array.py ../config/tbird_NGCSGC_nelder_mead_with_resum_4th_order.ini $SLURM_ARRAY_TASK_ID

#testing best-fit with the mean of the mocks
#python -u fit_BOSS_pybird_updated_chi_2_P4_nelder_mead_array.py ../config/tbird_NGCSGC_with_resum_4th_order.ini $SLURM_ARRAY_TASK_ID
