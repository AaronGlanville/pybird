#!/bin/bash
#SBATCH --job-name=dynesty
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=3-0 #3 days
#SBATCH -o Outputs/fit_BOSS_curved_dynesty_%a.out
#SBATCH -e Errors/fit_BOSS_P4_curved_dynesty_%a.err

source activate pybirdz1z2z3

#original config- gives flat fit, with wide error bars/bimodality in h
#python -u fit_BOSS_pybird_updated_chi_2_P4.py ../config/tbird_NGCSGC_curved.ini $SLURM_ARRAY_TASK_ID

#config with smaller/finer grid + hexadecapole
#running with updated grid + hexadecapole gives curved fits
#running again with curved grids, no hex- still gives very curved fits, for both codes
#running while pointing to original grid, no hex- original config, results consistent with flat
#original config + hex- results pretty consistent with flat tbh
#python -u fit_BOSS_pybird_curved_updated_chi2_dynesty.py ../config/tbird_NGCSGC_curved_updated_grid.ini $SLURM_ARRAY_TASK_ID

#Recomputed grid- optiresum=False, with_resum=True, order 3, smaller h, smaller O_b- Worked well!
#python -u fit_BOSS_pybird_updated_chi_2_P4.py ../config/tbird_NGCSGC_with_resum_smaller_h.ini $SLURM_ARRAY_TASK_ID

#with_resum=True, optiresum=False, order 4, smaller h, smaller Om_k- worked well!!
python -u fit_BOSS_pybird_updated_chi_2_P4.py ../config/tbird_NGCSGC_with_resum_4th_order.ini $SLURM_ARRAY_TASK_ID
