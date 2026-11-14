#!/bin/bash
#SBATCH --job-name=gval-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=gval-evaluate-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
# Submit with the A100 partition/account appropriate to your allocation.
module reset
module load pytorch-conda/2.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
gval_python=${GVAL_PYTHON:-python}
exec "$gval_python" -m grmhd_validation evaluate "$@"
