#!/bin/bash
#SBATCH --job-name=gval-ci
#SBATCH --account=bgun-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=gval-ci-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
source "$PWD/scripts/load_environment.sh" deltaai-aarch64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$GVAL_PYTHON" -m unittest discover -s tests -v
if [[ -n ${GVAL_MINIMUM_PYTHON:-} ]]; then
    "$GVAL_MINIMUM_PYTHON" -m unittest discover -s tests -v
fi
"$GVAL_PYTHON" tests/regression_tron.py --kind both
for gval_split in rc300_reduced_v1 5000run_v1; do
    "$GVAL_PYTHON" -m grmhd_validation conformance --split-id "$gval_split"
done
"$GVAL_PYTHON" -m grmhd_validation doctor
