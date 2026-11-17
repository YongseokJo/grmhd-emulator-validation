#!/bin/bash
#SBATCH --job-name=gval-report
#SBATCH --account=bgun-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gval-report-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/load_environment.sh deltaai-aarch64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$GVAL_PYTHON" -m grmhd_validation report "$@"
