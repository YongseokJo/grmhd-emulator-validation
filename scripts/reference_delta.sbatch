#!/bin/bash
#SBATCH --job-name=gval-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=gval-reference-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
# The reference always runs on CPU/one thread, including on a DeltaAI GPU node.
source "$PWD/scripts/load_environment.sh" auto
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
exec "$GVAL_PYTHON" -m grmhd_validation reference persistence --split-id "${1:?Usage: sbatch scripts/reference_delta.sbatch <split_id>}" --split validation --tier quick --device cpu "${@:2}"
