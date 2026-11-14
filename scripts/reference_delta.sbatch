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
# Delta CPU node: sbatch --account=<cpu-account> --partition=cpu scripts/reference_delta.sbatch <split_id>
# DeltaAI review: override account/partition and reserve a GPU if required by Slurm;
# the reference itself always runs on CPU. DeltaAI's base environment is not yet named.
if [[ $(uname -m) == x86_64 ]]; then
    module reset
    module load pytorch-conda/2.8
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
gval_python=${GVAL_PYTHON:-python}
exec "$gval_python" -m grmhd_validation reference persistence --split-id "${1:?Usage: sbatch scripts/reference_delta.sbatch <split_id>}" --split validation --tier quick --device cpu "${@:2}"
