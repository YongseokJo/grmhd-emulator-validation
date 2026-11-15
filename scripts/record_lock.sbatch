#!/bin/bash
#SBATCH --job-name=gval-record-lock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gval-record-lock-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
gval_platform=${1:?Usage: sbatch scripts/record_lock.sbatch <platform>}
source "$PWD/scripts/load_environment.sh" "$gval_platform"
exec "$GVAL_PYTHON" -m grmhd_validation.environment "$PWD" "$gval_platform"
