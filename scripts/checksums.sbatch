#!/bin/bash
#SBATCH --job-name=gval-checksums
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=gval-checksums-%j.log
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?}"
source "$PWD/scripts/load_environment.sh" auto
exec "$GVAL_PYTHON" -m grmhd_validation checksums --split-file "${1:?Usage: sbatch scripts/checksums.sbatch splits/<id>.json}" "${@:2}"
