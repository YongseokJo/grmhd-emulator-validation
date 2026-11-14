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
# Submit from the repository root with the account/partition appropriate to the machine.
# The selected Python must have this package installed with pip install --no-deps -e .
gval_python=${GVAL_PYTHON:-python}
exec "$gval_python" -m grmhd_validation checksums --split-file "${1:?Usage: sbatch scripts/checksums.sbatch splits/<id>.json}" "${@:2}"
