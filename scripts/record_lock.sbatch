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
# This reader needs only stdlib; it does not select a substitute base environment.
gval_bootstrap_python=${GVAL_BOOTSTRAP_PYTHON:-python3}
gval_base=$("$gval_bootstrap_python" - "$gval_platform" <<'PY'
import json,sys
base=json.load(open('environment/platforms.json')).get(sys.argv[1])
if base is None:
    print(json.dumps({'ok':False,'next':'Ask the owner to name this platform base environment','error':'Base environment is null or platform is unknown'}))
    sys.exit(1)
print(json.dumps(base))
PY
) || { printf '%s\n' "$gval_base"; exit 1; }
if [[ $("$gval_bootstrap_python" -c 'import json,sys; print(str(json.loads(sys.argv[1])["reset_modules"]).lower())' "$gval_base") == true ]]; then
  module reset
fi
mapfile -t gval_modules < <("$gval_bootstrap_python" -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["modules"]))' "$gval_base")
module load "${gval_modules[@]}"
exec python -m grmhd_validation.environment "$PWD" "$gval_platform"
