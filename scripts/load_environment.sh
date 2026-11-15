#!/bin/bash
# Source from a job after cd to the repository. GVAL_VENV selects a personal venv.
umask 007
gval_bootstrap_python=${GVAL_BOOTSTRAP_PYTHON:-python3}
gval_platform=${1:-auto}
if [[ "$gval_platform" == auto ]]; then
    gval_platform=$("$gval_bootstrap_python" -c 'from grmhd_validation.environment import platform_id; print(platform_id())')
fi
gval_supported=$("$gval_bootstrap_python" - "$gval_platform" <<'PY'
import json,sys
base=json.load(open('environment/platforms.json')).get(sys.argv[1])
if base is None:
    print(json.dumps({'ok':False,'next':'Ask the owner to name this platform base environment','error':'Base environment is null or platform is unknown'}))
    sys.exit(1)
if sys.argv[1] != 'deltaai-aarch64' or base['reset_modules']:
    raise ValueError('Unsupported platform recipe; consult the owner')
if base['activation'] != ['source "$(conda info --base)/etc/profile.d/conda.sh"', 'conda activate base']:
    raise ValueError('Unsupported activation recipe; consult the owner')
if base['venv']['create'] != ['python','-m','venv','--system-site-packages','<venv>']:
    raise ValueError('Unsupported venv recipe; consult the owner')
print(json.dumps(base))
PY
) || { printf '%s\n' "$gval_supported"; return 1; }
# Do not purge/reset: the module extensions need the default cudatoolkit.
mapfile -t gval_modules < <("$gval_bootstrap_python" -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["modules"]))' "$gval_supported")
module load "${gval_modules[@]}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
gval_base_python="$(conda info --base)/bin/python"
"$gval_base_python" -c 'from grmhd_validation.slurm import require_compute; require_compute()'
gval_base_prefix=$("$gval_base_python" -c 'import sys; print(sys.prefix)')
GVAL_VENV=${GVAL_VENV:-${HOME}/.cache/grmhd-validation/envs/${gval_platform}}
export GVAL_VENV
if [[ ! -e "$GVAL_VENV" ]]; then
    "$gval_base_python" -m venv --system-site-packages "$GVAL_VENV"
fi
source "$GVAL_VENV/bin/activate"
python - "$gval_base_prefix" <<'PY'
import pathlib,sys
configuration=(pathlib.Path(sys.prefix)/'pyvenv.cfg').read_text().lower()
if sys.prefix == sys.base_prefix or 'include-system-site-packages = true' not in configuration:
    raise RuntimeError('GVAL_VENV must be a --system-site-packages venv')
if pathlib.Path(sys.base_prefix).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise RuntimeError('GVAL_VENV was made from a different base interpreter')
from grmhd_validation.environment import prepare_base
prepare_base(pathlib.Path.cwd())
PY
GVAL_PYTHON="$VIRTUAL_ENV/bin/python"
export GVAL_PYTHON
