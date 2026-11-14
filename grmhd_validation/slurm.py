import os
import socket
import subprocess


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Heavy work requires a Slurm compute job; submit the supplied .sbatch script")
    result = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                            text=True, capture_output=True, check=True)
    allocated = {name.split(".")[0] for name in result.stdout.splitlines()}
    if socket.gethostname().split(".")[0] not in allocated:
        raise RuntimeError("This is not an allocated compute node; enter the allocation with srun")
