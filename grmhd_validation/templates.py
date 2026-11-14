"""Create the two user-written adapter files outside shared storage."""
import json
from pathlib import Path

from .permissions import SHARED_ROOT

TEMPLATE = '''from grmhd_validation import ModelAdapter, Geometry


class Adapter(ModelAdapter):
    name = "my-model"
    version = "v1"
    history_length = 1
    max_batch = 1
    requires_checkpoint = True

    def __init__(self, checkpoint: str, geometry: Geometry, device: str, options: dict):
        # Restore weights and training-only preprocessing here. Store your model
        # on device. Do not discover or read any simulation dumps.
        raise NotImplementedError("Implement model loading")

    def step(self, history):
        # Physical float32 [B,h,8,Nr,Ntheta,Nphi] on device. Return physical
        # float32 [B,8,Nr,Ntheta,Nphi]. The runner maintains rollout history.
        raise NotImplementedError("Implement one forecast step")
'''


def new_adapter(directory):
    root = Path(directory).expanduser().resolve()
    if root.is_relative_to(SHARED_ROOT.resolve()):
        raise ValueError("new-adapter writes only into the user's directory, outside shared storage")
    root.mkdir(parents=True, exist_ok=True)
    files = (root / "adapter.py", root / "run_request.json")
    if any(path.exists() or path.is_symlink() for path in files):
        raise FileExistsError("Refusing to overwrite adapter.py or run_request.json")
    request = {"model": "my-model", "version": "v1", "adapter": str(files[0]) + ":Adapter",
               "checkpoint": "/absolute/path/to/checkpoint.pt", "options": {}, "training_record": None, "notes": ""}
    for path, text in zip(files, (TEMPLATE, json.dumps(request, sort_keys=True, indent=1) + "\n")):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return {"ok": True, "next": "Implement " + str(files[0]) + "; then run gval check-adapter",
            "adapter": str(files[0]), "request": str(files[1])}
