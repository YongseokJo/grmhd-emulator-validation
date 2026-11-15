import json
from pathlib import Path

import jsonschema


def schema_validate(value, name, repo=None):
    root = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    # Python's json decoder accepts NaN; canonical pipeline documents do not.
    json.dumps(value, allow_nan=False)
    jsonschema.validate(value, json.loads((root / "schemas" / name).read_text()))
