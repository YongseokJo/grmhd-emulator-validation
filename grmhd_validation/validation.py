import json
from pathlib import Path

import jsonschema


def schema_validate(value, name, repo=None):
    root = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    # Python's json decoder accepts NaN; canonical pipeline documents do not.
    json.dumps(value, allow_nan=False)
    schema=json.loads((root / "schemas" / name).read_text())
    if 'x-maxSerializedBytes' in schema:
        from .artifacts import json_bytes
        if len(json_bytes(value))>schema['x-maxSerializedBytes']:
            raise ValueError('Summary exceeds hard serialized limit of 1,000,000 bytes')
    jsonschema.validate(value,schema)
