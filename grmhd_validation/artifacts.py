"""Canonical result serialization and invocation-owned artifact writes."""
import csv
import hashlib
import io
import json
from pathlib import Path

from .hashing import sha256


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: repr(value) if isinstance(value, float) else value for key, value in row.items()})
    return stream.getvalue().encode()


def result_files(root):
    root = Path(root)
    paths = [root / "summary.json", *sorted((root / "metrics").glob("*.json")), *sorted((root / "tables").glob("*.csv"))]
    if not paths[0].is_file():
        raise ValueError("Result summary is missing")
    if any(path.is_symlink() for path in paths) or any((root / name).is_symlink() for name in ("metrics", "tables")):
        raise ValueError("Result content may not contain symlinks")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def fingerprint(files):
    return hashlib.sha256("".join(f"{files[path]}  {path}\n" for path in sorted(files)).encode()).hexdigest()


class Writer:
    def __init__(self, storage, root):
        self.storage, self.root = storage, Path(root)

    def folder(self, relative):
        path = self.root / relative
        parts = path.relative_to(self.storage.root).parts
        return self.storage.folder(*parts)

    def open(self, relative):
        path = self.root / relative
        self.folder(path.parent.relative_to(self.root))
        if path in self.storage.created:
            return self.storage.rewrite_created(path)
        return self.storage.new_file(path.parent, path.name)

    def write(self, relative, content):
        with self.open(relative) as stream:
            stream.write(content)

    def json(self, relative, value):
        self.write(relative, json_bytes(value))
