"""Record real environments and compare them to the current platform's lock."""
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .hashing import sha256

CORE_PACKAGES = {"torch", "numpy", "scipy", "matplotlib"}


def comparison_freeze(entries, repo):
    """Ignore only this project's editable source; run.json records its commit."""
    result = []
    for entry in entries:
        editable = entry.removeprefix("-e ").removeprefix("--editable ")
        if editable != entry:
            egg = parse_qs(urlsplit(editable).fragment).get("egg", [""])[0].split("[", 1)[0]
            name = re.sub(r"[-_.]+", "-", egg).lower()
            if name == "grmhd-emulator-validation":
                continue
            if not urlsplit(editable).scheme and Path(editable).expanduser().resolve() == Path(repo).resolve():
                continue
        result.append(entry)
    return sorted(result)


def satisfies_minimum(version, minimum):
    """Compare the release/minimum forms used by the approved module packages."""
    pattern = r"v?(\d+)\.(\d+)(?:\.(\d+))?(.*)"
    actual, required = re.fullmatch(pattern, version), re.fullmatch(pattern, minimum)
    if not actual or not required:
        raise ValueError("Unsupported version spelling; owner must check it: " + version)
    release = tuple(int(actual[i] or 0) for i in (1, 2, 3))
    floor = tuple(int(required[i] or 0) for i in (1, 2, 3))
    return release > floor or (release == floor and not re.match(r"(?:a|b|rc|dev|\.dev)", actual[4]))


def minimums(repo):
    result = {}
    for line in (Path(repo) / "environment/requirements.in").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-z]+)>=(\d+\.\d+)", line)
        if not match:
            raise ValueError("Expected approved lower bounds only in requirements.in")
        result[match[1]] = match[2]
    return result


def prepare_base(repo):
    """Add only missing non-core dependencies; never offer core packages to pip."""
    import tempfile
    bounds = minimums(repo)
    if sys.version_info < (3, 10):
        raise RuntimeError("The base environment requires Python >=3.10")
    before = {}
    missing = []
    for name, floor in bounds.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in CORE_PACKAGES:
                raise RuntimeError("Approved module is missing core package: " + name)
            missing.append(name + ">=" + floor)
            continue
        if not satisfies_minimum(version, floor):
            raise RuntimeError(f"Installed {name} {version} is below approved minimum {floor}; pip must not replace it")
        if name in CORE_PACKAGES:
            before[name] = version
    if missing:
        with tempfile.TemporaryDirectory(prefix="gval-install-") as temporary:
            report = Path(temporary) / "plan.json"
            subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report), *missing], check=True)
            plan = json.loads(report.read_text())["install"]
            if any(item["metadata"]["name"].lower().replace("_", "-") in CORE_PACKAGES for item in plan):
                raise RuntimeError("pip would change a module package; refusing the installation plan")
            for item in plan:
                name = item["metadata"]["name"]
                try:
                    importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    continue
                raise RuntimeError("pip would change an installed base-environment package: " + name)
            packages = [item["metadata"]["name"] + "==" + item["metadata"]["version"] for item in plan]
            if packages:
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *packages], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo)], check=True)
    if any(importlib.metadata.version(name) != version for name, version in before.items()):
        raise RuntimeError("A module package changed during installation; do not record this environment")


def platform_id():
    machine = platform.machine().lower()
    names = {"x86_64": "delta-x86_64", "aarch64": "deltaai-aarch64"}
    if machine not in names:
        raise ValueError("Unsupported platform architecture: " + machine)
    return names[machine]


def snapshot():
    versions = {}
    for name in ("torch", "numpy", "scipy", "matplotlib", "jsonschema"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cuda, torch_error = None, None
    if versions["torch"]:
        try:
            import torch
            cuda = torch.version.cuda
        except (ImportError, OSError, RuntimeError) as error:
            torch_error = str(error)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True)
    if frozen.returncode:
        raise RuntimeError("pip freeze failed: " + frozen.stderr.strip())
    return {"schema_version": 1, "platform": platform_id(), "python": platform.python_version(),
            "torch": versions["torch"], "cuda": cuda, "packages": versions,
            "pip_freeze": sorted(line for line in frozen.stdout.splitlines() if line and not line.startswith("#")),
            "torch_error": torch_error}


def check_environment(repo, current=None):
    name = platform_id()
    path = Path(repo) / "environment" / (name + ".lock")
    if not path.is_file():
        return {"status": "warn", "platform": name, "environment_locked": False, "lock_sha256": None,
                "mismatches": [], "message": "No recorded lock for this platform; owner must run scripts/record_lock.sbatch " + name}
    expected = json.loads(path.read_text())
    current = snapshot() if current is None else current
    if expected["schema_version"] != 1 or expected["platform"] != name:
        raise ValueError("Environment lock schema/platform mismatch")
    mismatches = []
    for key in ("python", "torch", "cuda"):
        if expected[key] != current[key]:
            mismatches.append({"field": key, "expected": expected[key], "actual": current[key], "status": "fail"})
    for name in ("numpy", "scipy", "matplotlib"):
        if expected["packages"][name] != current["packages"][name]:
            mismatches.append({"field": name, "expected": expected["packages"][name], "actual": current["packages"][name], "status": "fail"})
    if current.get("torch_error"):
        mismatches.append({"field": "torch_import", "status": "fail", "actual": current["torch_error"]})
    expected_freeze = comparison_freeze(expected["pip_freeze"], repo)
    current_freeze = comparison_freeze(current["pip_freeze"], repo)
    if expected_freeze != current_freeze:
        mismatches.append({"field": "other_packages", "status": "warn",
                           "missing_or_changed": sorted(set(expected_freeze) - set(current_freeze)),
                           "added_or_changed": sorted(set(current_freeze) - set(expected_freeze))})
    return {"status": "fail" if any(m["status"] == "fail" for m in mismatches) else "warn" if mismatches else "pass",
            "platform": platform_id(), "environment_locked": True, "lock_sha256": sha256(path), "mismatches": mismatches}


def record_lock(repo, name):
    from .slurm import require_compute
    require_compute()
    if name != platform_id():
        raise ValueError("Requested platform does not match this machine")
    from .permissions import SharedStorage
    SharedStorage().require_owner()
    base = json.loads((Path(repo) / "environment/platforms.json").read_text())[name]
    if base is None:
        raise ValueError("Owner has not named this platform's base environment")
    current = snapshot()
    if current["torch_error"] or any(current["packages"][p] is None for p in current["packages"]):
        raise RuntimeError("Cannot record a lock from an incomplete or broken environment")
    output = Path(repo) / "environment" / (name + ".lock")
    output.write_text(json.dumps(current, sort_keys=True, indent=2) + "\n")
    return {"ok": True, "next": "gval doctor", "lock": str(output), "lock_sha256": sha256(output)}


if __name__ == "__main__":
    os.umask(0o007)
    try:
        from .slurm import require_compute
        require_compute()
        if sys.argv[2] != platform_id():
            raise ValueError("Requested platform does not match this machine")
        from .permissions import SharedStorage
        SharedStorage().require_owner()
        base = json.loads((Path(sys.argv[1]) / "environment/platforms.json").read_text())[sys.argv[2]]
        if base is None:
            raise ValueError("Owner has not named this platform's base environment")
        prepare_base(sys.argv[1])
        print(json.dumps(record_lock(sys.argv[1], sys.argv[2]), sort_keys=True))
    except Exception as error:
        print(json.dumps({"ok": False, "next": "Ask the owner to select the approved base environment", "error": str(error)}))
        sys.exit(1)
