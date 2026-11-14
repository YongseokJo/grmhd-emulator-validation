import contextlib
import copy
import grp
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema
import numpy as np

from grmhd_validation.checksums import checksum_bytes, generate, read_checksums, verify_file
from grmhd_validation.cli import main
from grmhd_validation.doctor import doctor, storage_checks
from grmhd_validation.environment import check_environment, prepare_base, satisfies_minimum
from grmhd_validation.hashing import sha256
from grmhd_validation.permissions import GROUP_AREAS, OWNER_AREAS, SharedStorage
from grmhd_validation.protocol import read_protocol
from grmhd_validation.slurm import require_compute
from grmhd_validation.splits import CHANNELS, SPLIT_NAMES, read_split

REPO = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        (self.data / "blind").mkdir(parents=True)
        np.savez(self.data / "geometry.npz", r=np.ones((2, 2, 2)))
        self.spec = {"schema_version": 2, "split_id": "synthetic_v1", "dataset": "synthetic",
                     "data_root": str(self.data), "geometry_file": "geometry.npz",
                     "geometry_sha256": sha256(self.data / "geometry.npz"), "times_file": None,
                     "array": {"shape": [8, 2, 2, 2], "dtype": "float32", "channels": list(CHANNELS),
                               "axes": ["channel", "r", "theta", "phi"]}, "dt_dump": 25., "spin": .94,
                     "total_dumps": 13, "segments": [], "splits": {s: {"count": 0, "chunks": [], "files": []} for s in SPLIT_NAMES}}
        for split, name, first, last in [("train", "train_1", 0, 2), ("excluded", "excluded_1", 3, 3),
                                         ("validation", "validation_1", 4, 7), ("excluded", "excluded_2", 8, 8),
                                         ("test", "test_1", 9, 12)]:
            chunk = {"chunk": name, "first": first, "last": last, "count": last - first + 1}
            self.spec["segments"].append({**chunk, "split": split})
            section = self.spec["splits"][split]
            section["chunks"].append(chunk)
            for i in range(first, last + 1):
                filename = ("blind/" if i == 12 else "") + f"dump_{i}.npy"
                section["files"].append(filename)
                np.save(self.data / filename, np.full((8, 2, 2, 2), i, dtype=np.float32))
            section["count"] += chunk["count"]
        self.file = self.root / "synthetic_v1.json"
        self.write_spec()
        self.shared = self.root / "shared"
        self.shared.mkdir(mode=0o2750)
        os.chmod(self.shared, 0o2750)
        for name in OWNER_AREAS + GROUP_AREAS:
            path = self.shared / name
            path.mkdir()
            os.chmod(path, 0o2750 if name in OWNER_AREAS else 0o3770)
        self.group = grp.getgrgid(os.getegid()).gr_name
        self.storage = SharedStorage(self.shared, self.group)
        self.old_umask = os.umask(0o007)
        self.addCleanup(os.umask, self.old_umask)

    def write_spec(self):
        self.file.write_text(json.dumps(self.spec, sort_keys=True))


class SplitTests(Fixture):
    def test_frozen_files_and_identity_under_override(self):
        for path in (REPO / "splits").glob("*_v1.json"):
            split = read_split(path)
            self.assertEqual(split.sha256, sha256(path))
            self.assertEqual(len(split.chunks("validation")), 3)
            self.assertEqual(len(split.chunks("test")), 3)
            self.assertEqual(len(split.dumps(("excluded",))), 120)
            overridden = read_split(path, self.data)
            self.assertEqual(overridden.sha256, split.sha256)
            self.assertEqual(overridden.data_root, self.data.resolve())

    def test_numeric_order_exact_paths_and_no_data_listing(self):
        with patch.object(Path, "iterdir", side_effect=AssertionError("Do not discover data")):
            split = read_split(self.file)
            dumps = split.dumps(SPLIT_NAMES)
            self.assertEqual([d.index for d in dumps], list(range(13)))
            self.assertEqual(split.dump_path(dumps[-1]), self.data / "blind/dump_12.npy")

    def test_model_view_never_exposes_excluded(self):
        split = read_split(self.file)
        excluded = {d.index for d in split.dumps(("excluded",))}
        for name in ("train", "validation", "test"):
            self.assertFalse(excluded & {d.index for d in split.model_dumps(name)})
        with self.assertRaisesRegex(ValueError, "Excluded"):
            split.model_dumps("excluded")

    def test_invalid_coverage_paths_counts_and_schema_fail(self):
        original = copy.deepcopy(self.spec)
        mutations = [lambda d: d.update(schema_version=1),
                     lambda d: d["splits"]["train"].update(count=99),
                     lambda d: d["segments"][1].update(first=4),
                     lambda d: d["splits"]["test"]["files"].__setitem__(-1, "../dump_12.npy"),
                     lambda d: d["splits"]["test"]["files"].__setitem__(-1, "dump_0.npy"),
                     lambda d: d["array"].update(dtype="float64")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.spec = copy.deepcopy(original); mutate(self.spec); self.write_spec()
                with self.assertRaises(ValueError):
                    read_split(self.file)


class ChecksumTests(Fixture):
    def test_all_splits_geometry_numeric_order_and_determinism(self):
        split = read_split(self.file)
        one, three = checksum_bytes(split, 1), checksum_bytes(split, 3)
        self.assertEqual(one, three)
        output = self.root / "checksums.sha256"
        output.write_bytes(one)
        hashes = read_checksums(output)
        self.assertEqual(len(hashes), 14)
        self.assertEqual(list(hashes)[:4], ["dump_0.npy", "dump_1.npy", "dump_2.npy", "dump_3.npy"])
        self.assertEqual(list(hashes)[-1], "geometry.npz")
        self.assertIn("dump_8.npy", hashes)
        verify_file(self.data / "blind/dump_12.npy", "blind/dump_12.npy", hashes)
        np.save(self.data / "blind/dump_12.npy", np.zeros((8, 2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_file(self.data / "blind/dump_12.npy", "blind/dump_12.npy", hashes)
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_file(self.data / "dump_0.npy", "missing.npy", hashes)

    def test_geometry_mismatch_and_duplicate_manifest_rejected(self):
        split = read_split(self.file)
        (self.data / "geometry.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Geometry"):
            checksum_bytes(split, 1)
        output = self.root / "duplicate.sha256"
        output.write_text(("a" * 64 + "  dump_0.npy\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_checksums(output)

    def test_failed_publish_removes_only_its_temporary_file(self):
        with patch("grmhd_validation.checksums.require_compute"), patch("grmhd_validation.checksums.os.link", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                generate(self.file, storage=self.storage)
        self.assertEqual(list((self.shared / "checksums").iterdir()), [])

    def test_publish_is_group_readable_idempotent_and_refuses_changed_data(self):
        with patch("grmhd_validation.checksums.require_compute"):
            result = generate(self.file, storage=self.storage)
            self.assertEqual(result["files"], 14)
            output = Path(result["output"])
            before = output.stat()
            self.assertTrue(before.st_mode & stat.S_IRGRP)
            self.assertEqual(before.st_gid, os.getegid())
            again = generate(self.file, storage=self.storage)
            self.assertEqual(again["sha256"], result["sha256"])
            self.assertEqual(output.stat().st_ino, before.st_ino)
            np.save(self.data / "dump_3.npy", np.ones((8, 2, 2, 2)))
            with self.assertRaises(FileExistsError):
                generate(self.file, storage=self.storage)
            self.assertEqual(output.stat().st_ino, before.st_ino)


class PermissionTests(Fixture):
    def test_modes_lazy_creation_and_no_chmod_existing(self):
        with patch("grmhd_validation.permissions.getpass.getuser", return_value="alice"):
            user = self.storage.staging_user()
        destination = self.storage.destination("synthetic_v1", "persistence")
        self.assertEqual(stat.S_IMODE(user.stat().st_mode), 0o3770)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o3770)
        self.assertEqual(destination.stat().st_gid, os.getegid())
        with patch("grmhd_validation.permissions.os.fchmod", side_effect=AssertionError("Existing chmod")):
            self.assertEqual(self.storage.destination("synthetic_v1", "persistence"), destination)
        owner = self.storage.folder("references", "synthetic_v1")
        self.assertEqual(stat.S_IMODE(owner.stat().st_mode), 0o2750)
        self.assertFalse((self.shared / "training_records/synthetic_v1").exists())

    def test_readonly_and_foreign_entries(self):
        run = self.storage.folder("staging", "alice", "run-a1")
        with self.storage.new_file(run, "summary.json") as stream:
            stream.write(b"{}")
        self.storage.make_readonly(run)
        self.assertFalse((run / "summary.json").stat().st_mode & 0o222)
        self.assertFalse(run.stat().st_mode & 0o222)
        # Restore only test-created paths for TemporaryDirectory teardown.
        self.addCleanup(os.chmod, run, 0o3770)
        other = self.storage.folder("staging", "alice", "run-a2")
        (other / "foreign.json").write_text("{}")
        with patch("grmhd_validation.permissions.os.fchmod", side_effect=AssertionError("Foreign chmod")):
            with self.assertRaisesRegex(PermissionError, "not created"):
                self.storage.make_readonly(other)

    def test_existing_bad_modes_and_symlink_are_not_repaired(self):
        path = self.shared / "staging/alice"
        path.mkdir(mode=0o700)
        with patch("grmhd_validation.permissions.os.fchmod", side_effect=AssertionError("Repair forbidden")):
            with self.assertRaises(PermissionError):
                self.storage.folder("staging", "alice")
        target = self.root / "elsewhere"; target.mkdir()
        (self.shared / "runs/escape").symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            self.storage.folder("runs", "escape")

    def test_nonowner_checksums_denied(self):
        with patch("grmhd_validation.permissions.os.geteuid", return_value=os.getuid() + 1):
            with self.assertRaises(PermissionError):
                self.storage.folder("checksums")


class EnvironmentTests(Fixture):
    def current(self):
        return {"schema_version": 1, "platform": "deltaai-aarch64", "python": "3.12.1", "torch": "2.3.0",
                "cuda": "12.1", "packages": {"torch": "2.3.0", "numpy": "1.24.4", "scipy": "1.10.1", "matplotlib": "3.7.1", "jsonschema": "4.18.0"},
                "pip_freeze": ["jsonschema==4.18.0"], "torch_error": None}

    def test_unlocked_is_warning_and_core_mismatch_fails(self):
        (self.root / "environment").mkdir()
        current = self.current()
        with patch("grmhd_validation.environment.platform_id", return_value="deltaai-aarch64"):
            unlocked = check_environment(self.root)
            self.assertEqual(unlocked["status"], "warn")
            self.assertFalse(unlocked["environment_locked"])
            path = self.root / "environment/deltaai-aarch64.lock"
            path.write_text(json.dumps(current))
            self.assertEqual(check_environment(self.root, current)["status"], "pass")
            for field in ("python", "torch", "cuda", "numpy", "scipy", "matplotlib"):
                changed = copy.deepcopy(current)
                (changed if field in ("python", "torch", "cuda") else changed["packages"])[field] = "different"
                self.assertEqual(check_environment(self.root, changed)["status"], "fail", field)
            changed = copy.deepcopy(current); changed["pip_freeze"] += ["other==1.0"]
            self.assertEqual(check_environment(self.root, changed)["status"], "warn")
        jsonschema.validate(current, json.loads((REPO / "schemas/environment_lock.schema.json").read_text()))

    def test_minimum_versions_include_numpy2_and_torch_local_versions(self):
        self.assertTrue(satisfies_minimum(np.__version__, "1.24"))
        self.assertTrue(satisfies_minimum("2.3.0+cu121", "2.3"))
        self.assertFalse(satisfies_minimum("2.3.0rc1", "2.3"))
        self.assertFalse(satisfies_minimum("1.23.5", "1.24"))

    def test_base_core_packages_never_offered_to_pip(self):
        versions = {"numpy": "1.23.5", "scipy": "1.10.1", "torch": "2.3.0", "matplotlib": "3.7.1", "jsonschema": "4.18.0"}
        with patch("grmhd_validation.environment.importlib.metadata.version", side_effect=versions.__getitem__), patch("grmhd_validation.environment.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "below approved"):
                prepare_base(REPO)
            run.assert_not_called()

    def test_install_plan_cannot_replace_module_packages(self):
        import importlib.metadata
        versions = {"numpy": "1.24.4", "scipy": "1.10.1", "torch": "2.3.0", "matplotlib": "3.7.1"}
        def version(name):
            if name == "jsonschema":
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        def dry_run(args, **kwargs):
            report = Path(args[args.index("--report") + 1])
            report.write_text(json.dumps({"install": [{"metadata": {"name": "numpy", "version": "2.0.0"}}]}))
            return subprocess.CompletedProcess(args, 0)
        with patch("grmhd_validation.environment.importlib.metadata.version", side_effect=version), patch("grmhd_validation.environment.subprocess.run", side_effect=dry_run) as run:
            with self.assertRaisesRegex(RuntimeError, "would change"):
                prepare_base(REPO)
            self.assertEqual(run.call_count, 1)


class CommandTests(Fixture):
    def test_json_errors_help_and_umask(self):
        for args in (["unknown"], ["checksums"], ["checksums", "--split-file", str(self.file)]):
            with self.subTest(args=args), patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(args), 1)
                verdict = json.loads(out.getvalue().splitlines()[-1])
                jsonschema.validate(verdict, json.loads((REPO / "schemas/verdict.schema.json").read_text()))
                self.assertFalse(verdict["ok"])
        previous = os.umask(0o077)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["unknown"])
            self.assertEqual(os.umask(0o007), 0o007)
        finally:
            os.umask(previous)
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            main(["--help"])
        self.assertTrue(json.loads(out.getvalue().splitlines()[-1])["ok"])

    def test_slurm_id_alone_does_not_allow_login_hashing(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_JOB_NODELIST": "compute001"}, clear=True), patch("grmhd_validation.slurm.socket.gethostname", return_value="login001"), patch("grmhd_validation.slurm.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "compute001\n", "")):
            with self.assertRaisesRegex(RuntimeError, "not an allocated"):
                require_compute()

    def test_protocol_exact_approved_settings_and_schema(self):
        protocol, digest = read_protocol(REPO / "protocols/forecast_v1.json")
        expected = {"H_max": 10, "max_steps": 50, "report_steps": [1, 10, 25, 50], "n_starts_per_chunk": 4,
                    "bootstrap_block": 10, "bootstrap_resamples": 1000, "seed": 42, "gpu_tolerance": 1e-4}
        self.assertEqual({key: protocol[key] for key in expected}, expected)
        self.assertEqual(protocol["tiers"]["quick"], ["single_step", "rollout_curve"])
        self.assertEqual(len(protocol["tiers"]["full"]), 25)
        self.assertNotIn("error_spectra", protocol["tiers"]["full"])
        jsonschema.validate(protocol, json.loads((REPO / "schemas/protocol.schema.json").read_text()))

    def test_approved_dependency_bounds_match_package_metadata(self):
        lines = {line for line in (REPO / "environment/requirements.in").read_text().splitlines() if line and not line.startswith("#")}
        self.assertEqual(lines, {"numpy>=1.24", "scipy>=1.10", "matplotlib>=3.7", "torch>=2.3", "jsonschema>=4.18"})
        project = (REPO / "pyproject.toml").read_text()
        runtime = re.search(r"dependencies\s*=\s*\[([^\]]*)\]", project, re.S)[1]
        self.assertEqual(set(re.findall(r'"([a-z]+>=[0-9.]+)"', runtime)), lines)
        self.assertIn('requires-python = ">=3.10"', project)

    def test_doctor_succeeds_with_unlocked_environment_and_creates_nothing(self):
        (self.root / "protocols").mkdir(); (self.root / "splits").mkdir()
        shutil.copyfile(REPO / "protocols/forecast_v1.json", self.root / "protocols/forecast_v1.json")
        shutil.copyfile(self.file, self.root / "splits/synthetic_v1.json")
        frozen = self.shared / "splits/synthetic_v1.json"
        shutil.copyfile(self.file, frozen); frozen.chmod(0o444)
        before = sorted(str(p) for p in self.shared.rglob("*"))
        with patch("grmhd_validation.doctor.storage_checks", return_value={"status": "pass"}):
            result = doctor(self.root, self.shared, self.group)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "warn")
        self.assertFalse(result["environment_locked"])
        self.assertEqual(before, sorted(str(p) for p in self.shared.rglob("*")))
        jsonschema.validate(result, json.loads((REPO / "schemas/verdict.schema.json").read_text()))

    @unittest.skipUnless(shutil.which("getfacl"), "ACL inspection requires getfacl")
    def test_doctor_readonly_and_effective_acl_reports_unreadable_file(self):
        path = self.shared / "reports/private.json"; path.write_text("{}"); path.chmod(0o600)
        before = {str(p): (p.stat().st_mtime_ns, stat.S_IMODE(p.stat().st_mode)) for p in self.shared.rglob("*")}
        with patch("grmhd_validation.doctor.os.chmod", side_effect=AssertionError("Doctor chmod")), patch("grmhd_validation.doctor.os.mkdir", side_effect=AssertionError("Doctor mkdir")):
            report = storage_checks(self.shared, self.group)
        self.assertIn(str(path), report["group_unreadable_files"])
        after = {str(p): (p.stat().st_mtime_ns, stat.S_IMODE(p.stat().st_mode)) for p in self.shared.rglob("*")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
