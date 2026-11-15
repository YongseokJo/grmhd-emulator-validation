import grp
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grmhd_validation.doctor import ACLReader, acl, storage_checks
from grmhd_validation.permissions import GROUP_AREAS, OWNER_AREAS


@unittest.skipUnless(shutil.which("getfacl") and shutil.which("setfacl"), "requires POSIX ACL tools")
class DoctorScalingTests(unittest.TestCase):
    def test_unfinished_reference_candidates_are_reported_without_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);root.chmod(0o2750)
            group=grp.getgrgid(os.getegid()).gr_name
            for name in OWNER_AREAS + GROUP_AREAS:
                path=root/name;path.mkdir();path.chmod(0o2750 if name in OWNER_AREAS else 0o3770)
            for path in [root,*(root/n for n in OWNER_AREAS+GROUP_AREAS)]:
                rights='rwx' if path.name in GROUP_AREAS and path!=root else 'r-x'
                subprocess.run(['setfacl','-m',f'g::{rights},m::{rights},d:u::rwx,d:g::{rights},d:m::{rights},d:o::---',str(path)],check=True)
            candidate=root/'references/synthetic_v1/forecast_v1/persistence_quick/.candidate-cancelled'
            candidate.mkdir(parents=True)
            (candidate/'run.json').write_text('{}\n')
            before={p:(p.stat().st_mode,p.stat().st_mtime_ns) for p in root.rglob('*')}
            result=storage_checks(root,group)
            self.assertEqual(result['status'],'warn',result)
            self.assertEqual(result['unfinished_reference_candidates'],[str(candidate)])
            self.assertEqual(before,{p:(p.stat().st_mode,p.stat().st_mtime_ns) for p in root.rglob('*')})

    def test_thousand_run_tree_and_shared_acl_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o2750)
            group = grp.getgrgid(os.getegid()).gr_name
            for name in OWNER_AREAS + GROUP_AREAS:
                path = root / name
                path.mkdir()
                os.chmod(path, 0o2750 if name in OWNER_AREAS else 0o3770)
            for path in [root, *(root / n for n in OWNER_AREAS + GROUP_AREAS)]:
                rights = "rwx" if path.name in GROUP_AREAS and path != root else "r-x"
                subprocess.run(["setfacl", "-m", f"g::{rights},m::{rights},d:u::rwx,d:g::{rights},d:m::{rights},d:o::---", str(path)], check=True)
            parent = root / "runs/synthetic/model"
            parent.mkdir(parents=True, mode=0o3770)
            for number in range(1000):
                run = parent / f"run-a{number}"
                run.mkdir(mode=0o3770)
                for filename in ("run.json", "summary.json", "checksums.json", "COMPLETE",
                                 "single_step.json", "rollout_curve.json", "scores.csv", "evaluate.log"):
                    path = run / filename
                    with path.open("w") as stream:
                        stream.write("{}\n")
                    path.chmod(0o640)
            unreadable = parent / "run-a999/summary.json"
            # The mask/mode says readable, but the owning group has no read grant.
            subprocess.run(["setfacl", "-m", f"g::---,u:{os.getuid()+1}:r--,m::r--", str(unreadable)], check=True)
            self.assertTrue(unreadable.stat().st_mode & stat.S_IRGRP)
            with patch("grmhd_validation.doctor.acl", wraps=acl) as calls:
                result = storage_checks(root, group)
            self.assertEqual(result["group_unreadable_files"], [str(unreadable)])
            self.assertGreaterEqual(result["scanned_entries"], 9000)
            self.assertLess(result["scan_duration_seconds"], 30)
            self.assertLess(calls.call_count, 12)
            self.assertEqual(result["acl_subprocesses"], calls.call_count)
            print("doctor synthetic scan:", {key: result[key] for key in
                  ("scanned_entries", "scan_duration_seconds", "acl_subprocesses")})

    def test_plain_mode_permissions_do_not_spawn_getfacl(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plain"
            path.write_text("plain\n")
            path.chmod(0o640)
            subprocess.run(["setfacl", "-b", str(path)], check=True)
            reader = ACLReader()
            with patch("grmhd_validation.doctor.acl", side_effect=AssertionError("plain mode needs no subprocess")):
                result = reader.read(path, os.stat(path, follow_symlinks=False))
            self.assertEqual(result["group:"], "r--")
            self.assertEqual(reader.subprocesses, 0)


if __name__ == "__main__":
    unittest.main()
