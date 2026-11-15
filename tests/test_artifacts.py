"""Serialization, schema, publication and provenance integrity checks."""
import copy
import ctypes
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from grmhd_validation.artifacts import csv_bytes,fingerprint,json_bytes,result_files
from grmhd_validation.runstore import no_replace_rename
from grmhd_validation.validation import schema_validate

REPO=Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_fixed_serialization_and_fingerprint_scope(self):
        payload={'z':1.2345678901234567,'a':None}
        self.assertEqual(json_bytes(payload),b'{\n "a": null,\n "z": 1.2345678901234567\n}\n')
        self.assertEqual(csv_bytes([{'a':1.2345678901234567,'b':None}],['a','b']),b'a,b\n1.2345678901234567,\n')
        with self.assertRaises(ValueError):
            json_bytes({'bad':float('nan')})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'summary.json').write_bytes(json_bytes(payload))
            (root/'metrics').mkdir();(root/'tables').mkdir()
            (root/'metrics/single_step.json').write_bytes(json_bytes(payload))
            (root/'tables/single_step.csv').write_bytes(b'a,b\n1,2\n')
            files=result_files(root);before=fingerprint(files)
            self.assertEqual(len(files),3)
            for name in ('run.json','out_res.txt','checksums.json','COMPLETE'):
                (root/name).write_text('metadata differs\n')
            self.assertEqual(fingerprint(result_files(root)),before)
            (root/'metrics/single_step.json').write_bytes(json_bytes({'z':2,'a':None}))
            self.assertNotEqual(fingerprint(result_files(root)),before)
            (root/'metrics/single_step.json').unlink()
            (root/'metrics/single_step.json').symlink_to(root/'summary.json')
            with self.assertRaisesRegex(ValueError,'symlink'):
                result_files(root)

    def test_atomic_rename_never_overwrites_even_empty_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'source';b=root/'target'
            a.mkdir();b.mkdir();(a/'value').write_text('preserve')
            with self.assertRaises(FileExistsError):
                no_replace_rename(a,b)
            self.assertEqual((a/'value').read_text(),'preserve')
            self.assertFalse(list(b.iterdir()))

    def test_shared_filesystem_fallback_serializes_writers_and_preserves_empty_target(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        def unsupported(*args):
            ctypes.set_errno(errno.EINVAL)
            return -1
        with tempfile.TemporaryDirectory() as temporary, \
                patch('grmhd_validation.runstore.ctypes.CDLL',return_value=SimpleNamespace(renameat2=unsupported)):
            root=Path(temporary);target=root/'target'
            left,right=root/'left',root/'right';left.mkdir();right.mkdir()
            inodes={p:p.stat().st_ino for p in (left,right)}
            def publish(source):
                try:
                    no_replace_rename(source,target)
                    return source
                except FileExistsError:
                    return None
            with ThreadPoolExecutor(max_workers=2) as executor:
                winners=[p for p in executor.map(publish,(left,right)) if p is not None]
            self.assertEqual(len(winners),1)
            self.assertEqual(target.stat().st_ino,inodes[winners[0]])
            loser=right if winners[0]==left else left
            self.assertTrue(loser.is_dir())
            with self.assertRaises(FileExistsError):
                no_replace_rename(loser,target)
            self.assertEqual(target.stat().st_ino,inodes[winners[0]])

    def test_real_rename_errors_do_not_trigger_fallback(self):
        from types import SimpleNamespace
        def denied(*args):
            ctypes.set_errno(errno.EACCES)
            return -1
        with tempfile.TemporaryDirectory() as temporary, \
                patch('grmhd_validation.runstore.ctypes.CDLL',return_value=SimpleNamespace(renameat2=denied)):
            root=Path(temporary);left=root/'left';left.mkdir()
            with patch('grmhd_validation.runstore.os.rename',side_effect=AssertionError('Do not mask access errors')):
                with self.assertRaises(PermissionError):
                    no_replace_rename(left,root/'target')
            self.assertTrue(left.is_dir())

    def test_all_schemas_valid_and_request_closed(self):
        for path in (REPO/'schemas').glob('*.json'):
            jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))
        from grmhd_validation.runner import persistence_request
        request=persistence_request();schema_validate(request,'request.schema.json')
        for key in request:
            changed=copy.deepcopy(request);del changed[key]
            with self.assertRaises(jsonschema.ValidationError):
                schema_validate(changed,'request.schema.json')
        request['protocol']={'max_steps':1}
        with self.assertRaises(jsonschema.ValidationError):
            schema_validate(request,'request.schema.json')

    def test_ports_have_attribution_and_provenance_entries(self):
        provenance=(REPO/'docs/provenance.md').read_text()
        for name in ('common.py','single_step.py','rollout_curve.py'):
            module=REPO/'grmhd_validation/analyses'/name
            header=module.read_text().split('"""')[1]
            for required in ('Kevin Fortune','Sachin','ebcc218','main','Fall26/'):
                self.assertIn(required,header)
            self.assertIn('grmhd_validation/analyses/'+name,provenance)


if __name__=='__main__':
    unittest.main()

class TemplateTests(unittest.TestCase):
    def test_template_is_valid_and_existing_files_are_preserved(self):
        from grmhd_validation.templates import new_adapter
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'model'
            result=new_adapter(target)
            self.assertTrue(result['ok'])
            request=json.loads((target/'run_request.json').read_text())
            schema_validate(request,'request.schema.json')
            compile((target/'adapter.py').read_text(),str(target/'adapter.py'),'exec')
            before=(target/'adapter.py').read_bytes()
            with self.assertRaises(FileExistsError):
                new_adapter(target)
            self.assertEqual((target/'adapter.py').read_bytes(),before)
