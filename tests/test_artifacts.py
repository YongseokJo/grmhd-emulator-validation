"""Serialization, schema, publication and provenance integrity checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

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
