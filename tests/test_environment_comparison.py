"""Pure-stdlib environment comparison checks; no scientific imports needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grmhd_validation.environment import check_environment


class EditableComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def current(self):
        return {"schema_version":1,"platform":"deltaai-aarch64","python":"3.12.9",
                "torch":"2.15.0+cu132","cuda":"13.2","torch_error":None,
                "packages":{"numpy":"2.5.4","scipy":"1.18.1","matplotlib":"3.11.2"},
                "pip_freeze":["jsonschema==4.26.0"]}

    def test_only_own_editable_revision_is_excluded_from_comparison(self):
        (self.root/'environment').mkdir()
        expected=self.current()
        expected['pip_freeze']+=['-e git+ssh://git@github.com/owner/repo.git@old#egg=grmhd_emulator_validation']
        path=self.root/'environment/deltaai-aarch64.lock';path.write_text(json.dumps(expected))
        changed=copy.deepcopy(expected)
        with patch('grmhd_validation.environment.platform_id',return_value='deltaai-aarch64'):
            for editable in ('-e git+https://example.org/repo.git@new#egg=grmhd-emulator-validation',
                             '--editable '+str(self.root)):
                changed['pip_freeze']=[*expected['pip_freeze'][:-1],editable]
                self.assertEqual(check_environment(self.root,changed)['status'],'pass')
            changed['pip_freeze']+=['-e git+https://example.org/other.git@new#egg=other-package']
            self.assertEqual(check_environment(self.root,changed)['status'],'warn')
            changed['packages']['numpy']='different'
            self.assertEqual(check_environment(self.root,changed)['status'],'fail')
        self.assertEqual(json.loads(path.read_text()),expected)

