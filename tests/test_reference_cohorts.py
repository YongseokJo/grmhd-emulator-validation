"""Quick content stays invariant when full semantics acquire a new hash cohort."""
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.conformance import reference
from test_runner import RunnerFixture
from unittest.mock import patch


class ReferenceCohortTests(RunnerFixture):
    def test_identical_content_hashes_pass_without_numeric_reparsing(self):
        from grmhd_validation.conformance import compare_results
        roots=[self.base/'first-result',self.base/'second-result']
        for root in roots:
            root.mkdir();(root/'summary.json').write_bytes(b'{"value": 1.25}\n')
        with patch('grmhd_validation.conformance.numeric_difference',side_effect=AssertionError('Identical hashes need no numerical traversal')):
            self.assertEqual(compare_results(*roots),{'ok':True,'differing_files':[],'largest_relative_difference':0.})

    def test_final_quick_reference_cannot_accept_changed_legacy_content(self):
        path=self.repo/'protocols/forecast_v1.json'
        self.protocol['constants']['full']['production_ready']=False
        path.write_bytes(json_bytes(self.protocol));legacy=self.golden()
        self.protocol['constants']['full']['production_ready']=True
        path.write_bytes(json_bytes(self.protocol));final=self.golden()
        self.assertEqual(legacy['result_fingerprint'],final['result_fingerprint'])
        self.assertNotEqual(legacy['reference_dir'],final['reference_dir'])
        self.protocol['constants']['loss_space']['floor_rho']=1000.
        path.write_bytes(json_bytes(self.protocol))
        with self.assertRaisesRegex(ValueError,'Quick result bytes changed'):
            reference('synthetic_v1',storage=self.storage(),repo=self.repo)
        self.assertEqual(list((self.shared/'references').rglob('.candidate-*')),[])
