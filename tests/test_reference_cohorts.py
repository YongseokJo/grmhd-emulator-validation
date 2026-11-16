"""Quick content stays invariant when full semantics acquire a new hash cohort."""
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.conformance import reference
from test_runner import RunnerFixture


class ReferenceCohortTests(RunnerFixture):
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
