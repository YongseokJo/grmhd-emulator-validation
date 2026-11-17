"""Format-only correction: bounded scalar summary, unchanged metric bytes."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from test_runner import RunnerFixture
from grmhd_validation.artifacts import json_bytes,result_files,fingerprint
from grmhd_validation.conformance import reference,reference_path,conformance
from grmhd_validation.hashing import sha256
from grmhd_validation.runner import evaluate
from grmhd_validation.summary import from_metrics,validate
from grmhd_validation.summary_correction import rebuild
from grmhd_validation.environment import platform_id
from grmhd_validation.protocol import read_protocol


class CompactSummaryTests(RunnerFixture):
    def setUp(self):
        super().setUp()
        scheduler=patch('grmhd_validation.summary_correction.require_compute');scheduler.start();self.addCleanup(scheduler.stop)

    def test_new_runs_bind_small_summary_to_metric_hashes(self):
        self.golden();self.conform();verdict=self.run_eval();self.assertTrue(verdict['ok'],verdict)
        path=Path(verdict['run_dir']);summary=json.loads((path/'summary.json').read_text());info=json.loads((path/'run.json').read_text())
        self.assertEqual(summary['schema_version'],2);self.assertLess((path/'summary.json').stat().st_size,1_000_000)
        for name,digest in summary['metric_sha256'].items():self.assertEqual(digest,sha256(path/'metrics'/(name+'.json')))
        self.assertEqual(info['result_content_sha256'],result_files(path));self.assertEqual(info['result_fingerprint'],fingerprint(result_files(path)))
        self.assertEqual(set(summary['rollout_curve']['at_steps']),{'1','2','4'})
        self.assertNotIn('steps',summary['rollout_curve'])

    def test_schema_and_byte_guard_reject_maps_profiles_and_oversized_values(self):
        self.golden()
        protocol,digest=read_protocol(self.repo/'protocols/forecast_v1.json')
        path=reference_path(self.storage(),'synthetic_v1',protocol,platform_id(),'quick',digest)
        summary=json.loads((path/'summary.json').read_text())
        invalid=copy.deepcopy(summary);invalid['single_step']['map']=[[1,2],[3,4]]
        with self.assertRaises(Exception):validate(invalid)
        oversized=copy.deepcopy(summary);oversized['single_step']['ci_method']='x'*1_000_001
        with self.assertRaisesRegex(ValueError,'1,000,000'):validate(oversized)

    def test_owner_migration_changes_only_summary_and_is_idempotent(self):
        self.golden();protocol,digest=read_protocol(self.repo/'protocols/forecast_v1.json')
        path=reference_path(self.storage(),'synthetic_v1',protocol,platform_id(),'quick',digest)
        compact=json.loads((path/'summary.json').read_text())
        legacy={k:v for k,v in compact.items() if k not in {'schema_version','metric_sha256'}}
        (path/'summary.json').write_bytes(json_bytes(legacy))
        files=result_files(path);old=fingerprint(files)
        golden=json.loads((path/'golden.json').read_text());golden.update(files=files,result_fingerprint=old);golden.pop('summary_format_version',None)
        (path/'golden.json').write_bytes(json_bytes(golden))
        info=json.loads((path/'run.json').read_text());info.update(result_fingerprint=old);info.pop('summary_format_version',None);info.pop('result_content_sha256',None)
        (path/'run.json').write_bytes(json_bytes(info))
        result=rebuild('synthetic_v1','quick',self.storage(),self.repo)
        self.assertEqual(result['old_fingerprint'],old);self.assertNotEqual(old,result['new_fingerprint'])
        self.assertEqual(result['metrics_before'],result['metrics_after'])
        self.assertEqual({k:v for k,v in files.items() if k!='summary.json'},{k:v for k,v in result_files(path).items() if k!='summary.json'})
        self.assertTrue(rebuild('synthetic_v1','quick',self.storage(),self.repo)['reused'])
        self.assertTrue(self.conform()['ok'])

    def test_coverage_only_diagnostics_do_not_manufacture_confidence_intervals(self):
        self.golden();protocol,digest=read_protocol(self.repo/'protocols/forecast_v1.json')
        path=reference_path(self.storage(),'synthetic_v1',protocol,platform_id(),'quick',digest)
        summary=json.loads((path/'summary.json').read_text())
        from grmhd_validation.summary import COVERAGE,project
        result={'space':'physical','data':{'aggregate':{'mean':{},'lower':{},'upper':{},'n_starts':12}}}
        for name in COVERAGE:
            summary[name]=project(name,result,protocol)
            self.assertEqual(summary[name]['ci95'],{})
        validate(summary)
