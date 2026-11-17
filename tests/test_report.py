"""Report eligibility, frozen snapshot replay and counts use synthetic runs."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from test_runner import RunnerFixture
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.hashing import sha256
from grmhd_validation.report import report,read_results,make_report,artifacts
from grmhd_validation.runner import evaluate


class ReportTests(RunnerFixture):
    def setUp(self):
        super().setUp()
        # Each test uses real synthetic common inference and sealed publication.
        self.golden();self.conform()
        scheduling=patch('grmhd_validation.slurm.require_compute');scheduling.start();self.addCleanup(scheduling.stop)
        self.export=self.base/'committed';self.export.mkdir()
        import shutil
        shutil.copytree(self.repo/'splits',self.export/'splits');shutil.copytree(self.repo/'protocols',self.export/'protocols')

    def run_report(self,**kwargs):
        return report('synthetic_v1',storage=self.storage(),repo=self.repo,**kwargs)

    def test_live_export_and_from_results_are_byte_identical_without_shared_storage(self):
        verdict=self.run_eval();self.assertTrue(verdict['ok'],verdict)
        live=self.run_report(export_results=self.export)
        self.assertEqual(live['runs'],1)
        live_md=(self.shared/'reports/synthetic_v1/leaderboard.md').read_bytes()
        result=report('synthetic_v1',from_results=True,repo=self.export,check=True)
        self.assertTrue(result['checked']);self.assertEqual(live_md,(self.export/'reports/synthetic_v1/leaderboard.md').read_bytes())
        with patch('grmhd_validation.report.SharedStorage',side_effect=AssertionError('Replay cannot consult shared storage')):
            report('synthetic_v1',from_results=True,repo=self.export,check=True)
        path=self.export/'reports/synthetic_v1/leaderboard.md';path.write_text('hand-edited')
        with self.assertRaisesRegex(ValueError,'differs'):report('synthetic_v1',from_results=True,repo=self.export,check=True)

    def test_failed_test_attempts_count_even_when_not_eligible(self):
        self.run_eval()
        from grmhd_validation.runstore import reserve
        root,run_id,attempt=reserve(self.storage(),'persistence','v1',None,'synthetic_v1','test','full')
        root.joinpath('run.json').write_bytes(json_bytes({'model':'persistence','checkpoint_sha256':None,'split_id':'synthetic_v1','split':'test','tier':'full','status':'failed'}))
        (root/'FAILED').write_text('failure')
        self.run_report(export_results=self.export)
        payload=json.loads((self.export/'reports/synthetic_v1/leaderboard.json').read_text())
        self.assertEqual(payload['groups'][0]['rows'][0]['test_attempts'],1)
        self.assertEqual(payload['excluded_attempts'][0]['reason'],'failed')
        report('synthetic_v1',from_results=True,repo=self.export,check=True)

    def test_unsealed_and_not_evaluated_stay_off_leaderboard(self):
        result=self.run_eval();root=Path(result['run_dir']);root.chmod(0o3770)
        from grmhd_validation.runstore import reserve
        other,name,_=reserve(self.storage(),'persistence','v1',None,'synthetic_v1','validation','quick')
        (other/'NOT_EVALUATED').write_text('missing predictions')
        (other/'run.json').write_bytes(json_bytes({'model':'persistence','checkpoint_sha256':None,'split_id':'synthetic_v1','split':'validation','tier':'quick','status':'not evaluated'}))
        self.run_report(export_results=self.export)
        payload=json.loads((self.export/'reports/synthetic_v1/leaderboard.json').read_text())
        self.assertFalse(payload['groups']);self.assertEqual({r['reason'] for r in payload['excluded_attempts']},{'unsealed','not evaluated'})
        report('synthetic_v1',from_results=True,repo=self.export,check=True)

    def test_training_unlock_and_attestation_labels_survive_replay(self):
        result=self.run_eval();root=Path(result['run_dir'])
        # Alter an internal projected row to check presentation without modifying
        # a completed bundle or claiming a new production evaluation.
        from grmhd_validation.report import scan,record_row,markdown
        records,ledger=scan(self.storage(),'synthetic_v1',self.repo)
        info,summary,path=records[0];info=copy.deepcopy(info)
        info.update(environment_locked=False,training_provenance={'label':'no, overlaps validation/test (25 dumps)','evidence':'operator declaration'},predictions={'attestation':'unattested'})
        payload=make_report([(info,summary,path)],ledger,self.repo);row=payload['groups'][0]['rows'][0]
        self.assertEqual(row['flags'],['unlocked','unattested'])
        self.assertEqual(row['training_label'],'no, overlaps validation/test (25 dumps)')
        self.assertIn('no, overlaps validation/test (25 dumps)',markdown(payload).decode())

    def test_corrupt_complete_does_not_replace_existing_report(self):
        result=self.run_eval();self.run_report()
        before=(self.shared/'reports/synthetic_v1/leaderboard.md').read_bytes()
        path=Path(result['run_dir'])/'summary.json';path.chmod(0o640);path.write_text('{}');path.chmod(0o440)
        with self.assertRaisesRegex(ValueError,'Corrupt COMPLETE'):self.run_report()
        self.assertEqual(before,(self.shared/'reports/synthetic_v1/leaderboard.md').read_bytes())

    def test_export_record_tampering_and_ledger_mismatch_fail(self):
        result=self.run_eval();self.run_report(export_results=self.export)
        ledger_path=self.export/'results/synthetic_v1/attempts.json';ledger=json.loads(ledger_path.read_text());ledger['attempts'][0]['eligible']=False;ledger_path.write_bytes(json_bytes(ledger))
        with self.assertRaisesRegex(ValueError,'eligibility'):report('synthetic_v1',from_results=True,repo=self.export,check=True)

    def test_grouping_separates_partition_tier_and_protocol_hash(self):
        result=self.run_eval()
        from grmhd_validation.report import scan
        records,ledger=scan(self.storage(),'synthetic_v1',self.repo);info,summary,path=records[0]
        # A temporary second protocol file/record exercises the actual SHA grouping.
        protocol=copy.deepcopy(self.protocol);protocol['protocol_id']='forecast_other'
        p=self.repo/'protocols/forecast_other.json';p.write_bytes(json_bytes(protocol))
        other=copy.deepcopy(info);other.update(protocol_id='forecast_other',protocol_sha256=sha256(p),run_id='other-protocol')
        test=copy.deepcopy(info);test.update(split='test',run_id='test-partition')
        payload=make_report([records[0],(other,summary,path),(test,summary,path)],ledger,self.repo)
        self.assertEqual(len(payload['groups']),3)

    def test_replay_check_does_not_render_or_repair_shared_artifacts(self):
        self.run_eval();self.run_report(export_results=self.export)
        with patch('grmhd_validation.report.figures',side_effect=AssertionError('CI text checks must not render PNGs')):
            report('synthetic_v1',from_results=True,repo=self.export,check=True)

    def test_export_symlinks_are_rejected(self):
        self.run_eval();self.run_report(export_results=self.export)
        (self.export/'results/synthetic_v1/other-model').symlink_to(self.data)
        with self.assertRaisesRegex(ValueError,'symlink'):report('synthetic_v1',from_results=True,repo=self.export,check=True)
