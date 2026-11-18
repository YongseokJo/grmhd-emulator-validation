"""Real Git status gates shell/Codex edits, including changes during evaluation."""
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from test_runner import RunnerFixture
from grmhd_validation.repo_state import inspect, require_clean, RepositoryModifiedError, PROTECTED
from grmhd_validation.report import report


class RepositoryStateTests(RunnerFixture):
    def git(self,*args):
        return subprocess.run(['git','-C',str(self.repo),*args],capture_output=True,text=True,check=True)

    def test_each_protected_path_and_staged_deleted_untracked_changes(self):
        self.assertFalse(inspect(self.repo)['repo_modified'])
        for folder in PROTECTED:
            path=self.repo/folder/'probe.txt';path.parent.mkdir(exist_ok=True);path.write_text('baseline\n')
        self.commit_fixture()
        for folder in PROTECTED:
            path=self.repo/folder/'probe.txt';path.write_text('shell edit\n')
            self.assertTrue(inspect(self.repo)['repo_modified'],folder)
            self.git('restore','--',str(path.relative_to(self.repo)))
        path=self.repo/'schemas/probe.txt';path.write_text('staged\n');self.git('add','schemas/probe.txt')
        self.assertTrue(inspect(self.repo)['repo_modified']);self.git('reset','-q','HEAD','--','schemas/probe.txt');self.git('restore','--','schemas/probe.txt')
        path.unlink();self.assertTrue(inspect(self.repo)['repo_modified']);self.git('restore','--','schemas/probe.txt')
        extra=self.repo/'prompts/new.txt';extra.write_text('untracked');self.assertTrue(inspect(self.repo)['repo_modified']);extra.unlink()
        (self.repo/'request.json').write_text('{}');self.assertFalse(inspect(self.repo)['repo_modified'])

    def test_nongit_repo_fails_closed_and_observed_changes_are_sticky(self):
        self.assertTrue(inspect(self.data)['repo_modified'])
        with patch('grmhd_validation.repo_state.subprocess.run',side_effect=FileNotFoundError('git')):
            self.assertTrue(inspect(self.repo)['repo_modified'])
        state={'repo_modified':True,'repo_status':[' M schemas/earlier.json']}
        with self.assertRaises(RepositoryModifiedError):require_clean(state,self.repo)
        self.assertIn(' M schemas/earlier.json',state['repo_status'])

    def test_dirty_evaluation_records_failure_without_loading_adapter(self):
        (self.repo/'prompts').mkdir();(self.repo/'prompts/probe.txt').write_text('shell edit')
        with patch('grmhd_validation.runner.adapter_source',side_effect=AssertionError('Dirty run must not load model')):
            result=self.run_eval()
        self.assertFalse(result['ok']);path=Path(result['run_dir'])
        info=json.loads((path/'run.json').read_text());self.assertTrue(info['repo_modified'])
        self.assertTrue((path/'FAILED').exists());self.assertFalse((path/'COMPLETE').exists())
        self.assertFalse(list((self.shared/'runs').iterdir()))
        with patch('grmhd_validation.slurm.require_compute'):
            report('synthetic_v1',storage=self.storage(),repo=self.repo)
        payload=json.loads((self.shared/'reports/synthetic_v1/leaderboard.json').read_text())
        self.assertFalse(payload['groups']);self.assertEqual(payload['excluded_attempts'][0]['reason'],'repository modified')
        self.assertTrue(payload['excluded_attempts'][0]['repo_modified'])

    def test_modified_invalid_split_is_recorded_before_parsing(self):
        self.file.write_text('invalid JSON from a shell edit')
        result=self.run_eval();self.assertFalse(result['ok'])
        info=json.loads((Path(result['run_dir'])/'run.json').read_text())
        self.assertTrue(info['repo_modified']);self.assertIn('Protected repository',result['error'])

    def test_change_during_inference_or_after_checksums_prevents_publication(self):
        self.golden();self.conform()
        from grmhd_validation.analyses import single_step
        original=single_step.run
        changed=self.repo/'grmhd_validation/probe.py';changed.parent.mkdir()
        def mutate(ctx):
            result=original(ctx);changed.write_text('# adapter side effect\n');return result
        with patch.object(single_step,'run',side_effect=mutate):result=self.run_eval()
        self.assertFalse(result['ok']);info=json.loads((Path(result['run_dir'])/'run.json').read_text());self.assertTrue(info['repo_modified'])
        changed.unlink()
        from grmhd_validation.runner import bundle_checksums
        def mutate_at_publish(writer,*args,**kwargs):
            bundle_checksums(writer,*args,**kwargs);changed.write_text('# late change\n')
        with patch('grmhd_validation.runner.bundle_checksums',side_effect=mutate_at_publish):result=self.run_eval()
        self.assertFalse(result['ok']);self.assertTrue(json.loads((Path(result['run_dir'])/'run.json').read_text())['repo_modified'])
        self.assertFalse((Path(result['run_dir'])/'COMPLETE').exists());self.assertFalse(list((self.shared/'runs').iterdir()))

    def test_dirty_conformance_cannot_create_a_passing_certificate(self):
        self.golden();(self.repo/'schemas').mkdir();(self.repo/'schemas/shell.json').write_text('{}')
        result=self.conform();self.assertFalse(result['ok'])
        info=json.loads((Path(result['output_dir'])/'run.json').read_text());self.assertTrue(info['repo_modified'])
        self.assertFalse((Path(result['output_dir'])/'COMPLETE').exists())
