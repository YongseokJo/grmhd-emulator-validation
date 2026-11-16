"""Test receipt control flow with stub commands; no scientific data are read."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/validate_milestone4.py'
EXPECTED={'rc300_reduced_v1':'6d8ecd5d35dcd85c89796e25fddb53bffcc55611f26f833c2168da0fcb815ec6',
          '5000run_v1':'cd9598fd3f8380f9ebc862db46eb42d070fcfc7f43e532e7bd0f9fc7c51cf648'}


class ValidationDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'protocols').mkdir()
        (self.root/'protocols/forecast_v1.json').write_text('{}\n')
        self.previous=Path.cwd();os.chdir(self.root);self.addCleanup(os.chdir,self.previous)

    def run_driver(self,arguments):
        def command(argv,**kwargs):
            split=argv[argv.index('--split-id')+1]
            return SimpleNamespace(returncode=0,stdout=json.dumps({'ok':True,'result_fingerprint':EXPECTED[split]})+'\n',stderr='')
        with patch('sys.argv',[str(SCRIPT),*arguments]),patch.dict(os.environ,{name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),patch('grmhd_validation.slurm.require_compute'),patch('subprocess.run',side_effect=command) as calls,contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(SCRIPT),run_name='__main__')
        return json.loads((self.root/'gval-m4-real-receipt.log').read_text()),calls.call_count

    def test_default_chain_runs_all_operations_for_both_datasets(self):
        receipt,count=self.run_driver([])
        self.assertTrue(receipt['ok']);self.assertEqual(count,6);self.assertEqual(len(receipt['steps']),6)

    def test_selected_operations_append_until_both_dataset_chains_are_complete(self):
        first,count=self.run_driver(['--split-id','rc300_reduced_v1'])
        self.assertFalse(first['ok']);self.assertEqual(count,3)
        second,count=self.run_driver(['--split-id','5000run_v1','--operation','reference-full','--operation','reference-quick'])
        self.assertFalse(second['ok']);self.assertEqual(count,2);self.assertEqual(len(second['steps']),5)
        last,count=self.run_driver(['--split-id','5000run_v1','--operation','conformance'])
        self.assertTrue(last['ok']);self.assertEqual(count,1);self.assertEqual(len(last['steps']),6)
