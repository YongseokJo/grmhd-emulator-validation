"""Acceptance checks for the owner-approved milestone-4 production changes."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_full_analyses import FullFixture
from grmhd_validation.comparison import comparison_key,report_eligible
from grmhd_validation.verify import verify_run


class Milestone4Tests(FullFixture):
    def test_full_preview_never_publishes_or_becomes_report_eligible(self):
        self.golden();self.conform()
        # Remains a gate regression even after the final protocol lifts it.
        with patch('grmhd_validation.runner.full_tier_ready',return_value=False):
            first=self.run_eval(tier='full');second=self.run_eval(tier='full')
        self.assertTrue(first['ok'],first);self.assertTrue(first['preview'])
        self.assertTrue(first['run_id'].endswith('__a1'))
        self.assertTrue(second['run_id'].endswith('__a2'))
        path=Path(first['run_dir'])
        self.assertIn(self.shared/'staging',path.parents)
        self.assertTrue((path/'PREVIEW').is_file());self.assertFalse((path/'COMPLETE').exists())
        self.assertFalse(list((self.shared/'runs').iterdir()))
        info=json.loads((path/'run.json').read_text())
        self.assertEqual(info['status'],'preview');self.assertTrue(info['preview_reason'])
        self.assertTrue(verify_run(path,self.repo)['preview'])
        self.assertFalse(report_eligible(path,info))
        self.assertFalse(path.stat().st_mode & 0o222)

    def test_comparison_groups_include_protocol_hash(self):
        info=dict(split_id='same',protocol_id='forecast_v1',protocol_sha256='a'*64,tier='full')
        self.assertNotEqual(comparison_key(info),comparison_key({**info,'protocol_sha256':'b'*64}))
        self.assertEqual(len(comparison_key(info)),4)

    def test_singleton_phi_mass_profiles_and_signed_log_errors(self):
        from grmhd_validation.analyses import shell_profiles,physics_table
        ctx=self.context();nphi=ctx.split.shape[-1]
        ctx.geometry=replace(ctx.geometry,gdet=ctx.geometry.gdet[...,:1],
                             gcov=ctx.geometry.gcov[...,:1],gcon=ctx.geometry.gcon[...,:1])
        truth=np.zeros(ctx.split.shape,dtype=np.float32);truth[0]=1.;truth[1]=.25;truth[5]=1.
        class TwiceDensity:
            name='mass-test';version='v1';history_length=1;max_batch=1
            def step(self,history):
                result=history[:,-1].clone();result[:,0]=2.;return result
        ctx.adapter=TwiceDensity();start=ctx.starts[0].rollout[0]
        with patch.object(ctx.reader,'numpy',side_effect=lambda _:truth.copy()):
            physical=shell_profiles.compute(ctx,space='physical',starts=[start])['per_start'][0]['values']
            legacy=shell_profiles.compute(ctx,space='loss',starts=[start])['per_start'][0]['values']
            physical_table=physics_table.compute(ctx,space='physical',starts=[start])['per_start'][0]['values']
            legacy_table=physics_table.compute(ctx,space='loss',starts=[start])['per_start'][0]['values']
        for row in physical['samples']:
            np.testing.assert_allclose(row['mass_pred'],2.,rtol=1e-14)
            np.testing.assert_allclose(row['mass_truth'],1.,rtol=1e-14)
            np.testing.assert_allclose(row['mass_error'],np.log(3)-np.log(2),rtol=1e-14)
        for row in legacy['samples']:
            np.testing.assert_allclose(row['mass_pred'],2*nphi,rtol=1e-6)
            np.testing.assert_allclose(row['mass_truth'],nphi,rtol=1e-6)
        np.testing.assert_allclose(physical_table['samples'][0]['mass_error'],np.log(3)-np.log(2),rtol=1e-14)
        np.testing.assert_allclose(legacy_table['samples'][0]['mass_error'],np.log(1+2*nphi)-np.log(1+nphi),rtol=1e-6)
