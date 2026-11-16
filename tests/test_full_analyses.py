"""Synthetic full-tier acceptance tests; real-data regressions are separate.

The geometry spans disk and funnel masks. Excluded dumps stay poisonous, so a
port that accidentally reads across a chunk boundary fails immediately.
"""
import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_runner import RunnerFixture
from grmhd_validation.artifacts import Writer, json_bytes, result_files
from grmhd_validation.adapter import Geometry
from grmhd_validation.checksums import checksum_bytes
from grmhd_validation.channel_scales import generate, read_channel_scales
from grmhd_validation.data import FrameReader, load_geometry
from grmhd_validation.hashing import sha256
from grmhd_validation.runner import Context, initialize
from grmhd_validation.adapters.persistence import Persistence
from grmhd_validation.splits import read_split
from grmhd_validation.windows import select_starts
from grmhd_validation.analyses._analysis_io import FullStore, bind
from grmhd_validation.analyses._port_manifest import ANALYSES


class FullFixture(RunnerFixture):
    def setUp(self):
        super().setUp()
        self.protocol.update(max_steps=8, report_steps=[1,2,4,8])
        (self.repo/'protocols/forecast_v1.json').write_bytes(json_bytes(self.protocol))
        nr,nt,nph=16,9,8
        r,t,p=np.meshgrid(np.geomspace(2,160,nr),np.linspace(.1,np.pi-.1,nt),
                          np.arange(nph)*2*np.pi/nph,indexing='ij')
        metric=np.broadcast_to(np.diag([-1.,1.,1.,1.])[:,:,None,None,None],(4,4,nr,nt,nph)).copy()
        np.savez(self.data/'geometry.npz',r=r,h=t,ph=p,gcov=metric,gcon=metric,
                 gdet=np.ones_like(r),resolved=r>=6.11,dx=np.array([.3,.4,.5]),
                 spin=np.array(.94),dt_dump=np.array(25.),eos_gamma=np.array(5./3.))
        self.spec['array']['shape']=[8,nr,nt,nph]
        self.spec['geometry_sha256']=sha256(self.data/'geometry.npz')
        for name,section in self.spec['splits'].items():
            for filename in section['files']:
                index=int(Path(filename).stem.split('_')[-1])
                phase=.07*index
                frame=np.stack((1+.2*np.sin(np.log(r)+p+phase),
                    .1+.02*np.cos(t+p-phase), -.1+.02*np.sin(p+phase),
                    .03*np.cos(t+phase)+np.zeros_like(r), .01*r**-.5+np.zeros_like(p),
                    2+.1*np.sin(p+t+phase), .2*np.cos(p+phase)+np.zeros_like(r),
                    .3*np.sin(t+p-phase)))
                if name=='excluded':frame.fill(np.nan)
                np.save(self.data/filename,frame.astype(np.float32))
        self.file.write_bytes(json_bytes(self.spec))
        (self.shared/'splits/synthetic_v1.json').write_bytes(self.file.read_bytes())
        self.split=read_split(self.file)
        (self.shared/'checksums/synthetic_v1.sha256').write_bytes(checksum_bytes(self.split,1))
        with patch('grmhd_validation.channel_scales.require_compute'):
            generate('synthetic_v1',storage=self.storage(),repo=self.repo)

    def context(self):
        storage=self.storage()
        root=storage.folder('staging','synthetic-core')
        writer=Writer(storage,root)
        device=torch.device('cpu');initialize(self.protocol,device)
        reader=FrameReader(self.split,'validation',self.shared/'checksums/synthetic_v1.sha256')
        geometry=load_geometry(self.split,device,reader.hashes)
        adapter=Persistence(None,geometry,'cpu',{})
        ctx=Context(adapter,geometry,reader,select_starts(self.split,'validation',self.protocol),
                    self.protocol,self.split,'validation',device,writer)
        ctx.channel_scales=read_channel_scales(self.shared/'references/synthetic_v1/channel_scales.json',self.split)
        ctx.run_info={};ctx.full=FullStore(ctx)
        return ctx


class FullAnalysisTests(FullFixture):
    def test_every_full_analysis_emits_schema_valid_json_and_csv(self):
        ctx=self.context()
        for name in [*ANALYSES,'correlation_times']:
            with self.subTest(analysis=name):
                result=importlib.import_module('grmhd_validation.analyses.'+name).run(ctx)
                self.assertEqual(result['analysis'],name)
                self.assertTrue((ctx.writer.root/'metrics'/(name+'.json')).is_file())
                self.assertTrue((ctx.writer.root/'tables'/(name+'.csv')).is_file())
                if name=='azimuthal_spectra':
                    spectra=result['data']['per_start'][0]['values']['truth']
                    self.assertTrue(np.isfinite(next(iter(next(iter(spectra.values())).values()))).all())
                if name=='mean_profiles':
                    self.assertTrue(np.isfinite(result['data']['per_start'][0]['values']['omega_truth']).all())
        ctx.full.cleanup()
        self.assertFalse((ctx.writer.root/'.scratch').exists())

    def test_full_runner_publishes_and_repeat_content_is_identical(self):
        self.golden();self.conform()
        # Publication lifecycle remains tested while unfinished production is gated.
        with patch('grmhd_validation.runner.full_tier_ready',return_value=True):
            first=self.run_eval(tier='full');self.assertTrue(first['ok'],first)
            second=self.run_eval(tier='full');self.assertTrue(second['ok'],second)
        self.assertEqual(result_files(first['run_dir']),result_files(second['run_dir']))
        info=json.loads((Path(first['run_dir'])/'run.json').read_text())
        self.assertIn('cost',info['analysis_runtime'])
        content=(Path(first['run_dir'])/'metrics/cost.json').read_text()
        self.assertNotIn('milliseconds',content)

    def test_spaces_and_explicit_start_selection_are_independent(self):
        from grmhd_validation.analyses import stability, spatial_error
        ctx=self.context();starts=[group.rollout[0] for group in ctx.starts]
        single=stability.compute(ctx,space='physical',starts=starts[:1])
        multiple=stability.compute(ctx,space='physical',starts=starts)
        self.assertEqual(single['per_start'][0],multiple['per_start'][0])
        self.assertEqual([row['start'] for row in multiple['per_start']],starts)
        for row in multiple['per_start']:
            np.testing.assert_allclose(row['values']['mean'],1.,rtol=1e-5)
        physical=spatial_error.compute(ctx,space='physical',starts=starts[:1])
        legacy=spatial_error.compute(ctx,space='loss',starts=starts[:1])
        self.assertNotEqual(json_bytes(__import__('grmhd_validation.analyses.common',fromlist=['clean']).clean(physical)),
                            json_bytes(__import__('grmhd_validation.analyses.common',fromlist=['clean']).clean(legacy)))

    def test_training_scales_read_only_train_and_are_pinned_to_split(self):
        path=self.shared/'references/synthetic_v1/channel_scales.json'
        scales=read_channel_scales(path,self.split)
        frames=np.stack([np.load(self.split.dump_path(d),allow_pickle=False).astype(np.float64)
                         for d in self.split.model_dumps('train')])
        np.testing.assert_allclose(scales['mean_abs'],np.mean(np.abs(frames),axis=(0,2,3,4)),rtol=1e-14)
        np.testing.assert_allclose(scales['rms'],np.sqrt(np.mean(frames**2,axis=(0,2,3,4))),rtol=1e-14)
        self.assertEqual(scales['n_dumps'],6)
        with self.assertRaises(ValueError):FrameReader(self.split,'train',verify=False)
        changed=copy.copy(self.split)
        object.__setattr__(changed,'sha256','0'*64)
        with self.assertRaisesRegex(ValueError,'frozen split'):read_channel_scales(path,changed)

    def test_correlation_mask_and_horizon_have_explicit_oracles(self):
        from grmhd_validation.analyses.correlation_times import disk_mask,horizon,decorrelation
        ctx=self.context()
        unmasked=disk_mask(ctx.geometry,ctx.constants)
        resolved=disk_mask(ctx.geometry,ctx.constants,True)
        self.assertTrue(np.any(unmasked & ~resolved))
        np.testing.assert_array_equal(resolved,unmasked & ctx.geometry.resolved.numpy())
        truth=decorrelation(ctx.reader,ctx.geometry,ctx.constants,resolved=True)
        self.assertEqual(truth['n_pairs'],21)
        model=np.ones((1,8,8));baseline=np.ones_like(model)
        model[:,2:,:]=2
        result=horizon(model,baseline,np.ones(8)*2,25.,ctx.constants)
        np.testing.assert_array_equal(result['crossing_steps'],3)
        np.testing.assert_array_equal(result['horizon_rg_c'],75.)
        np.testing.assert_array_equal(result['horizon_tau_c'],1.5)

    def test_cost_timing_stays_metadata_and_uses_controlled_clock(self):
        from grmhd_validation.analyses import cost
        ctx=self.context();start=ctx.starts[0].rollout[0]
        with patch.object(cost.time,'perf_counter',side_effect=[10.,10.4]):
            first=cost.compute(ctx,space='physical',starts=[start])
        self.assertAlmostEqual(ctx.run_info['analysis_runtime']['cost']['milliseconds_per_step'],20.)
        with patch.object(cost.time,'perf_counter',side_effect=[10.,11.]):
            second=cost.compute(ctx,space='physical',starts=[start])
        self.assertEqual(first,second)

    def test_physical_kernels_have_analytic_flat_geometry_oracles(self):
        from grmhd_validation.analyses import _physical as p
        ctx=self.context();shape=ctx.split.shape
        raw=np.zeros(shape,dtype=np.float64);raw[0]=1.;raw[1]=.25
        np.testing.assert_array_equal(p.mass(raw,ctx.geometry),shape[-2]*shape[-1])
        energy,angular=p.conserved(raw,ctx.geometry)
        np.testing.assert_allclose(energy,-.25,rtol=1e-14)
        np.testing.assert_array_equal(angular,0.)
        raw[2]=3.;raw[3]=4.
        velocity=p.ucon(raw[None,2:5],ctx.geometry.gcov,ctx.geometry.gcon)
        np.testing.assert_allclose(velocity[:,0],np.sqrt(26.),rtol=1e-14)
        np.testing.assert_array_equal(velocity[:,1],3.)
        np.testing.assert_array_equal(velocity[:,2],4.)
        # The stencil is independent of the metric-coordinate values: a linear
        # first-direction flux has derivative 1 in interior and 1/2 on edges.
        b=np.indices(shape[1:])[0][None,None].astype(np.float64)
        div=p.divergence(b,np.zeros_like(b),np.zeros_like(b),np.ones(shape[1:]),[1,1,1])
        np.testing.assert_array_equal(div[:,:,1:-1],1.)
        np.testing.assert_array_equal(div[:,:,[0,-1]],.5)
        self.assertEqual(div.dtype,np.float64)

    def test_full_forecast_helper_preserves_model_history(self):
        from grmhd_validation.analyses._analysis_io import forecast_one_step,to_device
        ctx=self.context();seen=[]
        class HistoryAdapter:
            name='history-test';version='v1';history_length=3;max_batch=1
            def step(self,history):
                seen.append(history.clone());return history[:,-1]+.125
        ctx.adapter=HistoryAdapter();start=ctx.starts[0].rollout[0]
        with bind(ctx,'cost','physical',[start]) as (_,model,norm,_,_,_,_):
            curr=to_device(ctx.reader.numpy(start),torch.device('cpu'))
            first=forecast_one_step(model,norm,curr,torch.device('cpu'))
            forecast_one_step(model,norm,to_device(first,torch.device('cpu')),torch.device('cpu'))
        torch.testing.assert_close(seen[0],ctx.reader.history(start,3,'cpu'))
        torch.testing.assert_close(seen[1][:,:-1],seen[0][:,1:])
        torch.testing.assert_close(seen[1][:,-1],first)

    def test_regression_checkpoint_missing_or_changed_is_a_hard_error(self):
        from regression_tron import check_checkpoint
        missing={'checkpoint':str(self.base/'absent.pth'),'checkpoint_sha256':'0'*64}
        with self.assertRaises(FileNotFoundError):check_checkpoint(missing)
        path=self.base/'fixture.pth';path.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            check_checkpoint({**missing,'checkpoint':str(path)})

    def test_physical_odd_lag_structure_moment_uses_equal_slices(self):
        from grmhd_validation.analyses._kevin_math import structure_function_exponents
        ctx=self.context();channel=np.arange(40,dtype=np.float64)[:,None,None]
        with bind(ctx,'turbulence_stats','physical',[ctx.starts[0].rollout[0]]):
            exponent,fit,s3=structure_function_exponents(channel)
        np.testing.assert_allclose(exponent,[2.,3.,4.],rtol=1e-14)
        np.testing.assert_allclose(fit,1.,rtol=1e-14)
        self.assertEqual(s3,8.)

    def test_nested_result_fields_and_full_protocol_settings_are_validated(self):
        from grmhd_validation.validation import schema_validate
        from grmhd_validation.protocol import read_protocol
        import jsonschema
        ctx=self.context()
        from grmhd_validation.analyses import physics_table
        result=physics_table.run(ctx)
        result['data']['per_start'][0]['values']['samples'][0]['unexpected']=1.
        with self.assertRaises(jsonschema.ValidationError):schema_validate(result,'physics_table.schema.json')
        for key,value in [('spatial_bins_min',100),('spectrum_fit_fraction',[.8,.2]),('conditional_bins',0)]:
            protocol=copy.deepcopy(self.protocol);protocol['constants']['full'][key]=value
            path=self.base/'invalid-protocol.json';path.write_bytes(json_bytes(protocol))
            with self.assertRaises((ValueError,jsonschema.ValidationError)):read_protocol(path)


if __name__=='__main__':unittest.main()
