"""Independent quick-tier oracles and runtime regression against the pinned source.

Originals: Kevin Fortune, Fall26/eval/evaluate_forecast_ds.py and
Fall26/common/normalizer.py, main@ebcc218; TRON normalization by Sachin.
Only selected AST definitions are loaded at runtime; no original is vendored.
"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch

from grmhd_validation.analyses.common import loss_space,nmae,relative_l1,skill
from grmhd_validation.analyses.single_step import block_indices,block_skill_ci
from grmhd_validation.analyses.rollout_curve import CHANNEL_COLORS,dynamic_xticks,start_skill_ci
from grmhd_validation.protocol import read_protocol

REPO=Path(__file__).resolve().parents[1]
PROTOCOL,_=read_protocol(REPO/'protocols/forecast_v1.json')
REFERENCE=Path(os.environ.get('GVAL_REFERENCE_CHECKOUT','/u/gkerex/project/reference/neural_operator_msai_spring26'))


class MetricTests(unittest.TestCase):
    def test_float64_transform_floors_no_ceilings_or_clamps(self):
        raw=np.full((8,2,2,2),1e25,dtype=np.float32)
        raw[0,0]=0;raw[1,0]=-1
        actual=loss_space(raw,PROTOCOL['constants'])
        self.assertEqual(actual.dtype,np.float64);self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual[0,0],-20.)
        np.testing.assert_array_equal(actual[1,0],-22.)
        self.assertGreater(actual[0,1,0,0],24.)
        self.assertGreater(actual[2,0,0,0],50.)
        physical=np.zeros_like(raw)
        self.assertTrue(np.isnan(relative_l1(raw,physical,PROTOCOL['constants'])).all())
        self.assertTrue(np.isnan(skill(np.ones(8),np.zeros(8),PROTOCOL['constants'])).all())
        self.assertTrue(np.isfinite(skill(np.ones(8),np.zeros(8),PROTOCOL['constants'],threshold=False)).all())

    def test_chunk_blocks_never_cross_boundaries_bootstrap_recomputes_ratio(self):
        protocol=copy.deepcopy(PROTOCOL);protocol.update(bootstrap_resamples=19,bootstrap_block=3)
        blocks=block_indices([5,4],3)
        self.assertEqual([b.tolist() for b in blocks],[[0,1,2],[3,4],[5,6,7],[8]])
        baseline=np.arange(1,73,dtype=np.float64).reshape(9,8)/10
        model=baseline**2/10
        rng=np.random.default_rng(protocol['seed'])
        values=[]
        for _ in range(19):
            indexes=np.concatenate([blocks[i] for i in rng.integers(0,4,size=4)])
            values.append(1-model[indexes].mean(0)/(baseline[indexes].mean(0)+1e-30))
        expected=np.nanpercentile(values,[2.5,97.5],axis=0)
        actual,count=block_skill_ci(model,baseline,[5,4],protocol)
        self.assertEqual(count,4);np.testing.assert_array_equal(actual,expected)

    def test_rollout_bootstrap_resamples_individual_starts(self):
        protocol=copy.deepcopy(PROTOCOL);protocol['bootstrap_resamples']=17
        model=np.arange(1,21,dtype=np.float64).reshape(4,5)
        baseline=model**2
        rng=np.random.default_rng(42);values=[]
        for _ in range(17):
            indexes=rng.integers(0,4,size=4)
            values.append(1-model[indexes].mean(0)/(baseline[indexes].mean(0)+1e-30))
        np.testing.assert_array_equal(start_skill_ci(model,baseline,protocol),np.percentile(values,[2.5,97.5],axis=0))

    def test_protocol_constants_are_validated(self):
        import tempfile
        import jsonschema
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'p.json'
            for group,key,value in [('loss_space','floor_rho',0),('ci','percentiles',[99.,2.]),('figures','backend','GUI')]:
                protocol=copy.deepcopy(PROTOCOL);protocol['constants'][group][key]=value
                path.write_text(json.dumps(protocol))
                with self.assertRaises((ValueError,jsonschema.ValidationError)):
                    read_protocol(path)


@unittest.skipUnless(REFERENCE.is_dir(),'Pinned private reference checkout is unavailable; set GVAL_REFERENCE_CHECKOUT')
class SourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # git show keeps this regression pinned even if the read-only checkout moves.
        def source(path):
            return subprocess.run(['git','-C',str(REFERENCE),'show','ebcc218:'+path],capture_output=True,text=True,check=True).stdout
        cls.namespace={'np':np,'torch':torch,'CONFIG':{'floor_rho':1e-20,'floor_ug':1e-22}}
        evaluator=ast.parse(source('Fall26/eval/evaluate_forecast_ds.py'))
        wanted={'to_tensor_cpu','nmae_per_channel','rel_l1_per_channel','dynamic_xticks'}
        nodes=[node for node in evaluator.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
        palette=[node for node in evaluator.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHANNEL_COLORS' for t in node.targets)]
        normalizer=ast.parse(source('Fall26/common/normalizer.py'))
        original=next(n for n in normalizer.body if isinstance(n,ast.ClassDef) and n.name=='DataNormalizer')
        forward=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        forward.name='original_forward'
        nodes+=[forward,*palette]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned reference definitions>','exec'),cls.namespace)

    def test_metrics_match_original_float32_normalizer_with_approved_tolerance(self):
        rng=np.random.default_rng(1729)
        for _ in range(5):
            truth=rng.uniform(.1,10,(8,4,3,2)).astype(np.float32)
            prediction=truth*rng.uniform(.6,1.6,truth.shape).astype(np.float32)
            prediction[2:]-=3.;truth[2:]-=3.
            norm=lambda raw:self.namespace['original_forward'](None,raw)
            expected=self.namespace['nmae_per_channel'](prediction,truth,norm)
            np.testing.assert_allclose(nmae(prediction,truth,PROTOCOL['constants']),expected,rtol=1e-5,atol=0)
            expected=self.namespace['rel_l1_per_channel'](prediction,truth)
            np.testing.assert_allclose(relative_l1(prediction,truth,PROTOCOL['constants']),expected,rtol=1e-5,atol=0)

    def test_palette_and_ticks_match_pinned_original(self):
        self.assertEqual(CHANNEL_COLORS,self.namespace['CHANNEL_COLORS'])
        for maximum in (1,4,20,21,50,51,100):
            self.assertEqual(dynamic_xticks(maximum),self.namespace['dynamic_xticks'](maximum))


if __name__=='__main__':
    unittest.main()

class AttractorTests(unittest.TestCase):
    def test_sigma_uses_every_tenth_dump_of_each_evaluated_chunk(self):
        from types import SimpleNamespace
        from grmhd_validation.analyses.rollout_curve import truth_sigma
        ids=[]
        def read(index):
            ids.append(index)
            # Nonlinear evolution makes the selected-dump rule observable.
            return np.arange(64,dtype=np.float32).reshape(8,2,2,2)*(index+1)**2+1
        chunks=[SimpleNamespace(first=4,last=28),SimpleNamespace(first=51,last=63)]
        ctx=SimpleNamespace(constants=PROTOCOL['constants'],partition='validation',
                            split=SimpleNamespace(chunks=lambda partition:chunks),reader=SimpleNamespace(numpy=read))
        actual=truth_sigma(ctx)
        expected_ids=[4,14,24,51,61]
        self.assertEqual(ids,expected_ids)
        expected=np.mean([loss_space(read(i),PROTOCOL['constants']).reshape(8,-1).std(axis=1) for i in expected_ids],axis=0)
        np.testing.assert_array_equal(actual,expected)


class ComparisonTests(unittest.TestCase):
    def test_cross_platform_and_gpu_relative_tolerances(self):
        import tempfile
        from grmhd_validation.artifacts import json_bytes
        from grmhd_validation.conformance import compare_results
        with tempfile.TemporaryDirectory() as temporary:
            a,b=(Path(temporary)/s for s in ('a','b'))
            a.mkdir();b.mkdir()
            (a/'summary.json').write_bytes(json_bytes({'score':1.0,'zero':0.,'undefined':None}))
            (b/'summary.json').write_bytes(json_bytes({'score':1.0+5e-13,'zero':0.,'undefined':None}))
            self.assertFalse(compare_results(a,b)['ok'])
            self.assertTrue(compare_results(a,b,1e-12)['ok'])
            (b/'summary.json').write_bytes(json_bytes({'score':1.0+5e-5,'zero':0.,'undefined':None}))
            self.assertFalse(compare_results(a,b,1e-12)['ok'])
            self.assertTrue(compare_results(a,b,1e-4)['ok'])
            (b/'summary.json').write_bytes(json_bytes({'score':1.01,'zero':0.,'undefined':None}))
            self.assertFalse(compare_results(a,b,1e-4)['ok'])
            (b/'summary.json').write_bytes(json_bytes({'score':1.,'zero':1e-30,'undefined':None}))
            self.assertFalse(compare_results(a,b,1e-4)['ok'])
