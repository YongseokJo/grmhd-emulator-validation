"""Independent oracles for the final chunk bootstrap and nonlinear estimators."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from grmhd_validation.analyses._aggregate import aggregate,blocks_for_starts


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.groups=[SimpleNamespace(first=0,last=9),SimpleNamespace(first=100,last=109)]
        self.starts=[0,1,2,100,101]
        self.protocol={'bootstrap_block':2,'bootstrap_resamples':31,'seed':42,
                       'constants':{'ci':{'percentiles':[2.5,97.5]}}}

    def test_short_blocks_never_cross_chunks_and_ci_uses_concatenated_samples(self):
        blocks=blocks_for_starts(self.starts,self.groups,2)
        self.assertEqual([b.tolist() for b in blocks],[[0,1],[2],[3,4]])
        values=np.array([1.,10.,100.,1000.,10000.])
        rows=[{'start':s,'values':{'value':value,'r':np.array([2.,3.])}} for s,value in zip(self.starts,values)]
        result=aggregate(rows,self.groups,self.protocol)
        rng=np.random.default_rng(42);expected=[]
        for _ in range(31):
            selected=np.concatenate([blocks[i] for i in rng.integers(0,3,size=3)])
            expected.append(values[selected].mean())
        self.assertEqual(result['mean']['value'],values.mean())
        np.testing.assert_array_equal([result['lower']['value'],result['upper']['value']],np.percentile(expected,[2.5,97.5]))
        np.testing.assert_array_equal(result['mean']['r'],[2.,3.])
        coordinates=[{'start':s,'values':{'centers':np.array([.1,.2]),'widths':np.array([.01,.01])}} for s in self.starts]
        grid=aggregate(coordinates,self.groups,self.protocol)
        for bound in ('mean','lower','upper'):
            np.testing.assert_array_equal(grid[bound]['centers'],[.1,.2])
            np.testing.assert_array_equal(grid[bound]['widths'],[.01,.01])
        self.assertEqual(result['bootstrap']['n_blocks'],3)

    def test_nonlinear_statistic_is_recomputed_in_each_draw(self):
        values=np.array([1.,2.,3.,10.,20.])
        rows=[{'start':s,'values':{'bias':np.array([v]),'rmse':np.array([abs(v)])}} for s,v in zip(self.starts,values)]
        reducer=lambda group:np.sqrt(np.mean(np.stack([row['bias']**2 for row in group]),axis=0))
        result=aggregate(rows,self.groups,self.protocol,{('rmse',):reducer})
        self.assertAlmostEqual(float(result['mean']['rmse'][0]),np.sqrt(np.mean(values**2)))
        self.assertNotEqual(float(result['mean']['rmse'][0]),np.mean(np.abs(values)))

    def test_undefined_efficiencies_keep_reasons_and_finite_counts(self):
        rows=[{'start':0,'values':{'eta':np.array([np.nan,np.nan]),'eta_reason':np.array(['nonpositive_mass_inflow']*2)}},
              {'start':100,'values':{'eta':np.array([2.,np.nan]),'eta_reason':np.array(['','nonpositive_mass_inflow'])}}]
        result=aggregate(rows,self.groups,self.protocol)
        self.assertEqual(result['mean']['eta'][0],2.)
        self.assertTrue(np.isnan(result['mean']['eta'][1]))
        np.testing.assert_array_equal(result['valid_counts']['eta'],[1,0])
        self.assertIn('per-start',result['mean']['eta_reason'][0])
        self.assertEqual(result['mean']['eta_reason'][1],'nonpositive_mass_inflow')
        finite=[{'start':s,'values':{'eta':np.array([value])}} for s,value in [(0,1.),(100,2.)]]
        np.testing.assert_array_equal(aggregate(finite,self.groups,self.protocol)['valid_counts']['eta'],[2])

    def test_duplicate_or_outside_starts_and_changed_axes_fail(self):
        with self.assertRaises(ValueError):blocks_for_starts([0,0],self.groups,2)
        with self.assertRaises(ValueError):blocks_for_starts([99],self.groups,2)
        rows=[{'start':0,'values':{'r':[2.,3.]}},{'start':100,'values':{'r':[2.,4.]}}]
        with self.assertRaisesRegex(ValueError,'Invariant'):aggregate(rows,self.groups,self.protocol)
