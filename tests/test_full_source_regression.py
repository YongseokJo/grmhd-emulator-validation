"""Synthetic kernel regression in Kevin's configuration, not benchmark scores.

Original: Kevin Fortune/Sachin, main@ebcc218, evaluate_forecast_ds.py and common.
These tests isolate faithful math from the separately tested physical-space
changes. Real fixed-checkpoint regression is a required explicit Slurm command.
"""
import unittest

import numpy as np
import torch

from test_full_analyses import FullFixture
from reference_support import REFERENCE,load_reference
from grmhd_validation.analyses import _kevin_math as port
from grmhd_validation.analyses._analysis_io import bind,geometry_views


@unittest.skipUnless(REFERENCE.is_dir(),'Pinned private source unavailable in public CI')
class FullSourceRegressionTests(FullFixture):
    def test_geometry_physics_statistics_and_structure_match_original(self):
        original,_=load_reference();ctx=self.context()
        norm=original['DataNormalizer'](gcov=ctx.geometry.gcov[None,None].float(),
                                        gcon=ctx.geometry.gcon[None,None].float())
        crit=original['PhysicsLoss3D'](torch.device('cpu'),str(self.data/'geometry.npz'))
        _,geo=geometry_views(ctx.geometry)
        raw=ctx.reader.numpy(6);true=ctx.reader.numpy(7)
        original['CONFIG']['eos_gamma']=float(ctx.geometry.eos_gamma)
        original['CONFIG']['dt_dump']=float(ctx.geometry.dt_dump)
        with bind(ctx,'physics_table','loss',[6]) as (_,_,pn,_,pc,_,_):
            calls={
              'active_disk_mask':((geo,),(geo,)),
              'nmae_per_channel':((raw,true,pn),(raw,true,norm)),
              'rel_l1_per_channel':((raw,true),(raw,true)),
              'mass_shells':((raw,pc,pn),(raw,crit,norm)),
              'conserved_shells':((raw,pc,pn),(raw,crit,norm)),
              'physical_divb':((raw,pc),(raw,crit)),
              'normalized_divb':((raw,pc,geo),(raw,crit,geo)),
              'gamma_stats':((raw,pn,torch.device('cpu')),(raw,norm,torch.device('cpu'))),
              'radial_spectrum':((torch.from_numpy(raw)[None],pc,0),(torch.from_numpy(raw)[None],crit,0)),
              'mdot_shells':((raw,pc,pn),(raw,crit,norm)),
              'magnetic_flux_shells':((raw,pc),(raw,crit)),
              'b2_magnitude':((raw,geo),(raw,geo)),
              'bsq_full_components':((raw,geo,pn),(raw,geo,norm)),
              'jet_power_shells':((raw,geo,pn,5./3.),(raw,geo,norm,5./3.)),
              'compute_radial_error_profile':((raw,true,geo),(raw,true,geo)),
              'structure_function_exponents':((raw[0],),(raw[0],)),
              'pdf_on_shared_bins':((raw[0],true[0]),(raw[0],true[0])),
              'pairwise_masked_corrcoef':((raw.reshape(8,-1).T,),(raw.reshape(8,-1).T,)),
            }
            for name,(adapted_args,source_args) in calls.items():
                with self.subTest(kernel=name):
                    actual=getattr(port,name)(*adapted_args)
                    expected=original[name](*source_args)
                    self.compare(actual,expected)
            args=(torch.from_numpy(raw)[None],torch.from_numpy(true)[None])
            self.compare(port.flux_balance_residual_local(*args,pc,pn,geo),
                         original['flux_balance_residual_local'](*args,crit,norm,geo))

    def compare(self,actual,expected):
        if isinstance(expected,(list,tuple)):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):self.compare(a,b)
        else:
            if isinstance(actual,torch.Tensor):actual=actual.detach().cpu().numpy()
            if isinstance(expected,torch.Tensor):expected=expected.detach().cpu().numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=0,equal_nan=True)

    def test_correlation_legacy_retains_original_float32_error_reduction(self):
        from grmhd_validation.analyses.correlation_times import legacy_nmae
        original,_=load_reference();ctx=self.context()
        norm=original['DataNormalizer'](gcov=ctx.geometry.gcov[None,None].float(),
                                       gcon=ctx.geometry.gcon[None,None].float())
        truth=ctx.reader.numpy(7);prediction=ctx.reader.numpy(6)
        np.testing.assert_array_equal(legacy_nmae(prediction,truth,ctx.constants),
                                      original['nmae_per_channel'](prediction,truth,norm))


if __name__=='__main__':unittest.main()
