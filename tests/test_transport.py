"""Transport sign, four-velocity and coarse-ring tests with independent metrics."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from grmhd_validation.adapter import Geometry
from grmhd_validation.analyses._physical import ucon
from grmhd_validation.analyses._transport import tensors,shell_fluxes,shells
from grmhd_validation.protocol import read_protocol
from pathlib import Path

PROTOCOL,_=read_protocol(Path(__file__).resolve().parents[1]/'protocols/forecast_v1.json')


def geometry(metric,radii=(10.,12.)):
    shape=(len(radii),3,4);r,t,ph=np.meshgrid(radii,[.2,np.pi/2,np.pi-.2],np.arange(4)*np.pi/2,indexing='ij')
    g=np.broadcast_to(np.asarray(metric)[:,:,None,None,None],(4,4,*shape)).copy()
    inverse=np.broadcast_to(np.linalg.inv(metric)[:,:,None,None,None],g.shape).copy()
    determinant=np.ones(shape)*np.sqrt(-np.linalg.det(metric))
    return Geometry(*(torch.from_numpy(value) for value in (r,t,ph,g,inverse,determinant,np.array([.1,.2,.3]),
                       np.array(0.),np.array(25.),r>=6.11,np.array(5./3.))))


class TransportTests(unittest.TestCase):
    def test_inward_energy_and_angular_momentum_have_independent_signed_oracles(self):
        geo=geometry(np.diag([-1.,1.,1.,1.]));raw=np.zeros((8,*geo.r.shape))
        raw[0]=2.;raw[1]=.5;raw[2]=-.3;raw[4]=.2
        area=3*4*.2*.3;enthalpy=2.+(5./3.)*.5;ut=np.sqrt(1+.3**2+.2**2)
        result=shell_fluxes(raw,geo,PROTOCOL['constants'])
        np.testing.assert_allclose(result['mdot'],2.*.3*area,rtol=1e-13)
        np.testing.assert_allclose(result['edot'],enthalpy*.3*ut*area,rtol=1e-13)
        np.testing.assert_allclose(result['ldot'],enthalpy*.3*.2*area,rtol=1e-13)
        np.testing.assert_allclose(result['eta'],1.-enthalpy*ut/2.,rtol=1e-13)
        np.testing.assert_array_equal(result['eta_jet'],0.)

    def test_shifted_velocity_obeys_metric_and_transport_is_not_utilde1(self):
        alpha=.8;beta=.4
        g=np.diag([-alpha**2+beta**2,1.,1.,1.]);g[0,1]=g[1,0]=beta
        geo=geometry(g);raw=np.zeros((8,*geo.r.shape));raw[0]=2.;raw[2]=.2
        u=ucon(raw[None,2:5],geo.gcov,geo.gcon)[0]
        cov=np.einsum('ij...,j...->i...',geo.gcov.numpy(),u)
        np.testing.assert_allclose(np.sum(u*cov,axis=0),-1.,rtol=1e-13,atol=1e-13)
        self.assertTrue(np.all(u[1]<0));self.assertTrue(np.all(raw[2]>0))
        result=shell_fluxes(raw,geo,PROTOCOL['constants'])
        weight=geo.gdet.numpy()*.2*.3
        expected=-np.sum(raw[0]*u[1]*weight,axis=(1,2))
        np.testing.assert_allclose(result['mdot'],expected,rtol=1e-13)
        self.assertTrue(np.all(result['mdot']>0))
        np.testing.assert_array_equal(result['ldot'],0.)

    def test_cold_marginally_bound_radial_inflow_has_zero_efficiency(self):
        # Schwarzschild lapse at r=10; u_t=-1 fixes the marginally-bound
        # free-fall energy. The metric is an independent exact local oracle.
        f=.8;g=np.diag([-f,1/f,100.,100.]);geo=geometry(g)
        raw=np.zeros((8,*geo.r.shape));raw[0]=1.;raw[2]=-np.sqrt(1-f)
        result=shell_fluxes(raw,geo,PROTOCOL['constants'])
        np.testing.assert_allclose(result['edot'],result['mdot'],rtol=1e-13)
        np.testing.assert_allclose(result['eta'],0.,atol=1e-13)
        np.testing.assert_array_equal(result['ldot'],0.)
        np.testing.assert_array_equal(result['p_jet'],0.)

    def test_stress_components_and_jet_rest_mass_subtraction(self):
        geo=geometry(np.diag([-1.,1.,1.,1.]))
        raw=np.zeros((8,*geo.r.shape));raw[0]=1.;raw[1]=.25;raw[2]=.5;raw[5]=.2;raw[6]=2.
        u,uc,b,bc,bsq,trt,trphi=tensors(raw,geo)
        # Independent cellwise tensor contraction, not the shell helper.
        np.testing.assert_allclose(np.sum(u*uc,axis=1),-1.,atol=1e-13)
        np.testing.assert_allclose(np.sum(u*bc,axis=1),0.,atol=1e-13)
        enthalpy=raw[0]+float(geo.eos_gamma)*raw[1]
        expected_t=(enthalpy+bsq[0,0])*u[0,1]*uc[0,0]-b[0,1]*bc[0,0]
        expected_phi=(enthalpy+bsq[0,0])*u[0,1]*uc[0,3]-b[0,1]*bc[0,3]
        result=shell_fluxes(raw,geo,PROTOCOL['constants']);weight=geo.gdet.numpy()*.2*.3
        np.testing.assert_allclose(result['edot'],np.sum(expected_t*weight,axis=(1,2)),rtol=1e-13)
        np.testing.assert_allclose(result['ldot'],-np.sum(expected_phi*weight,axis=(1,2)),rtol=1e-13)
        mask=np.abs(geo.theta.numpy()-np.pi/2)>1.
        old=-np.sum(expected_t*weight*mask,axis=(1,2))
        rest=np.sum(raw[0]*u[0,1]*weight*mask,axis=(1,2))
        np.testing.assert_allclose(result['p_jet'],old-rest,rtol=1e-13)
        self.assertTrue(np.isnan(result['eta']).all());self.assertTrue(np.isnan(result['eta_jet']).all())
        self.assertTrue(np.all(result['eta_jet_reason']=='nonpositive_mass_inflow'))

    def test_rc300_headline_is_first_wholly_resolved_shell(self):
        geo=geometry(np.diag([-1.,1.,1.,1.]),radii=(2.,6.,7.,10.))
        ctx=SimpleNamespace(geometry=geo,split=SimpleNamespace(split_id='rc300_synthetic'),constants=PROTOCOL['constants'])
        result=shells(ctx)
        self.assertTrue(result['horizon']['coarse_ring'])
        self.assertEqual(result['resolved']['radius_rg'],7.)
        self.assertFalse(result['resolved']['coarse_ring'])
        mask=geo.resolved.clone();mask[2,0,0]=False;ctx.geometry=replace(geo,resolved=mask)
        self.assertEqual(shells(ctx)['resolved']['radius_rg'],10.)
