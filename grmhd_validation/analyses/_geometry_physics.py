"""Ported geometry kernels: Kevin Fortune/Sachin, main@ebcc218.
Sources: Fall26/common/normalizer.py and Fall26/common/physics_loss.py.
Only evaluation kernels are retained; geometry arrives through the adapter API.
Legacy loss-space behavior is retained for numerical regression.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._analysis_state import CONFIG, current
from . import _physical as physical


class MetricNormalizer(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.register_buffer('gcov',geometry.gcov.detach().cpu()[None,None])
        self.register_buffer('gcon',geometry.gcon.detach().cpu()[None,None])

    def forward(self, raw):
        state=current()
        if state['space']=='physical':
            value=physical.array(raw)
            if state['analysis']=='stability':
                scale=np.asarray(state['scales']['rms'],dtype=np.float64).reshape(1,8,1,1,1)
                value=np.divide(value,scale,out=np.zeros_like(value),where=scale>1e-30)
            result=torch.from_numpy(value)
            if hasattr(raw,'_gval_dump_id'):result._gval_dump_id=raw._gval_dump_id
            return result
        rho=torch.log10(torch.clamp(raw[:,0],min=CONFIG['floor_rho']))
        ug=torch.log10(torch.clamp(raw[:,1],min=CONFIG['floor_ug']))
        return torch.cat((rho[:,None],ug[:,None],torch.arcsinh(raw[:,2:])),dim=1)

    def gamma_of_raw(self, utilde):
        """Exact Lorentz factor gamma = sqrt(1 + gamma_ij utilde^i utilde^j) from raw utilde."""
        if current()['space']=='physical':
            return torch.from_numpy(physical.gamma(utilde,self.gcov[0,0]))
        u1, u2, u3 = (utilde[:, i:i + 1].double() for i in range(3))
        g = self.gcov[0, 0].double()
        qsq = g[1, 1] * u1 ** 2 + g[2, 2] * u2 ** 2 + g[3, 3] * u3 ** 2 + 2.0 * (g[1, 2] * u1 * u2 + g[1, 3] * u1 * u3 + g[2, 3] * u2 * u3)
        return torch.sqrt(1.0 + torch.clamp(qsq, min=0.0))

    def u0_of(self, utilde):
        """Exact u^0 = gamma / alpha = gamma * sqrt(-g^{00}) from raw utilde."""
        if current()['space']=='physical':
            return torch.from_numpy(physical.ucon(utilde,self.gcov[0,0],self.gcon[0,0])[:,0:1])
        gamma = self.gamma_of_raw(utilde)
        alpha = 1.0 / torch.sqrt(-self.gcon[0, 0, 0, 0].double() + 1e-300)
        return (gamma / alpha).clamp(max=3e+38).float()

    def ucon_of(self, utilde):
        """Reconstruct full coordinate 4-velocity (u^0, u^1, u^2, u^3) from utilde."""
        if current()['space']=='physical':
            return torch.from_numpy(physical.ucon(utilde,self.gcov[0,0],self.gcon[0,0]))
        u1, u2, u3 = (utilde[:, i:i + 1].double() for i in range(3))
        gamma = self.gamma_of_raw(utilde)
        alpha = 1.0 / torch.sqrt(-self.gcon[0, 0, 0, 0].double() + 1e-300)
        u0 = gamma / alpha
        gcon00 = self.gcon[0, 0, 0, 0].double()
        u_coord1 = (u1 + u0 * (self.gcon[0, 0, 0, 1].double() / gcon00)).float()
        u_coord2 = (u2 + u0 * (self.gcon[0, 0, 0, 2].double() / gcon00)).float()
        u_coord3 = (u3 + u0 * (self.gcon[0, 0, 0, 3].double() / gcon00)).float()
        return torch.cat([u0.float(), u_coord1, u_coord2, u_coord3], dim=1)

class GeometryCriterion(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        for key in ('gdet','gcov','gcon'):
            self.register_buffer(key,getattr(geometry,key).detach().cpu()[None,None])
        self.dx,self.dy,self.dz=map(float,geometry.dx.detach().cpu())
        if geometry.eos_gamma is None:
            raise ValueError('Full physics analyses require eos_gamma in geometry.npz')
        self.eos_gamma=float(geometry.eos_gamma)
        self.dt_dump=float(geometry.dt_dump)
        nr,ntheta=self.gdet.shape[-3:-1]
        self.register_buffer('spec_win',(torch.hann_window(nr)[:,None,None]*torch.hann_window(ntheta)[None,:,None])[None,None])

    @staticmethod
    def _safe_rho_ug(raw, floor, ceil):
        if current()['space']=='physical':
            return raw.double()
        return torch.pow(10,torch.clamp(torch.log10(torch.clamp(raw,min=1e-300)),np.log10(floor),np.log10(ceil)))

    @staticmethod
    def _signed_log(x):
        """Signed-log transform sl(x) = sign(x)*log1p(|x|) for the shell terms."""
        if current()['space']=='physical':
            value=physical.array(x)
            return torch.from_numpy(np.sign(value)*np.log1p(np.abs(value)))
        x = x.double()
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def calc_divB(self, b1, b2, b3):
        if current()['space']=='physical':
            return torch.from_numpy(physical.divergence(b1,b2,b3,self.gdet,(self.dx,self.dy,self.dz)))
        f1, f2, f3 = (self.gdet.double() * b1.double(), self.gdet.double() * b2.double(), self.gdet.double() * b3.double())
        f1_pad = F.pad(f1, (0, 0, 0, 0, 1, 1), mode='replicate')
        d1 = (f1_pad[:, :, 2:] - f1_pad[:, :, :-2]) / (2 * self.dx)
        f2_pad = torch.cat([f2[:, :, :, 0:1, :], f2, f2[:, :, :, -1:, :]], dim=3)
        d2 = (f2_pad[:, :, :, 2:] - f2_pad[:, :, :, :-2]) / (2 * self.dy)
        f3_pad = F.pad(f3, (1, 1, 0, 0, 0, 0), mode='circular')
        d3 = (f3_pad[:, :, :, :, 2:] - f3_pad[:, :, :, :, :-2]) / (2 * self.dz)
        return (d1 + d2 + d3) / (self.gdet + 1e-20)
