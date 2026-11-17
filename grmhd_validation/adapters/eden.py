"""Eden Growney's start-11672 protocol-A examples, written by Claude/GPT/Gemini.

Sources: rc300_packets/start_11672_{claude,gpt,gemini}/submission/code/*,
eden/rc300-frontier-15nov@f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a.
This wrapper supplies arguments, validates coordinate assumptions and removes
packet paths; the credited kernels retain each original numerical method.
Historical pilot data exposure is provenance, never a benchmark score.
"""
import importlib
import numpy as np
import torch
from ..adapter import ModelAdapter


class Eden(ModelAdapter):
    version = 'v1'
    history_length = 1
    max_batch = 1
    requires_checkpoint = False
    agent = None

    def __init__(self, checkpoint, geometry, device, options):
        if checkpoint is not None or options:
            raise ValueError('Eden methods have no checkpoint or tunable benchmark options')
        self.geo = {key: value.detach().cpu().numpy() for key, value in {
            'r': geometry.r, 'h': geometry.theta, 'ph': geometry.phi,
            'gcov': geometry.gcov, 'gcon': geometry.gcon, 'dx': geometry.dx,
            'spin': geometry.spin, 'dt_dump': geometry.dt_dump}.items()}
        # The original kernels assume axisymmetry and a uniform periodic phi grid.
        # Compact a full-phi metric only after proving it is identical on phi.
        for key in ('gcov', 'gcon'):
            array = self.geo[key]
            if array.ndim != 5 or not np.allclose(array, array[..., :1], rtol=0, atol=0):
                raise ValueError('Eden methods require an axisymmetric metric')
            self.geo[key] = array[..., :1]
        phi = self.geo['ph'][0, 0]
        if len(phi) < 2 or not np.allclose(np.diff(phi), self.geo['dx'][2]) or not np.isclose(len(phi)*self.geo['dx'][2], 2*np.pi):
            raise ValueError('Eden methods require uniform periodic phi covering 2*pi')
        if not np.isfinite(float(geometry.dt_dump)) or float(geometry.dt_dump) <= 0:
            raise ValueError('Positive dump cadence required')
        if self.agent == 'claude':
            radial = self.geo['r'][:, 0, 0]
            theta = self.geo['h'][0, :, 0]
            if len(radial) < 2 or not np.allclose(np.diff(np.log(radial)), self.geo['dx'][0], rtol=1e-4, atol=1e-6):
                raise ValueError('Claude method requires uniform x1=log(r) spacing')
            if not np.allclose(theta, np.pi-theta[::-1], atol=1e-5):
                raise ValueError('Claude method requires an equator-symmetric theta grid')
            if not 0 < float(geometry.spin) < 1:
                raise ValueError('Claude orbital cap requires 0 < spin < 1')
        self.kernel = importlib.import_module('._eden_'+self.agent, __package__)
        self.provenance = {'protocol': 'A', 'author': 'Eden Growney', 'originating_agent': self.agent,
                           'source_commit': 'f8cd379a562e9b0fc6f96ee36d5f9855ddc1d27a',
                           'known_prior_data_exposure': 'Developed in Eden rc300 start-11672 pilot; blindness to current split is not established'}

        from pathlib import Path
        from ..hashing import sha256
        self.provenance['source_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('_eden_*.py')}

    def step(self, history):
        value = history[0, -1].detach().cpu().numpy()
        if self.agent == 'claude':
            result = self.kernel.forecast(value, self.geo, float(self.geo['dt_dump']))
        else:
            result = self.kernel.predict_next(value, self.geo)
        return torch.from_numpy(result).to(history.device).unsqueeze(0)


class Claude(Eden):
    name = 'eden-claude'
    agent = 'claude'


class GPT(Eden):
    name = 'eden-gpt'
    agent = 'gpt'


class Gemini(Eden):
    name = 'eden-gemini'
    agent = 'gemini'
