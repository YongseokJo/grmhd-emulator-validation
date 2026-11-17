"""Physical adapters ported from Kevin Fortune's evaluators (TRON: Sachin).

Sources: Fall26/eval/evaluate_tron.py, fall26-tron-rc300@2667a5f;
Fall26/eval/evaluate_vit.py and Fall26/common/normalizer.py, main@ebcc218.
Changes: geometry supplied by argument; checkpoint config instead of experiment
folders; no filesystem data discovery, physics loss, scoring or plotting.
"""
import numpy as np
import torch
from ..adapter import ModelAdapter


class Decode(torch.nn.Module):
    """Kevin's model-owned normalization and exact Lorentz clamp."""
    def __init__(self, geometry):
        super().__init__()
        self.register_buffer('gcov', geometry.gcov.float())
        self._clamps = 0

    def forward(self, raw):
        return torch.cat((torch.log10(raw[:, :1].clamp(min=1e-20)),
                          torch.log10(raw[:, 1:2].clamp(min=1e-22)),
                          torch.arcsinh(raw[:, 2:])), dim=1)

    def inverse(self, value):
        rho = torch.pow(10, value[:, :1].clamp(np.log10(1e-20), 5.))
        ug = torch.pow(10, value[:, 1:2].clamp(np.log10(1e-22), 5.))
        velocity = torch.sinh(value[:, 2:5].clamp(-80., 80.))
        magnetic = torch.sinh(value[:, 5:].clamp(-80., 80.))
        u1, u2, u3 = (velocity[:, i:i+1].double() for i in range(3))
        g = self.gcov.double()
        q = (g[1, 1]*u1**2 + g[2, 2]*u2**2 + g[3, 3]*u3**2
             + 2*(g[1, 2]*u1*u2 + g[1, 3]*u1*u3 + g[2, 3]*u2*u3))
        gamma = torch.sqrt(1 + q.clamp(min=0))
        need = gamma > 50.
        scale = torch.where(need, torch.sqrt((2499./(gamma**2-1).clamp(min=1e-30)).clamp(max=1)), torch.ones_like(gamma))
        self._clamps = self._clamps + need.sum()
        velocity = (velocity.double()*scale).float()
        return torch.cat((rho, ug, velocity, magnetic), dim=1).float()

    def lorentz_clamp_count(self, reset=True):
        result = torch.as_tensor(self._clamps, dtype=torch.float64)
        if reset:self._clamps = 0
        return result


class Learned(ModelAdapter):
    history_length = 1
    version = 'v1'
    max_batch = 1
    requires_checkpoint = True

    def __init__(self, checkpoint, geometry, device, options):
        # No arbitrary pickle objects are needed by these state-dict checkpoints.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if set(options) - {'config'}:
            raise ValueError('Only an explicit architecture config override is supported')
        config = options.get('config', payload.get('config'))
        self.model = self.build(config, geometry).to(device)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.eval()
        self.normalizer = Decode(geometry).to(device)
        self.provenance = {'architecture': self.name, 'use_amp': bool(getattr(self.model, 'use_amp', False)),
                           'normalization': 'Kevin main@ebcc218 fixed log10/arcsinh; exact gamma<=50 decode',
                           'config': config}

        from pathlib import Path
        from ..hashing import sha256
        self.provenance['source_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('_*_model.py')}

    def step(self, history):
        current = self.normalizer(history[:, -1])
        return self.normalizer.inverse(current + self.model(current))

    def lorentz_clamp_count(self, reset=True):
        return self.normalizer.lorentz_clamp_count(reset)


class TRON(Learned):
    name = 'tron'

    def build(self, config, geometry):
        from ._tron_model import TRON as Network
        if config is not None and config.get('in_channels', 8) != 8:
            raise ValueError('TRON requires the eight physical channels')
        return Network(in_c=8)


class ViT(Learned):
    name = 'vit'

    def build(self, config, geometry):
        from ._vit_model import build_model
        if not isinstance(config, dict):
            raise ValueError('ViT requires checkpoint config or options.config, never a guessed experiment')
        coords = tuple(value.detach().cpu().numpy() for value in (geometry.r, geometry.theta, geometry.phi))
        return build_model(config, coords)
