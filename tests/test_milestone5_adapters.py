"""Faithfulness checks against pinned private sources, read-only at runtime."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from grmhd_validation.adapter import Geometry,validate_output
from grmhd_validation.adapters.learned import Decode,ViT
from grmhd_validation.adapters import _eden_claude,_eden_gpt,_eden_gemini
from grmhd_validation.adapters.eden import Claude,GPT,Gemini

from reference_support import REFERENCE
REF=str(REFERENCE)


def original(commit,path,keep=None):
    text=subprocess.check_output(['git','-C',REF,'show',commit+':'+path],text=True)
    tree=ast.parse(text)
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            if keep is None or node.name in keep:selected.append(node)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.ImportFrom) and node.level:continue
            selected.append(node)
        elif isinstance(node,ast.Assign):
            names={target.id for target in node.targets if isinstance(target,ast.Name)}
            if not names & {'ROOT','SUB','INPUTS','SUBMISSION'}:selected.append(node)
    module=ast.Module(body=selected,type_ignores=[])
    namespace={};exec(compile(module,path,'exec'),namespace)
    return namespace


def geometry(shape=(8,6,16)):
    r=np.exp(2+np.arange(shape[0])*.1)
    theta=(np.arange(shape[1])+.5)*np.pi/shape[1]
    phi=(np.arange(shape[2])+.5)*2*np.pi/shape[2]
    r,t,p=np.meshgrid(r,theta,phi,indexing='ij')
    metric=np.broadcast_to(np.diag([-1.,1.,1.,1.])[:,:,None,None,None],(4,4,shape[0],shape[1],1)).copy()
    return Geometry(*(torch.from_numpy(v) for v in (r,t,p,metric,metric,np.ones((*shape[:2],1)),np.array([.1,np.pi/shape[1],2*np.pi/shape[2]]),np.array(.94),np.array(100.),np.ones(shape,bool))))


class AdapterTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_dir(),'Pinned private source unavailable in public CI')
    def test_decode_matches_original_normalizer_including_floors_and_lorentz(self):
        geo=geometry();source=original('ebcc218','Fall26/common/normalizer.py',{'DataNormalizer'})
        source['CONFIG']={'floor_rho':1e-20,'floor_ug':1e-22,'ceiling_rho':1e5,'ceiling_ug':1e5,'max_gamma':50.}
        normalizer=source['DataNormalizer'](gcov=geo.gcov.float()[None,None],gcon=geo.gcon.float()[None,None])
        port=Decode(geo)
        raw=torch.randn(1,8,*geo.shape);raw[:,:2]=raw[:,:2].abs()
        raw[:,0,0]=1e-30;raw[:,1,1]=1e-30
        torch.testing.assert_close(port(raw),normalizer(raw),rtol=0,atol=0)
        value=port(raw);value[:,2:5]*=40;value[:,0,0]=10;value[:,1,1]=-50
        torch.testing.assert_close(port.inverse(value),normalizer.inverse(value),rtol=0,atol=0)
        self.assertEqual(int(port.lorentz_clamp_count()),int(normalizer.lorentz_clamp_count()))

    @unittest.skipUnless(REFERENCE.is_dir(),'Pinned private source unavailable in public CI')
    def test_vit_adapter_matches_pinned_model_and_original_decode(self):
        geo=geometry((4,4,4));cfg={'patch':[2,2,2],'d_model':12,'depth':1,'n_heads':2,'mlp_ratio':2,'use_amp':False,'grad_checkpoint':False,'token_mode':'per_patch'}
        source=original('ebcc218','Fall26/vit/model.py')
        coords=tuple(v.numpy() for v in (geo.r,geo.theta,geo.phi))
        model=source['build_model'](cfg,coords).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.pt';torch.save({'model':model.state_dict(),'config':cfg},path)
            adapter=ViT(path,geo,'cpu',{}).eval()
            history=torch.rand(1,1,8,*geo.shape)
            norm=Decode(geo)
            with torch.inference_mode():
                state=norm(history[:,-1]);expected=norm.inverse(state+model(state))
                actual=adapter.step(history)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            validate_output(actual,history,geo)

    @unittest.skipUnless(REFERENCE.is_dir(),'Pinned private source unavailable in public CI')
    def test_tron_architecture_matches_pinned_state_dict_and_forward(self):
        from grmhd_validation.adapters._tron_model import TRON
        source=original('2667a5f','Fall26/tron/model.py')
        # The minimum padded bottleneck needs 8x8x8; one CPU forward suffices.
        torch.manual_seed(42);reference=source['TRON'](in_c=8).eval()
        port=TRON(in_c=8).eval();port.load_state_dict(reference.state_dict(),strict=True)
        value=torch.randn(1,8,8,8,8)
        with torch.inference_mode():
            torch.testing.assert_close(port(value),reference(value),rtol=0,atol=0)

    @unittest.skipUnless(REFERENCE.is_dir(),'Pinned private source unavailable in public CI')
    def test_eden_kernels_match_originals_and_wrappers_preserve_physical_units(self):
        geo=geometry();random=np.random.default_rng(42)
        state=random.normal(size=(8,*geo.shape)).astype(np.float32);state[:2]=np.abs(state[:2])+.01
        mapping={'r':geo.r.numpy(),'h':geo.theta.numpy(),'ph':geo.phi.numpy(),'gcov':geo.gcov.numpy(),'gcon':geo.gcon.numpy(),
                 'dx':geo.dx.numpy(),'dt_dump':geo.dt_dump.numpy(),'dt_dump_exact':geo.dt_dump.numpy(),'spin':geo.spin.numpy()}
        cases=[('claude','forecast_one_step.py',_eden_claude,Claude,'forecast'),('gpt','predict_one_step.py',_eden_gpt,GPT,'predict_next'),('gemini','predict_step.py',_eden_gemini,Gemini,'predict_next')]
        for agent,file,module,wrapper,function in cases:
            path=f'eden_rc300_pilot/benchmark_pilot/rc300_packets/start_11672_{agent}/submission/code/{file}'
            source=original('f8cd379',path)
            # Source hard-codes rc300 output shape: parameterize only that guard,
            # never its numerical kernels, for a small independent oracle.
            source['SHAPE']=state.shape
            if agent=='gpt':
                tree=ast.parse(subprocess.check_output(['git','-C',REF,'show','f8cd379:'+path],text=True))
                node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
                for n in ast.walk(node):
                    if isinstance(n,ast.Tuple) and [getattr(v,'value',None) for v in n.elts]==[8,112,48,96]:
                        n.elts=[ast.Constant(v) for v in state.shape]
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),path,'exec'),source)
            args=(state,mapping,100.) if agent=='claude' else (state,mapping)
            expected=source[function](*args)
            actual=getattr(module,function)(*args)
            np.testing.assert_array_equal(actual,expected)
            adapter=wrapper(None,geo,'cpu',{})
            history=torch.from_numpy(state)[None,None]
            np.testing.assert_array_equal(adapter.step(history)[0].numpy(),expected)
            self.assertIn('known_prior_data_exposure',adapter.provenance)

    def test_eden_rejects_unsupported_geometry(self):
        geo=geometry((8,6,4));geo.phi[0,0,1]+=.01
        with self.assertRaisesRegex(ValueError,'uniform'):Gemini(None,geo,'cpu',{})

    def test_signed_colourbar_ticks_are_two_decades_apart_plus_zero(self):
        from grmhd_validation.analyses._figures import color_map
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure();FigureCanvasAgg(fig);ax=fig.subplots()
        image=np.linspace(-1,1,400).reshape(20,20)*np.logspace(-8,0,20)[None,:]
        color_map(fig,ax,image,'Signed bias','Density',error_scale='signed')
        ticks=fig.axes[1].get_yticks();self.assertIn(0.,ticks)
        positives=ticks[ticks>0]
        if len(positives)>1:np.testing.assert_allclose(np.diff(np.log10(positives)),2.)
